"""Compare already-frozen Parity winners with the generator's privileged posterior."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair
from fusionbench.methods.okapi import _identity
from loguru import logger

from experiments.guided.run import AVAILABLE_FAMILIES, SIZES, data, providers
from experiments.parity_operator.run import gate_tree
from okapi.globals import set_backend, set_postprocessing_function
from okapi.operators import MAX, MEAN, MIN, PROBABILISTIC_XOR


def check_frozen_choices(directory):
    choices = pd.read_csv(directory / "choices.csv")
    scores = pd.read_csv(directory / "runs.csv")
    winners = choices.query("family == 'parity' and method == 'grouped_xor'")
    archived = scores.query("family == 'parity' and method == 'grouped_xor'").set_index(["bank", "split"])
    source, shifted = _scenario_pair("parity")
    gates = {"min": MIN, "max": MAX, "mean": MEAN, "xor": PROBABILISTIC_XOR}
    family = AVAILABLE_FAMILIES.index("parity")
    rows = []
    for winner in winners.itertuples(index=False):
        bank = winner.bank
        inputs, fit_y = data("parity", bank, "fit")
        tensors = providers(inputs)
        groups = json.loads(winner.groups)
        tree = gate_tree(tensors, list(tensors), groups, winner.selected_pool, gates[winner.selected_gate])
        ordered_ids = np.random.default_rng([bank, family, 402]).permutation(inputs.shape[1])
        recovered = {frozenset(ordered_ids[group]) for group in groups}
        actual = {frozenset(range(start, start + source.rep)) for start in range(0, inputs.shape[1], source.rep)}
        for index, (split, n) in enumerate(SIZES.items()):
            scenario = shifted if split == "shift" else source
            rng = np.random.default_rng([bank, family, index])
            ordered, y, latent = scenario._generate(n, rng)
            anonymous, observed_y = data("parity", bank, split)
            np.testing.assert_array_equal(ordered[:, ordered_ids], anonymous)
            np.testing.assert_array_equal(y, observed_y)
            p = tree.do_pred_on_another_tensors(current_tensors=providers(anonymous)).ravel()
            oracle = scenario.bayes_oracle(ordered, latent)[:, 1]
            loss = np.mean((p - y) ** 2)
            np.testing.assert_allclose(loss, archived.loc[(bank, split), "brier"], rtol=0, atol=2e-15)
            rows.append(
                dict(
                    bank=bank,
                    split=split,
                    n=n,
                    recovered_groups=recovered == actual,
                    source_prevalence=fit_y.mean(),
                    prevalence_brier=np.mean((fit_y.mean() - y) ** 2),
                    known_posterior_brier=np.mean((oracle - y) ** 2),
                    grouped_brier=loss,
                    max_abs_probability_gap=np.max(np.abs(p - oracle)),
                )
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    directory = parser.parse_args().results
    logger.disable("okapi")
    set_backend("numpy")
    set_postprocessing_function(_identity)
    check_frozen_choices(directory).to_csv(directory / "posterior.csv", index=False)
