"""Source prevalence and privileged population references on the fixed transfer splits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair

from experiments.guided.run import AVAILABLE_FAMILIES, SIZES, data


def run_bank(family, bank):
    source, shifted = _scenario_pair(family)
    _, fit_y = data(family, bank, "fit")
    prevalence = fit_y.mean()
    rows = []
    for index, (split, n) in enumerate(SIZES.items()):
        scenario = shifted if split == "shift" else source
        rng = np.random.default_rng([bank, AVAILABLE_FAMILIES.index(family), index])
        x, y, latent = scenario._generate(n, rng)
        anonymous, benchmark_y = data(family, bank, split)
        np.testing.assert_array_equal(benchmark_y, y)
        np.testing.assert_allclose(x[:, :, 1].mean(axis=1), anonymous[:, :, 1].mean(axis=1), rtol=0, atol=1e-15)
        oracle = scenario.bayes_oracle(x, latent)[:, 1]
        for method, pred in (("source_prevalence", np.full(n, prevalence)), ("known_posterior", oracle)):
            rows.append(dict(family=family, bank=bank, split=split, method=method, n=n, fit_prevalence=prevalence,
                             brier=np.mean((pred-y)**2), accuracy=np.mean((pred > .5) == y)))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for family in ("weakest_link", "parity") for bank in range(28100, 28112) for row in run_bank(family, bank)]
    pd.DataFrame(rows).to_csv(args.out, index=False)
