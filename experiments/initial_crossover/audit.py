"""Exact before-mutation opportunities in the recorded eight-provider seed populations."""

import argparse
import gzip
import json
from fractions import Fraction
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import t

from experiments.donor_opportunity.native_kernel import child_distribution
from experiments.initial_crossover.pairs import child_tables, prepare, productive_by_target
from experiments.initial_crossover.tournament import tournament_weights
from fusionbench.eval.lattice_signature import lattice_signature, provider_signatures
from fusionbench.eval.seed_geometry_focal import data_for, integer_brier, public_seeds, target_signature
from okapi import Okapi
from okapi.globals import use_postprocessing_function
from okapi.operators import MAX, MIN

KEY = ["law", "data_seed", "method", "gp_seed"]
METHODS = ("branch", "chain", "stock")
NATIVE_PAIRS = ((0, 8), (8, 9), (17, 33), (37, 37))


def checked_geometry(population, masks):
    programs = [prepare(p, masks) for p in population]
    for i, j in NATIVE_PAIRS:
        native, _ = child_distribution(
            population[i], population[j], signature=lambda tree: lattice_signature(tree, masks)
        )
        assert native == child_tables(programs[i], programs[j])
    return productive_by_target(programs)


def weighted_chance(opportunities, weights):
    n = len(weights)
    tournament = uniform = Fraction(0)
    legal_pairs = selected_pairs = 0
    for i, j, chance in opportunities:
        orders = 1 if i == j else 2
        legal_pairs += orders
        weight = weights[i] * weights[j]
        selected_pairs += orders * int(weight > 0)
        tournament += orders * weight * chance
        uniform += orders * chance / n**2
    return dict(p_tournament=float(tournament), p_uniform=float(uniform), exact_tournament=str(tournament),
                exact_uniform=str(uniform), ordered_legal_pairs=legal_pairs, positive_weight_pairs=selected_pairs)


def run(source, output):
    start = perf_counter()
    logger.disable("okapi")
    with gzip.open(source / "focal-seeds.jsonl.gz", "rt") as inp:
        archives = [row for row in map(json.loads, inp) if row["width"] == 8 and row["law"] in ("branch", "chain")]
    recorded = pd.read_csv(source / "focal-runs.csv.gz", low_memory=False)
    selected = recorded.query("width == 8 and selector == 'fit' and law.isin(['branch', 'chain']) and method in @METHODS").set_index([*KEY, "checkpoint"])
    rows, first_geometry = [], {}
    with TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"p{i}.npy" for i in range(8)]
        labels_file = Path(tmp) / "labels.npy"
        for archive in archives:
            key = tuple(archive[k] for k in KEY)
            law, bank, method, seed = key
            roles, splits = data_for(8, bank, law)
            xfit, yfit = splits["fit"]
            for i, file in enumerate(paths):
                np.save(file, xfit[:, i:i+1])
            np.save(labels_file, yfit[:, None].astype(int))
            target = target_signature(8, roles, law)
            assert hex(target) == archive["target_hex"]
            fit_units = []

            def objective(pred, _):
                loss = integer_brier(pred.ravel(), yfit)
                fit_units.append(loss)
                return 1 - loss / (400 * len(yfit))

            with use_postprocessing_function(np.asarray):
                gp = Okapi(paths, labels_file, 8, 3, 5, backend="numpy", seed=seed, allowed_ops=(MIN, MAX),
                           minimize_node_count=True, objective_functions=(objective,), operator_point_mutation=False,
                           selection_fitness_decimals=3)
                public_seeds(gp, method, seed)
                fitness = gp._selection_fitnesses(gp.population)
            assert (len(gp.population), gp.tournament_size, gp.max_depth, gp.max_nodes) == (38, 5, None, None)
            masks = provider_signatures(gp.ids)
            sigs = [lattice_signature(p, masks) for p in gp.population]
            structures = tuple(repr(p.structural_key()) for p in gp.population)
            assert [hex(s) for s in sigs] == archive["singleton_signatures"] + archive["seed_signatures"]
            assert list(structures[8:]) == archive["seed_structures"]
            initial, final = selected.loc[(*key, "initial")], selected.loc[(*key, "1200")]
            assert min(fit_units) == initial.selected_fit_units
            assert hex(sigs[int(np.argmin(fit_units))]) == initial.selected_hex
            assert (target in sigs) == initial.true_available
            geometry_key = method, seed
            if geometry_key not in first_geometry:
                first_geometry[geometry_key] = structures, checked_geometry(gp.population, masks)
            prior, target_pairs = first_geometry[geometry_key]
            assert prior == structures
            weights = tournament_weights(fitness, gp.tournament_size, gp.optimal_point)
            case = weighted_chance(target_pairs.get(target, ()), weights)
            rows.append(dict(zip(KEY, key), initial_hit=target in sigs, hit_1200=bool(final.true_available), **case))
    assert len(rows) == 144 and len(first_geometry) == 6
    frame = pd.DataFrame(rows)
    groups, paired = [], []
    for (law, method), block in frame.groupby(["law", "method"]):
        for status, part in (("all", block), ("initial_hit", block[block.initial_hit]),
                             ("initial_absent", block[~block.initial_hit]),
                             ("late_hit", block[~block.initial_hit & block.hit_1200]),
                             ("never_hit", block[~block.hit_1200])):
            groups.append(dict(law=law, method=method, group=status, n=len(part), positive_tournament=int(part.p_tournament.gt(0).sum()),
                               uniform_only=int((part.p_uniform.gt(0) & part.p_tournament.eq(0)).sum()),
                               no_legal_pair=int(part.p_uniform.eq(0).sum()), mean_tournament=part.p_tournament.mean(),
                               mean_uniform=part.p_uniform.mean()))
    for law, metric in product(("branch", "chain"), ("p_tournament", "p_uniform")):
        bank_means = frame[frame.law.eq(law)].groupby(["data_seed", "method"])[metric].mean().unstack()
        difference = bank_means.branch - bank_means.chain
        margin = t.ppf(.975, len(difference)-1) * difference.sem()
        paired.append(dict(law=law, metric=metric, banks=len(difference), difference=difference.mean(),
                           lower=difference.mean()-margin, upper=difference.mean()+margin))
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "opportunities.csv", index=False)
    pd.DataFrame(groups).to_csv(output / "groups.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired.csv", index=False)
    summary = dict(populations=len(frame), objective_replays=len(frame)*38, new_training_runs=0,
                   native_conditional_pair_checks=len(first_geometry)*len(NATIVE_PAIRS),
                   seconds=perf_counter()-start)
    (output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
