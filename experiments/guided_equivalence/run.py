"""Paid library construction, scoring and separate full-function audit."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair
from loguru import logger
from scipy.stats import t
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from okapi.globals import set_backend, set_postprocessing_function
from okapi.guided import correlation_templates

FAMILIES = ("two_pathways", "correlated")
SIZES = {"fit": 4096, "validation": 256, "iid": 4096, "shift": 4096}


def draw(family, bank, split):
    native, shifted = _scenario_pair(family)
    scenario = shifted if split == "shift" else native
    family_id = FAMILIES.index(family)
    rng = np.random.default_rng([bank, family_id, tuple(SIZES).index(split)])
    x, y, _ = scenario._generate(SIZES[split], rng)
    permutation = np.random.default_rng([bank, family_id, 710]).permutation(x.shape[1])
    return {f"p{j}": x[:, column, 1:2] for j, column in enumerate(permutation)}, y


def metric(name, pred, y):
    if name == "brier":
        return float(np.mean((pred.ravel()-y)**2))
    return float(average_precision_score(y, pred.ravel()))


def select(values, y, objective, deduplicate):
    start = perf_counter()
    trees = correlation_templates(values, deduplicate=deduplicate)
    build_s = perf_counter()-start
    start = perf_counter()
    scores = [metric(objective, tree.predict(), y) for tree in trees]
    idx = int(np.argmin(scores) if objective == "brier" else np.argmax(scores))
    score_s = perf_counter()-start
    cost = dict(build_s=build_s, score_s=score_s, search_s=build_s+score_s, fit_calls=len(trees),
                fit_row_scores=len(trees)*len(y), selected_index=idx, selected_tree=repr(trees[idx]))
    return trees, trees[idx], scores[idx], cost


def audit_functions(trees, datasets):
    digests = set()
    start = perf_counter()
    for tree in trees:
        digest = hashlib.sha256()
        for values, _ in datasets.values():
            scores = tree.do_pred_on_another_tensors(current_tensors=values)
            digest.update(np.ascontiguousarray(scores).tobytes())
        digests.add(digest.digest())
    return digests, perf_counter()-start


def run_cell(bank, family, index):
    fit = draw(family, bank, "fit")
    winners, cost, libraries = {}, {}, {}
    for objective in ("brier", "ap") if index % 2 == 0 else ("ap", "brier"):
        order = (False, True) if (index + (objective == "ap")) % 2 == 0 else (True, False)
        for compressed in order:
            key = (objective, compressed)
            trees, winner, fit_value, budget = select(*fit, objective, compressed)
            winners[key] = winner
            cost[key] = dict(**budget, fit_value=fit_value, first=compressed == order[0])
            libraries[compressed] = trees  # Representative library for the diagnostic pass.
        assert repr(winners[objective, False]) == repr(winners[objective, True])
        assert cost[objective, False]["fit_value"] == cost[objective, True]["fit_value"]

    datasets = {"fit": fit, **{s: draw(family, bank, s) for s in SIZES if s != "fit"}}
    risks = []
    for split, (x, y) in datasets.items():
        for objective in ("brier", "ap"):
            outputs = []
            for compressed in (False, True):
                start = perf_counter()
                pred = winners[objective, compressed].do_pred_on_another_tensors(current_tensors=x)
                result = metric(objective, pred, y)
                elapsed = perf_counter()-start
                outputs.append(pred)
                risks.append(dict(bank=bank, family=family, split=split, objective=objective, compressed=compressed,
                                  metric=result, postselection_s=elapsed, n=len(y), **cost[objective, compressed]))
            np.testing.assert_array_equal(*outputs)

    certificates = []
    reference = None
    for compressed in (False, True):
        trees = libraries[compressed]
        signatures, elapsed = audit_functions(trees, datasets)
        if reference is not None and reference != signatures:
            raise AssertionError("Compressed library changed the joint set of function values")
        reference = signatures
        certificates.append(dict(bank=bank, family=family, compressed=compressed, templates=len(trees),
                                 distinct=len(signatures), signature=hashlib.sha256(b"".join(sorted(signatures))).hexdigest(),
                                 audit_calls=len(trees)*len(SIZES), audit_row_scores=len(trees)*sum(SIZES.values()),
                                 audit_s=elapsed))
    return risks, certificates


def contrasts(risks):
    timings = risks[risks.split == "fit"]
    rows = []
    for (family, objective), group in timings.groupby(["family", "objective"]):
        for metric_name in ("search_s", "build_s", "score_s", "fit_calls"):
            by_bank = group.pivot(index="bank", columns="compressed", values=metric_name)
            delta = by_bank[True]-by_bank[False]
            margin = t.ppf(.975, len(delta)-1)*delta.sem()
            rows.append(dict(family=family, objective=objective, metric=metric_name, n=len(delta),
                             full=by_bank[False].mean(), compact=by_bank[True].mean(),
                             ratio=by_bank[True].mean()/by_bank[False].mean(),
                             delta=delta.mean(), lo=delta.mean()-margin, hi=delta.mean()+margin))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/guided_equivalence/results"))
    parser.add_argument("--banks", type=int, nargs="+", default=list(range(27100, 27110)))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logger.disable("okapi")
    set_backend("numpy")
    set_postprocessing_function(lambda x: x)
    cpu = None
    if hasattr(os, "sched_getaffinity"):
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
    rows, audit = [], []
    with threadpool_limits(1):
        for family in FAMILIES:
            for index, bank in enumerate(args.banks):
                measurements, certificates = run_cell(bank, family, index)
                rows.extend(measurements)
                audit.extend(certificates)
    risks = pd.DataFrame(rows).assign(cpu=cpu)
    risks.to_csv(args.out / "runs.csv", index=False)
    pd.DataFrame(audit).to_csv(args.out / "functions.csv", index=False)
    contrasts(risks).to_csv(args.out / "paired.csv", index=False)


if __name__ == "__main__":
    main()
