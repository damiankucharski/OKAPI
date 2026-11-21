"""Compare two equal-size OKAPI gate palettes on anonymous binary providers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from fusionbench.methods.averaging import TopN
from fusionbench.methods.okapi import _identity
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from experiments.guided import run as guided
from okapi.globals import set_backend, set_postprocessing_function
from okapi.guided import _average_link_cuts, _correlation_distances
from okapi.node import ValueNode
from okapi.operators import LOGIT_MEAN, MAX, MEAN, MIN, PROBABILISTIC_XOR
from okapi.tree import Tree

FAMILIES = ("parity", "correlated")
STARTS = (311, 751)
GP_CALLS = 160


def losses(pred, y):
    p = np.asarray(pred).ravel()
    return float(np.mean((p - y) ** 2)), float(np.mean((p > 0.5) == y))


def group_root(values, keys, members, pool):
    leaves = [ValueNode(None, values[keys[j]], keys[j]) for j in members]
    root, children = leaves[0], leaves[1:]
    op = MEAN(children) if pool == "mean" else LOGIT_MEAN(children, temperature=len(leaves), shift=0.0)
    root.add_child(op)
    return root


def gate_tree(values, keys, groups, pool, gate):
    left, right = [group_root(values, keys, members, pool) for members in groups]
    node = gate([right], temperature=1.0, shift=0.0) if gate is LOGIT_MEAN else gate([right])
    left.add_child(node)
    return Tree(left)


def gate_search(x, y, family, bank, start_seed, with_xor):
    begin = perf_counter()
    values = guided.providers(x)
    keys = list(values)
    cut = _average_link_cuts(_correlation_distances(x[:, :, 1]))[2]
    grouped = start_seed == -1
    assignment = np.arange(len(keys))
    if not grouped:
        shuffle_seed = np.random.SeedSequence([bank, guided.AVAILABLE_FAMILIES.index(family), start_seed, 717]).generate_state(1)[0]
        assignment = np.random.default_rng(shuffle_seed).permutation(len(keys))
    groups = [[int(assignment[j]) for j in members] for members in cut]
    grouping_s = perf_counter() - begin
    begin_build = perf_counter()
    gates = (("min", MIN), ("max", MAX), ("mean", MEAN), ("xor", PROBABILISTIC_XOR) if with_xor else ("logit_mean", LOGIT_MEAN))
    candidates = [(pool, name, gate_tree(values, keys, groups, pool, op)) for pool in ("mean", "logit_sum") for name, op in gates]
    build_s = perf_counter() - begin_build
    begin_score = perf_counter()
    scores = [losses(tree.predict(), y)[0] for _, _, tree in candidates]
    best = int(np.argmin(scores))
    score_s = perf_counter() - begin_score
    pool, gate, tree = candidates[best]
    method = ("grouped" if grouped else "shuffled") + ("_xor" if with_xor else "_substitution")
    key = dict(family=family, bank=bank, start=start_seed, method=method)
    budget = dict(
        unique_fit_labels=len(y),
        fit_tree_calls=8,
        fit_tree_row_scores=8 * len(y),
        grouping_s=grouping_s,
        build_s=build_s,
        score_s=score_s,
        fit_seconds=perf_counter() - begin,
        selected_index=best,
        selected_pool=pool,
        selected_gate=gate,
        selected_fit_brier=scores[best],
        groups=json.dumps(groups),
        nodes=tree.nodes_count,
        tree=repr(tree),
        num_providers=len(keys),
    )
    records = [dict(**key, candidate=i, pool=p, gate=g, fit_brier=scores[i], selected=i == best) for i, (p, g, _) in enumerate(candidates)]
    return method, tree, budget, records


def native_control(x, y, seed):
    begin = perf_counter()
    with TemporaryDirectory() as tmp:
        paths = []
        for name, scores in guided.providers(x).items():
            path = Path(tmp) / name
            np.save(path, scores)
            paths.append(path)
        yfile = Path(tmp) / "labels.npy"
        np.save(yfile, y)
        # The observer terminates at its last checkpoint, even mid-batch.
        # Use the published implementation and restore its schedule afterwards.
        with patch.object(guided, "BUDGETS", (GP_CALLS,)):
            snapshots, _ = guided.trial(paths, yfile, seed, "native", None)
        point = snapshots[GP_CALLS]
    assert point.budget == GP_CALLS
    return point.model, dict(
        unique_fit_labels=len(y),
        fit_tree_calls=point.budget,
        fit_tree_row_scores=point.budget * len(y),
        fit_seconds=perf_counter() - begin,
        nodes=point.model.nodes_count,
        tree=repr(point.model),
        incumbent_call=point.fit_call,
        selected_fit_brier=point.timing["tracked_fit_brier"],
        num_providers=x.shape[1],
    )


def fit_all(banks):
    fitted, candidates = {}, []
    for family in FAMILIES:
        for bank in banks:
            x, y = guided.data(family, bank, "fit")
            m = x.shape[1]
            models = []
            fitted[family, bank] = models
            models.append(("mean", -1, None, dict(fit_seconds=0.0, unique_fit_labels=0, fit_tree_calls=0, fit_tree_row_scores=0, num_providers=m)))
            for method in ("topn", "stack"):
                begin = perf_counter()
                estimator = TopN(metric="brier") if method == "topn" else StackingLogit(c_value=1.0)
                estimator.fit(x, y, task="binary", n_classes=2)
                cost = dict(fit_seconds=perf_counter() - begin, unique_fit_labels=len(y), fit_tree_calls=0, fit_tree_row_scores=0, num_providers=m)
                if method == "topn":
                    cost.update(
                        n_selected=estimator.n_selected_,
                        ranking=json.dumps(estimator.ranking_.tolist()),
                        prefix_scores=json.dumps(estimator.prefix_scores_.tolist()),
                        individual_scores=m,
                        prefix_count=m,
                        reference_row_scores=2 * m * len(y),
                    )
                else:
                    cost.update(solver_iters=int(estimator.model[-1].n_iter_.sum()))
                models.append((method, -1, estimator, cost))

            for i, seed in enumerate((-1, *STARTS)):
                for with_xor in (False, True) if (bank + i) % 2 == 0 else (True, False):
                    method, tree, cost, records = gate_search(x, y, family, bank, seed, with_xor)
                    candidates.extend(records)
                    models.append((method, seed, tree, cost))
                if seed != -1:
                    native, cost = native_control(x, y, seed)
                    models.append(("native_160", seed, native, cost))
    return fitted, candidates


def evaluate(models):
    rows, choices = [], []
    for (family, bank), frozen in models.items():
        for method, seed, _, budget in frozen:
            choices.append(dict(family=family, bank=bank, method=method, start=seed, validation_choices=0, **budget))
        for split in guided.SIZES:
            x, y = guided.data(family, bank, split)
            for method, seed, model, budget in frozen:
                begin = perf_counter()
                if method == "mean":
                    pred = x[:, :, 1].mean(axis=1)
                elif method in ("topn", "stack"):
                    pred = model.predict(x)[:, 1]
                else:
                    pred = model.do_pred_on_another_tensors(current_tensors=guided.providers(x))
                inference_s = perf_counter() - begin
                brier, accuracy = losses(pred, y)
                if split == "fit" and "selected_fit_brier" in budget:
                    np.testing.assert_allclose(brier, budget["selected_fit_brier"], rtol=0, atol=2e-15)
                rows.append(
                    dict(
                        family=family,
                        bank=bank,
                        method=method,
                        start=seed,
                        split=split,
                        n=len(y),
                        split_calls=1,
                        brier=brier,
                        accuracy=accuracy,
                        predict_seconds=inference_s,
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(choices)


def summarize(frame):
    rows = []
    for (family, split), part in frame.groupby(["family", "split"]):
        for metric in ("brier", "accuracy"):
            cells = part.pivot_table(index="bank", columns="method", values=metric)
            for arm in ("grouped_xor", "shuffled_xor", "grouped_substitution"):
                for reference in cells.columns.drop(arm):
                    delta = cells[arm] - cells[reference]
                    margin = t.ppf(0.975, len(delta) - 1) * delta.sem() if len(delta) > 1 else np.nan
                    rows.append(
                        dict(
                            family=family,
                            split=split,
                            metric=metric,
                            arm=arm,
                            reference=reference,
                            n=len(delta),
                            mean=cells[arm].mean(),
                            reference_mean=cells[reference].mean(),
                            delta=delta.mean(),
                            lo=delta.mean() - margin,
                            hi=delta.mean() + margin,
                            lower_banks=int((delta < 0).sum()),
                            higher_banks=int((delta > 0).sum()),
                        )
                    )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/parity_operator/results"))
    parser.add_argument("--banks", nargs="+", type=int, default=list(range(30400, 30412)))
    args = parser.parse_args()
    logger.disable("okapi")
    set_backend("numpy")
    set_postprocessing_function(_identity)
    cpu = None
    if hasattr(os, "sched_getaffinity"):
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
    with threadpool_limits(limits=1):
        models, candidates = fit_all(args.banks)
        results, choices = evaluate(models)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "runs.csv", index=False)
    choices.assign(cpu=cpu).to_csv(args.out / "choices.csv", index=False)
    pd.DataFrame(candidates).to_csv(args.out / "candidates.csv", index=False)
    summarize(results).to_csv(args.out / "paired.csv", index=False)


if __name__ == "__main__":
    main()
