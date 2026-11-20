"""Prospective survivor-reuse workload test. See survivor_workload.md."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair, _tensor_map
from fusionbench.methods.okapi import OKAPIMethod
from loguru import logger
from scipy.stats import t
from sklearn.metrics import average_precision_score
from survivor_reuse import SELECT, TEST, VALID, setup, split
from threadpoolctl import threadpool_limits

SIZES = (1024, 32768)
GENERATIONS = 6
SEEDS = range(9200, 9208)


def ap(predictions, labels):
    return average_precision_score(labels.ravel(), predictions.ravel())


def train_arm(train, valid, opt_seed, reuse, generations):
    x, y = train
    start = perf_counter()
    with TemporaryDirectory() as tmp:
        paths = []
        for provider_id, scores in _tensor_map(x).items():
            path = Path(tmp) / provider_id
            np.save(path, scores)
            paths.append(path)
        labels_file = Path(tmp) / "y.npy"
        np.save(labels_file, y)
        adapter = OKAPIMethod(n_seeds=6, max_nodes=15)
        try:
            gp, _ = setup(paths, labels_file, opt_seed, reuse, adapter)
        finally:
            adapter._tmp.cleanup()
        assert gp.fitnesses is None
        gp.objective_functions = (ap,)
        for _ in range(generations):
            gp.run_iteration()
    search_s = perf_counter() - start

    start = perf_counter()
    tensors = _tensor_map(valid[0])
    indices = np.argsort(-gp.fitnesses[:, 0], kind="stable")[:SELECT]
    assert len(indices) == SELECT
    candidates = [gp.population[i] for i in indices]
    scores = [ap(tree.do_pred_on_another_tensors(current_tensors=tensors), valid[1]) for tree in candidates]
    winner = candidates[int(np.argmax(scores))]
    validation_s = perf_counter() - start
    return winner, dict(fit_calls=gp.calls, validation_calls=SELECT, search_s=search_s,
                        validation_s=validation_s, fit_select_s=search_s + validation_s)


def run(seed, index, pin):
    opt_seed = 153 + index
    native, shifted = _scenario_pair("two_pathways")
    train_x, train_y = split(native, max(SIZES), seed, 0)
    valid = split(native, VALID, seed, 1)
    size_order = SIZES if index % 2 == 0 else SIZES[::-1]
    rows = []
    for size in size_order:
        # AB / BA balance within each size; AB / BA also differ at fixed seed.
        order = (False, True) if (index + SIZES.index(size)) % 2 == 0 else (True, False)
        winners, timings = {}, {}
        for reuse in order:
            winners[reuse], timings[reuse] = train_arm((train_x[:size], train_y[:size]), valid, opt_seed, reuse, GENERATIONS)
        for domain, (x, y) in (("iid", split(native, TEST, seed, 2)), ("shift", split(shifted, TEST, seed, 3))):
            predictions = {}
            for reuse in order:
                start = perf_counter()
                pred = winners[reuse].do_pred_on_another_tensors(current_tensors=_tensor_map(x))
                infer_s = perf_counter() - start
                predictions[reuse] = pred
                rows.append(dict(data_seed=seed, search_seed=opt_seed, fit_n=size, domain=domain,
                                 method="reuse" if reuse else "stock", first=reuse == order[0], cpu=pin,
                                 heldout_ap=ap(pred, y), predict_s=infer_s, **timings[reuse]))
            np.testing.assert_array_equal(predictions[False], predictions[True])
    return rows


def summary(frame):
    rows = []
    for size, group in frame.loc[frame.domain == "iid"].groupby("fit_n"):
        for metric in ("fit_select_s", "fit_calls"):
            pairs = group.pivot(index="data_seed", columns="method", values=metric)
            diff = pairs.reuse - pairs.stock
            margin = t.ppf(.975, len(diff) - 1) * diff.sem()
            rows.append(dict(fit_n=size, metric=metric, stock=pairs.stock.mean(), reuse=pairs.reuse.mean(),
                             difference=diff.mean(), lo=diff.mean()-margin, hi=diff.mean()+margin,
                             ratio=pairs.reuse.mean()/pairs.stock.mean()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/survivor_workload.csv"))
    args = parser.parse_args()
    logger.disable("okapi")
    pin = None
    if hasattr(os, "sched_getaffinity"):
        pin = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {pin})
    with threadpool_limits(limits=1):
        native, _ = _scenario_pair("two_pathways")
        for reuse in (False, True):
            train_arm(split(native, 64, 9199, 0), split(native, 64, 9199, 1), 152, reuse, 1)
        rows = []
        for index, seed in enumerate(SEEDS):
            rows.extend(run(seed, index, pin))
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(summary(frame).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
