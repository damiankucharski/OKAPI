"""Paired CPU benchmark for reusing unchanged OKAPI survivor fitness."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair, _tensor_map
from fusionbench.methods.okapi import OKAPIMethod, _brier_objective, _identity
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from okapi.okapi import Okapi
from okapi.pareto import maximize

SCENARIOS = ("two_pathways", "correlated")
GENS, FIT, VALID, TEST, SELECT = 12, 1024, 256, 4096, 6


class BudgetExhausted(Exception):
    """Raised before evaluating a group of candidates that would exceed the cap."""


class MeteredOkapi(Okapi):
    calls = 0
    limit = None

    def _calculate_fitnesses(self, trees=None):
        size = len(self.population if trees is None else trees)
        if self.limit is not None and self.calls + size > self.limit:
            raise BudgetExhausted
        result = super()._calculate_fitnesses(trees)
        self.calls += size
        return result


def split(scenario, n, seed, stream):
    preds, labels, _ = scenario._generate(n, np.random.default_rng([seed, stream]))
    return preds, labels


def setup(paths, labels_file, search_seed, reuse, adapter):
    start = perf_counter()
    gp = MeteredOkapi(paths, labels_file, population_size=len(paths), population_multiplier=2,
                      tournament_size=3, objective_functions=(_brier_objective,), objectives=(maximize,),
                      backend="numpy", seed=search_seed, postprocessing_function=_identity,
                      max_nodes=15, reuse_survivor_fitness=reuse)
    adapter._seed_population(gp, np.random.default_rng(search_seed))
    return gp, perf_counter() - start


def fingerprint(gp):
    """Hash generation boundary state and functional outputs outside the timed path."""
    digest = hashlib.sha256()
    digest.update(pickle.dumps(gp.rng.get_state(), protocol=4))
    assert not gp.additional_population
    for tree, fitness in zip(gp.population, gp.fitnesses, strict=True):
        nodes = tree.root.get_nodes()
        assert all(node.evaluation is None for node in tree.value_nodes)
        # repr alone rounds weights. Preserve child order, exact parameter values
        # and provider IDs, and independently check the actual fit predictions.
        descriptors = [(type(node).__name__, len(node.children), getattr(node, "id", None),
                        getattr(node, "_weights", None), getattr(node, "threshold", None),
                        getattr(node, "temperature", None), getattr(node, "shift", None),
                        getattr(node, "close", None)) for node in nodes]
        digest.update(pickle.dumps((descriptors, tree.mutation_chance), protocol=4))
        digest.update(fitness.tobytes())
        digest.update(tree.copy().predict().tobytes())
    return digest.hexdigest()


def winner(population, fitnesses, heldout):
    x, y = heldout
    shortlist = np.argsort(-fitnesses[:, 0], kind="stable")[:SELECT]
    assert len(shortlist) == SELECT
    predictions = _tensor_map(x)
    candidates = [population[i] for i in shortlist]
    losses = [1 - _brier_objective(tree.do_pred_on_another_tensors(current_tensors=predictions), y) for tree in candidates]
    return candidates[int(np.argmin(losses))]


def run_cell(name, seed, search_seed, reverse_order):
    native, shifted = _scenario_pair(name)
    x, y = split(native, FIT, seed, 0)
    heldout = split(native, VALID, seed, 1)
    order = ("reuse", "stock") if reverse_order else ("stock", "reuse")
    with TemporaryDirectory() as tmp:
        io_start = perf_counter()
        paths = []
        for provider_id, scores in _tensor_map(x).items():
            path = Path(tmp) / provider_id
            np.save(path, scores)
            paths.append(path)
        labels_file = Path(tmp) / "y.npy"
        np.save(labels_file, y)
        io_seconds = perf_counter() - io_start
        adapter = OKAPIMethod(n_seeds=6, max_nodes=15)
        engines, seconds = {}, {}
        try:
            for method in order:
                engines[method], seconds[method] = setup(paths, labels_file, search_seed, method == "reuse", adapter)
        finally:
            adapter._tmp.cleanup()
        trace = []
        for generation in range(1, GENS + 1):
            hashes = {}
            for method in order:
                start = perf_counter()
                gp = engines[method]
                gp.run_iteration()
                seconds[method] += perf_counter() - start
                hashes[method] = fingerprint(gp)
            if hashes["stock"] != hashes["reuse"]:
                raise AssertionError(f"Trajectory diverged at {name=} {seed=} {search_seed=} {generation=}")
            trace.append(dict(scenario=name, data_seed=seed, search_seed=search_seed,
                              generation=generation, **hashes,
                              stock_calls=engines["stock"].calls, reuse_calls=engines["reuse"].calls))

        candidates = {}
        for method in order:
            gp = engines[method]
            candidates[method] = (gp.population, gp.fitnesses, gp.calls, GENS, seconds[method])
        # Match the actual stock budget in this paired cell. Keep the last
        # completed population; an over-budget generational proposal is never scored.
        stock_calls = engines["stock"].calls
        gp = engines["reuse"]
        gp.limit = stock_calls
        start = perf_counter()
        completed = GENS
        while completed < 4 * GENS:
            population, fitnesses = gp.population, gp.fitnesses
            try:
                gp.run_iteration()
            except BudgetExhausted:
                break
            completed += 1
        else:
            population, fitnesses = gp.population, gp.fitnesses
        candidates["reinvest"] = (population, fitnesses, gp.calls, completed, seconds["reuse"] + perf_counter() - start)

        # Freeze each individual validation choice before drawing either test sample.
        trained, cost = {}, {}
        for method, (population, fitnesses, calls, gens, search_s) in candidates.items():
            start = perf_counter()
            tree = winner(population, fitnesses, heldout)
            val_s = perf_counter() - start
            trained[method] = tree
            cost[method] = dict(generations=gens, cap=stock_calls, fit_calls=calls, fit_row_scores=calls * FIT,
                                validation_calls=SELECT, validation_row_scores=SELECT * VALID, nodes=tree.nodes_count,
                                data_io_s=io_seconds, search_s=search_s, validation_s=val_s,
                                fit_select_s=io_seconds + search_s + val_s)
        start = perf_counter()
        stack = StackingLogit().fit(np.concatenate([x, heldout[0]]), np.concatenate([y, heldout[1]]),
                                    task="binary", n_classes=2)
        stack_s = perf_counter() - start
        rows = []
        for domain, (test, labels) in (("iid", split(native, TEST, seed, 2)), ("shift", split(shifted, TEST, seed, 3))):
            reference = None
            for method in ("stock", "reuse", "reinvest", "mean", "stack"):
                start = perf_counter()
                if method in trained:
                    pred = trained[method].do_pred_on_another_tensors(current_tensors=_tensor_map(test)).ravel()
                    budget = cost[method]
                else:
                    pred = test[:, :, 1].mean(axis=1) if method == "mean" else stack.predict(test)[:, 1]
                    budget = {"fit_select_s": 0.0 if method == "mean" else stack_s}
                predict_s = perf_counter() - start
                if method == "stock":
                    reference = pred
                elif method == "reuse":
                    np.testing.assert_array_equal(reference, pred)
                rows.append(dict(scenario=name, data_seed=seed, search_seed=search_seed, method=method,
                                 domain=domain, brier=np.mean((pred - labels) ** 2),
                                 accuracy=np.mean((pred > .5) == labels), predict_s=predict_s,
                                 total_s=budget["fit_select_s"] + predict_s, **budget))
    return rows, trace


def paired_contrasts(metrics):
    rows = []
    for (name, domain), block in metrics.groupby(["scenario", "domain"]):
        for method in ("reuse", "reinvest"):
            for metric in ("brier", "accuracy", "fit_calls", "generations", "fit_select_s", "total_s"):
                means = block.pivot_table(index="data_seed", columns="method", values=metric)
                deltas = means[method] - means["stock"]
                margin = t.ppf(.975, len(deltas) - 1) * deltas.sem() if len(deltas) > 1 else np.nan
                rows.append(dict(scenario=name, domain=domain, method=method, metric=metric,
                                 n_data=len(deltas), stock=means["stock"].mean(), value=means[method].mean(),
                                 delta=deltas.mean(), lo=deltas.mean()-margin, hi=deltas.mean()+margin))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/survivor_reuse"))
    parser.add_argument("--data-seeds", type=int, nargs="+", default=list(range(8200, 8212)))
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[53, 97])
    args = parser.parse_args()
    if len(set(args.data_seeds)) != len(args.data_seeds) or len(set(args.search_seeds)) != len(args.search_seeds):
        parser.error("seeds must be distinct within each axis")
    logger.disable("okapi")
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(scenarios=SCENARIOS, data_seeds=args.data_seeds, search_seeds=args.search_seeds,
                  generations=GENS, fit=FIT, validation=VALID, test=TEST, shortlist=SELECT,
                  python=platform.python_version(), numpy=np.__version__)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows, traces = [], []
    with threadpool_limits(1):
        for name in SCENARIOS:
            for i, seed in enumerate(args.data_seeds):
                for j, search_seed in enumerate(args.search_seeds):
                    scores, trace = run_cell(name, seed, search_seed, (i + j) % 2 == 1)
                    rows.extend(scores)
                    traces.extend(trace)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(traces).to_csv(args.out / "trajectories.csv", index=False)
    paired_contrasts(metrics).to_csv(args.out / "paired.csv", index=False)


if __name__ == "__main__":
    main()
