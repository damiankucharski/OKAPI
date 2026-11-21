"""Preregistered, evaluation-capped OKAPI island comparison."""

import argparse
import itertools
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair, _tensor_map
from fusionbench.methods.okapi import OKAPIMethod, _identity
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from okapi.islands import migrate_ring
from okapi.okapi import Okapi
from okapi.pareto import minimize
from okapi.population import choose_pareto_then_proximity

SCENARIOS = ("two_pathways", "correlated")
ARMS = ("ring", "isolated", "panmixia")


def brier(p, y):
    return float(np.mean((p.ravel() - y.ravel()) ** 2))


def split(generator, seed, split_id, count):
    x, y, _ = generator._generate(count, np.random.default_rng([seed, split_id]))
    return x, y


class PartialGeneration(Exception):
    def __init__(self, trees, scores):
        self.trees, self.scores = trees, scores


class CappedOkapi(Okapi):
    calls = 0
    allowance = 120

    def _calculate_fitnesses(self, trees=None):
        batch = self.population if trees is None else trees
        remaining = self.allowance - self.calls
        prefix = batch[:remaining]
        scores = super()._calculate_fitnesses(prefix)
        self.calls += len(prefix)
        if len(prefix) < len(batch):
            raise PartialGeneration(prefix, scores.round(3))
        return scores

    def advance(self):
        if self.calls == self.allowance:
            return
        parents, fitnesses = self.population, self.fitnesses
        try:
            self.run_iteration()
        except PartialGeneration as tail:
            # In reuse mode only novel offspring reach the final scorer.
            assert not {repr(tree) for tree in parents}.intersection(map(repr, tail.trees))
            self.population, self.fitnesses = choose_pareto_then_proximity(
                parents + tail.trees, np.concatenate((fitnesses, tail.scores)), self.objectives,
                self.population_size, self.minimize_node_count,
            )
            self.additional_population = []


def initialize(x, y, seed, panmixia):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        paths = []
        for key, values in _tensor_map(x).items():
            path = root / key
            np.save(path, values)
            paths.append(path)
        np.save(root / "labels.npy", y)
        engines = []
        adapter = OKAPIMethod(n_seeds=3, max_nodes=15)
        try:
            for i in range(4):
                optimizer_seed = int(np.random.SeedSequence([seed, i, 77]).generate_state(1)[0])
                gp = CappedOkapi(paths, root / "labels.npy", population_size=3, population_multiplier=2,
                                 tournament_size=2, objective_functions=(brier,), objectives=(minimize,),
                                 backend="numpy", seed=optimizer_seed, postprocessing_function=_identity,
                                 max_nodes=15, reuse_survivor_fitness=True)
                adapter._seed_population(gp, np.random.default_rng(optimizer_seed))
                engines.append(gp)
        finally:
            adapter._tmp.cleanup()
        if panmixia:
            union = [tree for engine in engines for tree in engine.population]
            merged = engines[0]
            merged.population = union
            merged.population_size = len(union)
            merged.allowance = 480
            engines = [merged]
        for engine in engines:
            engine.fitnesses = engine._calculate_fitnesses(engine.population).round(3)
        return engines


def snapshot(engines, elapsed, generation, migrations):
    return dict(time=elapsed, generation=generation, calls=sum(engine.calls for engine in engines),
                migrations=migrations.copy(), population=[tree for engine in engines for tree in engine.population],
                scores=np.concatenate([engine.fitnesses for engine in engines])[:, 0])


def search(arm, x, y, seed, period):
    start = perf_counter()
    engines = initialize(x, y, seed, arm == "panmixia")
    duration = perf_counter() - start
    migrations = dict(attempts=0, assessments=0, admissions=0)
    history = [snapshot(engines, duration, 0, migrations)]
    for generation in range(1, 1001):
        start = perf_counter()
        for engine in engines:
            engine.advance()
        if arm == "ring" and generation % period == 0 and all(e.calls < e.allowance for e in engines):
            events = migrate_ring(engines)
            migrations["attempts"] += len(events)
            migrations["assessments"] += sum(event.evaluated for event in events)
            migrations["admissions"] += sum(event.admitted for event in events)
        duration += perf_counter() - start
        history.append(snapshot(engines, duration, generation, migrations))
        if all(e.calls == e.allowance for e in engines):
            return history
    raise RuntimeError("No progress within 1,000 generations")


def select_candidate(checkpoint, validation):
    x, y = validation
    start = perf_counter()
    unique = {}
    for idx in np.argsort(checkpoint["scores"], kind="stable"):
        tree = checkpoint["population"][idx]
        unique.setdefault(repr(tree), tree)
        if len(unique) == 4:
            break
    assert len(unique) == 4
    tensors = _tensor_map(x)
    trees = list(unique.values())
    losses = [brier(tree.do_pred_on_another_tensors(current_tensors=tensors), y) for tree in trees]
    winner = int(np.argmin(losses))
    return trees[winner], losses[winner], perf_counter() - start


def diversity(snapshot_, fit_x):
    tensors = _tensor_map(fit_x[:32])
    vectors = {np.round(tree.do_pred_on_another_tensors(current_tensors=tensors).ravel(), 10).tobytes()
               for tree in snapshot_["population"]}
    return len(vectors)


def development():
    rows = []
    for scenario in SCENARIOS:
        source, _ = _scenario_pair(scenario)
        for seed in range(21100, 21104):
            fit = split(source, seed, 0, 384)
            val = split(source, seed, 1, 256)
            for period in (2, 5):
                end = search("ring", *fit, seed, period)[-1]
                _, loss, validation_s = select_candidate(end, val)
                rows.append(dict(scenario=scenario, seed=seed, period=period, validation_brier=loss,
                                 search_s=end["time"], validation_s=validation_s, fit_calls=end["calls"],
                                 validation_calls=4, **end["migrations"]))
    frame = pd.DataFrame(rows)
    scores = frame.groupby("period").validation_brier.mean()
    return frame, 2 if scores[2] < scores[5] else 5


def confirmation(scenario, seed, period):
    source, shifted = _scenario_pair(scenario)
    fit = split(source, seed, 0, 384)
    val = split(source, seed, 1, 256)
    order = list(itertools.permutations(ARMS))[(seed - 21200) % 6]
    curves = {arm: search(arm, *fit, seed, period) for arm in order}
    horizon = min(curve[-1]["time"] for curve in curves.values())
    winners = []
    for arm, curve in curves.items():
        at_time = max(i for i, pt in enumerate(curve) if pt["time"] <= horizon)
        for budget, step in (("evaluations", len(curve) - 1), ("time", at_time)):
            checkpoint = curve[step]
            tree, val_loss, validation_s = select_candidate(checkpoint, val)
            next_gap = 0.0 if step + 1 == len(curve) else curve[step + 1]["time"] - checkpoint["time"]
            cost = dict(arm=arm, budget=budget, fit_calls=checkpoint["calls"], selection_calls=4,
                        rounds=step, search_s=checkpoint["time"], validation_s=validation_s,
                        common_time_s=horizon, next_checkpoint_gap_s=next_gap, selected_nodes=tree.nodes_count,
                        distinct_fit_vectors=diversity(checkpoint, fit[0]), validation_brier=val_loss, **checkpoint["migrations"])
            winners.append((lambda x, t=tree: t.do_pred_on_another_tensors(current_tensors=_tensor_map(x)), cost))
    start = perf_counter()
    stack = StackingLogit().fit(np.concatenate([fit[0], val[0]]), np.concatenate([fit[1], val[1]]), task="binary", n_classes=2)
    stack_s = perf_counter()-start
    for arm, model, elapsed in (("mean", lambda x: x[:, :, 1].mean(axis=1), 0.0), ("stack", lambda x: stack.predict(x)[:, 1], stack_s)):
        winners.append((model, dict(arm=arm, budget="baseline", search_s=elapsed, validation_s=0., fit_calls=0, selection_calls=0)))
    rows = []
    for domain, generator, index in (("iid", source, 2), ("shift", shifted, 3)):
        x, y = split(generator, seed, index, 4096)
        for predict, cost in winners:
            start = perf_counter()
            p = predict(x).ravel()
            infer_time = perf_counter() - start
            rows.append(dict(scenario=scenario, seed=seed, period=period, domain=domain,
                             brier=brier(p, y), accuracy=np.mean((p >= .5) == y.ravel()), infer_s=infer_time, **cost))
    return rows


def contrasts(frame):
    out = []
    for (scenario, budget, domain), cell in frame[frame.budget != "baseline"].groupby(["scenario", "budget", "domain"]):
        p = cell.pivot(index="seed", columns="arm", values="brier")
        for control in ("isolated", "panmixia"):
            delta = p.ring - p[control]
            margin = t.ppf(.975, len(delta) - 1) * delta.sem()
            out.append(dict(scenario=scenario, budget=budget, domain=domain, control=control, banks=len(delta),
                            difference=delta.mean(), lower=delta.mean()-margin, upper=delta.mean()+margin, wins=int((delta < 0).sum())))
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "confirmation"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(21200, 21212)))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    logger.disable("okapi")
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with threadpool_limits(1):
        if args.stage == "development":
            results, interval = development()
            results.to_csv(args.out / "development.csv", index=False)
            (args.out / "chosen.json").write_text(json.dumps({"period": interval}) + "\n")
        else:
            interval = json.loads((args.out / "chosen.json").read_text())["period"]
            records = [record for name in SCENARIOS for seed in args.seeds for record in confirmation(name, seed, interval)]
            frame = pd.DataFrame(records)
            frame.to_csv(args.out / "confirmation.csv", index=False)
            contrasts(frame).to_csv(args.out / "contrasts.csv", index=False)
