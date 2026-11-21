"""Equal-fitness-budget GP with correlation, shuffled, or singleton initializers."""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair
from fusionbench.methods.okapi import _brier_objective, _identity
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from okapi.guided import correlation_templates
from okapi.okapi import Okapi
from okapi.pareto import maximize
from okapi.population import choose_pareto_then_proximity

BUDGETS = (160, 640)
ARMS = ("native", "guided", "placebo")
FAMILIES = ("two_pathways", "correlated")
AVAILABLE_FAMILIES = (*FAMILIES, "weakest_link", "parity")
SIZES = {"fit": 384, "validation": 256, "iid": 4096, "shift": 4096}


class FinishedBudget(Exception):
    """Stop immediately after the final scored tree, including mid-generation."""


@dataclass
class Checkpoint:
    model: object
    fit_call: int
    stage: str
    budget: int
    elapsed: float
    search_generation: int
    template_count: int
    timing: dict = field(default_factory=dict)


class ObservedGP(Okapi):
    """Stock batches with exact full-precision incumbent observation and stopping."""

    def start_tracking(self, start):
        self.started = start
        self.calls = 0
        self.stage = "singleton"
        self.highest = -np.inf
        self.incumbent = None
        self.checkpoints = {}
        self.generation = 0
        self.template_count = 0
        self.timing = {}

    def snapshot(self):
        tree, call, stage = self.incumbent
        return Checkpoint(tree.copy(), call, stage, self.calls, perf_counter()-self.started,
                          self.generation, self.template_count, dict(self.timing, tracked_fit_brier=1-self.highest))

    def _calculate_fitnesses(self, trees=None):
        cohort = self.population if trees is None else trees
        results = []
        offset = 0
        while offset < len(cohort):
            boundary = next(b for b in BUDGETS if b > self.calls)
            n = min(len(cohort)-offset, boundary-self.calls)
            portion = cohort[offset:offset+n]
            fitness = super()._calculate_fitnesses(portion)
            best = int(np.argmax(fitness[:, 0]))
            if fitness[best, 0] > self.highest:
                self.highest = fitness[best, 0]
                self.incumbent = (portion[best], self.calls + best + 1, self.stage)
            results.append(fitness)
            self.calls += n
            offset += n
            if self.calls == boundary:
                self.checkpoints[boundary] = self.snapshot()
                if boundary == BUDGETS[-1]:
                    raise FinishedBudget
        return np.concatenate(results) if results else np.empty((0, 1))


def trial(paths, yfile, seed, arm, shuffle_seed):
    start = perf_counter()
    gp = ObservedGP(paths, yfile, len(paths), 2, 2, backend="numpy", seed=seed,
                    objective_functions=(_brier_objective,), objectives=(maximize,),
                    postprocessing_function=_identity, reuse_survivor_fitness=False)
    gp.start_tracking(start)
    singles = gp._calculate_fitnesses(gp.population)
    gp.fitnesses = singles.round(3)
    gp.timing["singleton_s"] = perf_counter() - start
    template_only = None
    if arm != "native":
        step = perf_counter()
        templates = correlation_templates(gp.train_tensors, shuffle_seed=shuffle_seed if arm == "placebo" else None)
        gp.timing["build_s"] = perf_counter() - step
        gp.template_count = len(templates)
        assert gp.template_count == 135
        gp.stage = "template"
        step = perf_counter()
        fits = gp._calculate_fitnesses(templates)
        gp.timing["template_score_s"] = perf_counter() - step
        best = int(np.argmax(fits[:, 0]))
        template_only = Checkpoint(templates[best].copy(), len(paths)+best+1, "template", gp.calls,
                                   perf_counter()-start, 0, len(templates), dict(gp.timing, tracked_fit_brier=1-fits[best, 0]))
        step = perf_counter()
        gp.population, gp.fitnesses = choose_pareto_then_proximity(
            gp.population + templates, np.vstack([singles, fits]).round(3), gp.objectives, len(paths), gp.minimize_node_count)
        gp.timing["initial_survival_s"] = perf_counter() - step
    gp.stage = "evolution"
    try:
        while True:
            gp.generation += 1
            gp.run_iteration()
    except FinishedBudget:
        pass
    assert gp.calls == BUDGETS[-1] and tuple(gp.checkpoints) == BUDGETS
    return gp.checkpoints, template_only


def data(name, bank, split):
    native, shifted = _scenario_pair(name)
    family = AVAILABLE_FAMILIES.index(name)
    seed = np.random.default_rng([bank, family, tuple(SIZES).index(split)])
    scenario = shifted if split == "shift" else native
    x, y, _ = scenario._generate(SIZES[split], seed)
    order = np.random.default_rng([bank, family, 402]).permutation(x.shape[1])
    return x[:, order], y


def providers(x):
    return {f"p{j}.npy": x[:, j, 1:2] for j in range(x.shape[1])}


def freeze_all(banks, optimizers, families=FAMILIES):
    """No test or validation data is generated during this phase."""
    frozen = []
    orders = list(itertools.permutations(ARMS))
    for name in families:
        for bank in banks:
            x, y = data(name, bank, "fit")
            start = perf_counter()
            stack = StackingLogit(c_value=1.).fit(x, y, task="binary", n_classes=2)
            stack_s = perf_counter()-start
            for arm, model, elapsed in (("mean", None, 0.), ("stack", stack, stack_s)):
                frozen.append((name, bank, -1, arm, Checkpoint(model, 0, "control", 0, elapsed, 0, 0)))
            for search_index, seed in enumerate(optimizers):
                with TemporaryDirectory() as tmp:
                    start = perf_counter()
                    paths = []
                    for identifier, pred in providers(x).items():
                        path = Path(tmp) / identifier
                        np.save(path, pred)
                        paths.append(path)
                    yfile = Path(tmp) / "y.npy"
                    np.save(yfile, y)
                    io_s = perf_counter()-start
                    shuffle_seed = np.random.SeedSequence([bank, AVAILABLE_FAMILIES.index(name), seed, 717]).generate_state(1)[0]
                    for arm in orders[(bank - 25100 + 3*search_index) % len(orders)]:
                        snapshots, initial = trial(paths, yfile, seed, arm, int(shuffle_seed))
                        for point in snapshots.values():
                            point.timing["io_s"] = io_s
                            frozen.append((name, bank, seed, arm, point))
                        if initial is not None:
                            initial.timing["io_s"] = io_s
                            frozen.append((name, bank, seed, arm + "_library", initial))
    return frozen


def evaluate(frozen):
    rows = []
    for (name, bank), items in itertools.groupby(frozen, lambda item: item[:2]):
        items = list(items)
        for split in SIZES:
            x, y = data(name, bank, split)
            tensors = providers(x)
            for _, _, seed, arm, point in items:
                start = perf_counter()
                if arm == "mean":
                    pred = x[:, :, 1].mean(axis=1)
                elif arm == "stack":
                    pred = point.model.predict(x)[:, 1]
                else:
                    pred = point.model.do_pred_on_another_tensors(current_tensors=tensors).ravel()
                elapsed = perf_counter() - start
                tree = point.model if point.stage != "control" else None
                brier = np.mean((pred-y)**2)
                if tree is not None and split == "fit":
                    np.testing.assert_allclose(brier, point.timing["tracked_fit_brier"], atol=2e-15, rtol=0)
                rows.append(dict(family=name, bank=bank, gp_seed=seed, arm=arm, split=split, n=len(y), budget=point.budget,
                                 unique_fit_labels=0 if arm == "mean" else SIZES["fit"], fit_row_scores=point.budget*SIZES["fit"],
                                 validation_choices=0, split_score_calls=1,
                                 incumbent_call=point.fit_call, incumbent_stage=point.stage, generation_started=point.search_generation,
                                 template_count=point.template_count, nodes=tree.nodes_count if tree is not None else None,
                                 tree=repr(tree) if tree is not None else "", fit_seconds=point.elapsed + point.timing.get("io_s", 0),
                                 score_seconds=elapsed, brier=brier, accuracy=np.mean((pred>.5)==y), **point.timing))
    return pd.DataFrame(rows)


def stats(frame):
    rows = []
    for (name, split), sub in frame.groupby(["family", "split"]):
        for budget in BUDGETS:
            data = sub[(sub.budget == budget) | sub.arm.isin(["mean", "stack", "guided_library", "placebo_library"])]
            for metric in ("brier", "accuracy", "fit_seconds", "nodes"):
                means = data.pivot_table(index="bank", columns="arm", values=metric)
                paired = data.pivot_table(index=["bank", "gp_seed"], columns="arm", values=metric)
                for challenger in ("native", "guided", "placebo"):
                    for reference in means:
                        delta = (means[challenger] - means[reference]).dropna()
                        if challenger == reference or len(delta) == 0:
                            continue
                        margin = t.ppf(.975, len(delta)-1)*delta.sem() if len(delta)>1 else np.nan
                        by_start = paired[challenger].groupby(level="bank").std()
                        rows.append(dict(family=name, split=split, budget=budget, metric=metric, arm=challenger, reference=reference,
                                         n_banks=len(delta), mean=means[challenger].mean(), reference_mean=means[reference].mean(),
                                         delta=delta.mean(), lo=delta.mean()-margin, hi=delta.mean()+margin,
                                         lower_banks=int((delta<0).sum()), higher_banks=int((delta>0).sum()),
                                         between_bank_sd=means[challenger].std(), mean_within_bank_start_sd=by_start.mean()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/guided/results"))
    parser.add_argument("--banks", nargs="+", type=int, default=list(range(25100,25112)))
    parser.add_argument("--optimizer-seeds", nargs="+", type=int, default=[311,751])
    parser.add_argument("--families", nargs="+", choices=AVAILABLE_FAMILIES, default=FAMILIES)
    args = parser.parse_args()
    logger.disable("okapi")
    with threadpool_limits(limits=1):
        frozen = freeze_all(args.banks, args.optimizer_seeds, args.families)
        frame = evaluate(frozen)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "runs.csv", index=False)
    stats(frame).to_csv(args.out / "paired.csv", index=False)


if __name__ == "__main__":
    main()
