"""Trace printable versus structural population deduplication under paired fit budgets."""

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from fusionbench.methods.averaging import TopN
from fusionbench.methods.okapi import _brier_objective, _identity
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from experiments.guided.run import BUDGETS, FAMILIES, SIZES, FinishedBudget, ObservedGP, data, providers
from okapi.node import ValueNode
from okapi.pareto import maximize
from okapi.utils import first_uniques_mask

core = import_module("okapi.okapi")
production_mask = core._unique_tree_mask


def legacy_mask(trees):
    return first_uniques_mask(np.asarray([repr(tree) for tree in trees]))


def layout(tree):
    return tuple(
        (type(node), len(node.children), (type(node.id), node.id) if isinstance(node, ValueNode) else None) for node in tree.root.get_nodes()
    )


class CohortAudit:
    def __init__(self, policy, probe):
        self.policy = policy
        self.probe = probe
        self.batches = []
        self.collisions = []
        self.shadow_s = 0.0

    def __call__(self, trees):
        chosen = legacy_mask(trees) if self.policy == "legacy" else production_mask(trees)
        start = perf_counter()
        if self.policy == "legacy":
            old, exact = chosen, production_mask(trees)
        else:
            old, exact = legacy_mask(trees), chosen
        old, exact = np.asarray(old, dtype=bool), np.asarray(exact, dtype=bool)
        novel = np.flatnonzero(~old & exact)
        generation = len(self.batches) + 1
        if len(novel):
            first = {repr(trees[i]): i for i in np.flatnonzero(old)}
            cached = {}

            def predictions(i):
                if i not in cached:
                    cached[i] = trees[i].do_pred_on_another_tensors(current_tensors=self.probe)
                return cached[i]

            for i in novel:
                j = first[repr(trees[i])]
                diff = np.abs(predictions(i) - predictions(j))
                self.collisions.append(
                    dict(
                        generation=generation,
                        index=i,
                        first_index=j,
                        same_layout=layout(trees[i]) == layout(trees[j]),
                        max_abs_diff=float(diff.max()),
                        mean_abs_diff=float(diff.mean()),
                        changed_rows=int((diff != 0).sum()),
                        tree=repr(trees[i]),
                        nodes=trees[i].nodes_count,
                    )
                )
        self.batches.append(
            dict(
                generation=generation,
                proposals=len(trees),
                legacy_keep=int(old.sum()),
                exact_keep=int(exact.sum()),
                suppressed=len(novel),
                reverse=int((old & ~exact).sum()),
            )
        )
        self.shadow_s += perf_counter() - start
        return chosen


class AuditGP(ObservedGP):
    """Exclude comparative diagnostics from the search timer at each checkpoint."""

    def snapshot(self):
        snap = super().snapshot()
        snap.elapsed -= self.monitor.shadow_s
        snap.timing.update(shadow_audit_s=self.monitor.shadow_s, join_batches=len(self.monitor.batches))
        return snap


def fit_search(paths, gt, seed, policy, probe):
    tracer = CohortAudit(policy, probe)
    start = perf_counter()
    gp = AuditGP(
        paths,
        gt,
        len(paths),
        2,
        2,
        backend="numpy",
        seed=seed,
        objective_functions=(_brier_objective,),
        objectives=(maximize,),
        postprocessing_function=_identity,
        reuse_survivor_fitness=False,
    )
    gp.monitor = tracer
    gp.start_tracking(start)
    gp.fitnesses = gp._calculate_fitnesses(gp.population).round(3)
    gp.stage = "evolution"
    with patch.object(core, "_unique_tree_mask", tracer):
        try:
            while True:
                gp.generation += 1
                gp.run_iteration()
        except FinishedBudget:
            pass
    assert gp.calls == 640 and tuple(gp.checkpoints) == BUDGETS
    return gp.checkpoints, tracer


@dataclass
class Frozen:
    policy: str
    predict: object
    search_seed: int = -1
    budget: int = 0
    search_s: float = 0.0
    shadow_s: float = 0.0
    nodes: int = 0
    generation: int = 0
    best_call: int = 0
    selected_n: int = 0


def run_bank(family, bank):
    x, y = data(family, bank, "fit")
    candidates = [Frozen("mean", lambda p: p[:, :, 1].mean(axis=1), selected_n=x.shape[1])]
    for arm, learner in (("topn", TopN(metric="brier")), ("stack", StackingLogit(c_value=1.0))):
        start = perf_counter()
        learner.fit(x, y, task="binary", n_classes=2)
        elapsed = perf_counter() - start
        candidates.append(
            Frozen(arm, lambda p, m=learner: m.predict(p)[:, 1], search_s=elapsed, selected_n=learner.n_selected_ if arm == "topn" else x.shape[1])
        )
    batches, collisions = [], []
    with TemporaryDirectory() as temp:
        start = perf_counter()
        directory = Path(temp)
        paths = []
        for name, col in providers(x).items():
            path = directory / name
            np.save(path, col)
            paths.append(path)
        gt = directory / "ground_truth.npy"
        np.save(gt, y)
        io_s = perf_counter() - start
        for idx, seed in enumerate((311, 751)):
            order = ("legacy", "structural") if (bank + idx) % 2 == 0 else ("structural", "legacy")
            for policy in order:
                checkpoints, tracer = fit_search(paths, gt, seed, policy, providers(x[:64]))
                common = dict(family=family, bank=bank, search_seed=seed, policy=policy)
                batches.extend(dict(**common, **r) for r in tracer.batches)
                collisions.extend(dict(**common, **r) for r in tracer.collisions)
                for limit, point in checkpoints.items():
                    tree = point.model
                    candidates.append(
                        Frozen(
                            policy,
                            lambda p, tr=tree: tr.do_pred_on_another_tensors(current_tensors=providers(p)).ravel(),
                            seed,
                            limit,
                            io_s + point.elapsed,
                            point.timing["shadow_audit_s"],
                            tree.nodes_count,
                            point.search_generation,
                            point.fit_call,
                        )
                    )
    rows = []
    # All four GP checkpoints per policy and all three references are fit-only.
    for split in SIZES:
        xx, yy = data(family, bank, split)
        for model in candidates:
            start = perf_counter()
            p = model.predict(xx)
            infer_s = perf_counter() - start
            rows.append(
                dict(
                    family=family,
                    bank=bank,
                    split=split,
                    policy=model.policy,
                    search_seed=model.search_seed,
                    budget=model.budget,
                    fit_rows=len(y),
                    test_rows=len(yy),
                    nodes=model.nodes,
                    brier=np.mean((p - yy) ** 2),
                    accuracy=np.mean((p > 0.5) == yy),
                    search_s=model.search_s,
                    shadow_s=model.shadow_s,
                    infer_s=infer_s,
                    generation=model.generation,
                    best_call=model.best_call,
                    selected_n=model.selected_n,
                    validation_choices=0,
                )
            )
    return rows, batches, collisions


def paired(frame):
    comparisons = []
    for (family, split), cell in frame.groupby(["family", "split"]):
        for budget in BUDGETS:
            by_bank = cell[cell.budget.isin((0, budget))].pivot_table(index="bank", columns="policy", values="brier")
            for ref in ("legacy", "mean", "topn", "stack"):
                d = by_bank.structural - by_bank[ref]
                m = t.ppf(0.975, len(d) - 1) * d.sem()
                comparisons.append(
                    dict(
                        family=family,
                        split=split,
                        budget=budget,
                        reference=ref,
                        delta=d.mean(),
                        lower=d.mean() - m,
                        upper=d.mean() + m,
                        wins=int((d < 0).sum()),
                        ties=int((d == 0).sum()),
                    )
                )
    return pd.DataFrame(comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--banks", type=int, nargs="+", default=list(range(29200, 29212)))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    logger.disable("okapi")
    scores, batches, collisions = [], [], []
    with threadpool_limits(1):
        for family in FAMILIES:
            for bank in args.banks:
                r, b, c = run_bank(family, bank)
                scores.extend(r)
                batches.extend(b)
                collisions.extend(c)
    frame = pd.DataFrame(scores)
    frame.to_csv(args.out / "scores.csv", index=False)
    pd.DataFrame(batches).to_csv(args.out / "batches.csv", index=False)
    pd.DataFrame(collisions).to_csv(args.out / "collisions.csv", index=False)
    paired(frame).to_csv(args.out / "paired.csv", index=False)
