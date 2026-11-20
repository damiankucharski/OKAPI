"""Compare local provider reassignment with continued GP on fresh FusionBench draws."""

import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair, _tensor_map
from fusionbench.methods.okapi import OKAPIMethod, _identity
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy.stats import t

from okapi.okapi import Okapi
from okapi.pareto import minimize
from okapi.refinement import refine_provider_assignments


def brier(probabilities, labels):
    return float(np.mean((probabilities[:, 0] - labels.ravel()) ** 2))


class AllowanceSpent(Exception):
    """Stop the trial after the last allowed fit evaluation."""


class BudgetedOkapi(Okapi):
    """Expose already-evaluated proposals to the control, even in a partial generation."""

    calls = 0
    ceiling = None
    best_tree = None
    best_loss = np.inf

    def _calculate_fitnesses(self, trees=None):
        candidates = self.population if trees is None else trees
        n = len(candidates)
        if self.ceiling is not None:
            n = min(n, self.ceiling - self.calls)
        scores = super()._calculate_fitnesses(candidates[:n])
        self.calls += n
        if self.ceiling is not None and n:
            best = int(np.argmin(scores[:, 0]))
            if scores[best, 0] < self.best_loss:
                self.best_loss, self.best_tree = scores[best, 0], candidates[best]
        if n < len(candidates):
            raise AllowanceSpent
        return scores


def draw(scenario, seed, split, size):
    x, y, _ = scenario._generate(size, np.random.default_rng([seed, split]))
    return x, y


def checkpoint(x, y, seed):
    adapter = OKAPIMethod(n_seeds=6)
    try:
        directory = adapter._write_split(x, "fit")
        labels = Path(adapter._tmp.name) / "labels.npy"
        np.save(labels, y)
        gp = BudgetedOkapi(
            [directory / key for key in _tensor_map(x)], labels, population_size=min(8, x.shape[1]),
            population_multiplier=2, tournament_size=3, objective_functions=(brier,), objectives=(minimize,),
            backend="numpy", postprocessing_function=_identity, seed=seed, max_depth=7, max_nodes=21,
        )
        adapter._seed_population(gp, np.random.default_rng(seed))
        for _ in range(8):
            gp.run_iteration()
        winner = gp.pareto_trees[int(np.argmin(gp.pareto_fitnesses[:, 0]))]
        return gp, winner
    finally:
        adapter._tmp.cleanup()


def continue_gp(gp, incumbent, allowance, reuse):
    gp.reuse_survivor_fitness = reuse
    start = gp.calls
    gp.best_loss = brier(incumbent.copy().predict(), gp.gt_tensor)
    gp.best_tree = incumbent
    gp.calls += 1
    gp.ceiling = start + allowance
    # Empty batches are possible with cached survivors. A generation guard only
    # catches a permanently exhausted search, which would invalidate the match.
    for _ in range(1000):
        if gp.calls == gp.ceiling:
            break
        try:
            gp.run_iteration()
        except AllowanceSpent:
            break
    assert gp.calls == gp.ceiling, "continuation did not spend the specified allowance"
    return gp.best_tree


def predict(tree, x):
    return tree.do_pred_on_another_tensors(current_tensors=_tensor_map(x))


def studies(scenario, data_seed, search_seed):
    source, shifted = _scenario_pair(scenario)
    fit_x, fit_y = draw(source, data_seed, 0, 256)
    val_x, val_y = draw(source, data_seed, 1, 256)
    start = perf_counter()
    gp, incumbent = checkpoint(fit_x, fit_y, search_seed)
    search_s = perf_counter() - start
    start = perf_counter()
    local = refine_provider_assignments(incumbent, gp.train_tensors, gp.gt_tensor, brier,
                                        max_evaluations=32, seed=np.random.SeedSequence([data_seed, search_seed, 4]))
    local_s = perf_counter() - start
    candidates = {"reassign": (local.tree, local_s)}
    for name, reuse in (("continue_gp", False), ("continue_reuse", True)):
        # Forking the same checkpoint is experimental bookkeeping, outside the
        # policy timer. A deployed continuation already owns this state.
        fork = deepcopy(gp)
        start = perf_counter()
        nominee = continue_gp(fork, incumbent, local.evaluations, reuse)
        candidates[name] = nominee, perf_counter() - start

    choices = {}
    costs = {}
    for method, (challenger, extra_s) in candidates.items():
        start = perf_counter()
        original = brier(predict(incumbent, val_x), val_y)
        challenge = brier(predict(challenger, val_x), val_y)
        chosen = challenger if challenge < original else incumbent
        choices[method] = chosen
        costs[method] = dict(search_s=search_s, extra_s=extra_s, validation_s=perf_counter() - start,
                             gp_calls=gp.calls, extra_calls=local.evaluations, validation_calls=2,
                             changed=chosen is not incumbent, nodes=chosen.nodes_count)
    choices["frozen"] = incumbent
    costs["frozen"] = dict(search_s=search_s, extra_s=0, validation_s=0, gp_calls=gp.calls,
                           extra_calls=0, validation_calls=0, changed=False, nodes=incumbent.nodes_count)
    start = perf_counter()
    stack = StackingLogit().fit(np.concatenate((fit_x, val_x)), np.concatenate((fit_y, val_y)), task="binary", n_classes=2)
    stack_s = perf_counter() - start
    for method, fit_s in (("mean", 0), ("stack", stack_s)):
        costs[method] = dict(search_s=fit_s, extra_s=0, validation_s=0, gp_calls=0, extra_calls=0,
                             validation_calls=0, changed=False, nodes=None)

    # Nothing above this point can access test-domain inputs or labels.
    rows = []
    for domain, distribution, stream in (("iid", source, 2), ("shift", shifted, 3)):
        x, y = draw(distribution, data_seed, stream, 4096)
        for method, cost in costs.items():
            start = perf_counter()
            if method in choices:
                p = predict(choices[method], x)
            elif method == "stack":
                p = stack.predict(x)[:, 1:2]
            else:
                p = x[:, :, 1].mean(axis=1, keepdims=True)
            predict_s = perf_counter() - start
            rows.append(dict(scenario=scenario, data_seed=data_seed, search_seed=search_seed,
                             domain=domain, method=method, brier=brier(p, y), accuracy=np.mean((p[:, 0] >= .5) == y),
                             predict_s=predict_s, **cost))
    return rows


def paired(frame):
    banks = frame.groupby(["scenario", "domain", "data_seed", "method"]).brier.mean().unstack()
    rows = []
    for (scenario, domain), block in banks.groupby(level=["scenario", "domain"]):
        for control in ("continue_gp", "continue_reuse", "frozen", "mean", "stack"):
            d = block.reassign - block[control]
            margin = t.ppf(.975, len(d) - 1) * d.sem()
            rows.append(dict(scenario=scenario, domain=domain, control=control, banks=len(d), difference=d.mean(),
                             lower=d.mean()-margin, upper=d.mean()+margin, wins=int((d < 0).sum())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-seeds", type=int, nargs="+", default=list(range(20200, 20212)))
    args = parser.parse_args()
    logger.disable("okapi")
    records = []
    for scenario in ("two_pathways", "correlated"):
        for data_seed in args.data_seeds:
            for search_seed in (451, 452):
                records.extend(studies(scenario, data_seed, search_seed))
    frame = pd.DataFrame(records)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    frame.to_csv(args.out, index=False)
    paired(frame).to_csv(args.out.with_name(args.out.stem + "_contrasts.csv"), index=False)
