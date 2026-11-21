"""Fit one method using the observed providers and labels from one world."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from fusionbench.methods.okapi import _identity
from fusionbench.methods.stacking import StackingLogit

from experiments.ambiguous_groups.worlds import perfect_matchings, rank_matchings
from okapi.globals import set_backend, set_postprocessing_function
from okapi.guided import _binary_compositions
from okapi.node import ValueNode
from okapi.okapi import Okapi
from okapi.operators import MEAN
from okapi.pareto import minimize
from okapi.tree import Tree


def matching_code(matching):
    return "-".join(f"{a}{b}" for a, b in matching)


def _providers(x):
    return {f"p{j}.npy": x[:, j, 1:2] for j in range(8)}


def _loss(p, y):
    return float(np.mean((p.ravel() - y.ravel()) ** 2))


def _templates(matching, values):
    groups = []
    for a, b in matching:
        ida, idb = f"p{a}.npy", f"p{b}.npy"
        root = ValueNode(None, values[ida], ida)
        root.add_child(MEAN([ValueNode(None, values[idb], idb)]))
        groups.append(root)
    return [Tree(root) for root in _binary_compositions(groups)]


def _proposal_candidates(arm, fit_x, fit_y, seed, stats):
    start = perf_counter()
    matchings = perfect_matchings()
    if arm == "random_two":
        indices = np.random.default_rng([seed, 83]).choice(len(matchings), 2, replace=False)
        offered = [matchings[i] for i in indices]
    else:
        ranked = rank_matchings(fit_x, matchings)
        offered = ranked[:1 if arm == "top_one" else 2]
    stats["grouping_s"] = perf_counter() - start
    codes = [matching_code(matching) for matching in offered]
    stats["offered_matchings"] = ";".join(codes)

    start = perf_counter()
    values = _providers(fit_x)
    trees = [(tree, code) for matching, code in zip(offered, codes, strict=True)
             for tree in _templates(matching, values)]
    stats["build_s"] = perf_counter() - start
    assert len(trees) == 120 * len(offered)
    return [(_loss(tree.predict(), fit_y), tree, code) for tree, code in trees]


class _BudgetReached(Exception):
    """The last paid fit evaluation has completed."""


class _BudgetGP(Okapi):
    """Run stock variation and survival; retain each actual unrounded fit score."""

    def _calculate_fitnesses(self, trees=None):
        cohort = self.population if trees is None else trees
        portion = cohort[:240 - len(self.archive)]
        scores = super()._calculate_fitnesses(portion)
        self.archive.extend((float(s[0]), tree, "") for tree, s in zip(portion, scores, strict=True))
        if len(self.archive) == 240:
            raise _BudgetReached
        return scores


def _gp_candidates(fit_x, fit_y, seed, stats):
    start = perf_counter()
    with TemporaryDirectory() as tmp:
        files = []
        for name, values in _providers(fit_x).items():
            path = Path(tmp) / name
            np.save(path, values)
            files.append(path)
        label_file = Path(tmp) / "y.npy"
        np.save(label_file, fit_y)
        search = _BudgetGP(files, label_file, 8, 2, 2, backend="numpy", seed=seed,
                           objective_functions=(_loss,), objectives=(minimize,),
                           postprocessing_function=_identity, reuse_survivor_fitness=False)
        stats["build_s"] = perf_counter() - start
        search.archive = []
        try:
            while True:
                search.run_iteration()
        except _BudgetReached:
            return search.archive


def fit_method(arm, fit_x, fit_y, val_x, val_y, *, seed):
    """Return a frozen predict(X) callable and per-world fit/selection accounting."""
    start = perf_counter()
    stats = dict(fit_calls=0, validation_calls=0, grouping_s=0., build_s=0., validation_s=0.,
                 diagnostic_s=0., fit_best_brier=None, selected_matching="", offered_matchings="",
                 nodes=None, tree="", stack_fits=0, solver_iterations=0)
    if arm in ("mean", "stack"):
        if arm == "mean":
            def predict(x):
                return x[:, :, 1].mean(axis=1)
        else:
            model = StackingLogit(c_value=1.).fit(fit_x, fit_y, task="binary", n_classes=2)
            stats.update(stack_fits=1, solver_iterations=int(model.model[-1].n_iter_.sum()))

            def predict(x):
                return model.predict(x)[:, 1]

        stats["search_s"] = perf_counter() - start
        diagnostic_start = perf_counter()
        stats.update(fit_brier=_loss(predict(fit_x), fit_y),
                     validation_brier=_loss(predict(val_x), val_y))
        stats["diagnostic_s"] = perf_counter() - diagnostic_start
        return predict, stats

    set_backend("numpy")
    set_postprocessing_function(_identity)
    if arm == "gp":
        records = _gp_candidates(fit_x, fit_y, seed, stats)
    elif arm in ("top_one", "top_two", "random_two"):
        records = _proposal_candidates(arm, fit_x, fit_y, seed, stats)
    else:
        raise ValueError(f"Unknown arm: {arm!r}")

    # Stable ordering resolves equal fit losses by time of evaluation. Repeated
    # assessments of a surviving GP resident still consume the fit budget.
    ordered = sorted(records, key=lambda record: record[0])
    finalists = []
    seen = set()
    for candidate in ordered:
        if arm == "gp":
            code = repr(candidate[1])
            if code in seen:
                continue
            seen.add(code)
        finalists.append(candidate)
        if len(finalists) == 4:
            break
    assert len(finalists) == 4
    stats.update(search_s=perf_counter() - start, fit_best_brier=ordered[0][0], fit_calls=len(records))

    validation_start = perf_counter()
    val_inputs = _providers(val_x)
    losses = [_loss(tree.do_pred_on_another_tensors(current_tensors=val_inputs), val_y)
              for _, tree, _ in finalists]
    best = int(np.argmin(losses))
    fit_loss, winner, selected = finalists[best]
    stats.update(validation_s=perf_counter() - validation_start, fit_brier=fit_loss,
                 validation_brier=losses[best], validation_calls=len(finalists),
                 nodes=winner.nodes_count, tree=repr(winner), selected_matching=selected)

    def predict(x):
        return winner.do_pred_on_another_tensors(current_tensors=_providers(x)).ravel()

    return predict, stats
