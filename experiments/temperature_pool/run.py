"""Evaluate frozen scalar-temperature pools and controls on paired multiclass banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter, process_time
from typing import Callable
from unittest.mock import patch

import numpy as np
import pandas as pd
from fusionbench.eval.suite_point_gp import fit_suite_point, predict_tree
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.clr import StackingCLR
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy import optimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from experiments.temperature_pool.panel import FAMILIES, bindings, data, tree_at_temperature
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import set_backend, use_postprocessing_function
from okapi.node import LogitMeanNode
from okapi.temperature import fit_multiclass_temperature_tree

FIT_SIZES = (128, 512)
GP_SEEDS = (1631, 1637)
GP_BUDGET = 600


@dataclass
class Fitted:
    method: str
    predict: Callable[[np.ndarray], np.ndarray]
    cost: dict
    gp_seed: int = -1


def as_list(array):
    return json.dumps(np.asarray(array).tolist(), separators=(",", ":"))


def sha(array, dtype='<f8'):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def mean_log(p):
    clipped = np.clip(p, LogitMeanNode._EPS, 1 - LogitMeanNode._EPS)
    return np.mean(np.log(clipped), axis=1)


def native_temperature_prediction(tree, p):
    with use_postprocessing_function(scale_vector_to_sum_1):
        return tree.do_pred_on_another_tensors(current_tensors=bindings(p))


def timed(function, *args):
    start_wall, start_cpu = perf_counter(), process_time()
    fitted = function(*args)
    fitted.cost.update(fit_wall_s=perf_counter() - start_wall, fit_cpu_s=process_time() - start_cpu)
    return fitted


def reference_methods(name, p, y):
    options = dict(task='multiclass', n_classes=p.shape[2])
    if name == 'mean':
        model = SimpleAverage().fit(p, y, **options)
        return Fitted(name, model.predict, dict(fit_calls=0, fit_labels=0))
    if name == 'top_n':
        model = TopN('brier').fit(p, y, **options)
        return Fitted(name, model.predict, dict(
            fit_calls=2 * p.shape[1], fit_labels=len(y), n_selected=model.n_selected_, ranking=as_list(model.ranking_),
            prefix_scores=as_list(model.prefix_scores_), individual_scores=as_list(model.individual_scores_),
        ))
    constructor = {'stack_logit': StackingLogit, 'stack_clr': StackingCLR}[name]
    model = constructor(c_value=1.)
    # sklearn uses the joint objective-and-gradient L-BFGS callback. Observe its
    # returned counters without adding an evaluation or modifying the parameters.
    solver_results = []
    minimize = optimize.minimize

    def observe(*args, **kwargs):
        result = minimize(*args, **kwargs)
        solver_results.append(result)
        return result

    with patch.object(optimize, 'minimize', observe):
        model.fit(p, y, **options)
    if not solver_results:
        raise RuntimeError('No L-BFGS result observed for the public fitted stacker')
    return Fitted(name, model.predict, dict(
        fit_calls=sum(r.nfev for r in solver_results), fit_labels=len(y), c_value=1.,
        optimizer='lbfgs', optimizer_iterations=sum(r.nit for r in solver_results),
        gradient_evaluations=sum(r.njev for r in solver_results), optimizer_status=as_list([r.status for r in solver_results]),
        feature_columns=p.shape[1] * p.shape[2], stored_coef_and_intercept=p.shape[2] * (1 + p.shape[1] * p.shape[2]),
    ))


def fixed_temperature(p):
    tree = tree_at_temperature(p, 1.)
    return Fitted('temperature_one', partial(native_temperature_prediction, tree), dict(
        fit_calls=0, fit_labels=0, temperature=1., nodes=tree.nodes_count,
    ))


def fit_temperature(p, y):
    fitted = fit_multiclass_temperature_tree(bindings(p), y)
    return Fitted('temperature_fit', partial(native_temperature_prediction, fitted.tree), dict(
        fit_calls=fitted.gradient_evaluations + fitted.nll_evaluations, fit_labels=len(y),
        gradient_evaluations=fitted.gradient_evaluations, nll_evaluations=fitted.nll_evaluations,
        temperature=fitted.temperature, lower=fitted.lower, upper=fitted.upper, reference_nll=fitted.reference_nll,
        selected_nll=fitted.fit_nll, nodes=fitted.tree.nodes_count,
    ))


def random_temperatures(p, y, family, bank, lower, upper, calls):
    rng = np.random.default_rng([bank, FAMILIES.index(family), len(y), 17])
    values = np.r_[1., rng.uniform(lower, upper, size=calls - 1)]
    features = mean_log(p)
    observed = features[np.arange(len(y)), y]
    # The registered fit bounds keep native output clamps inactive on fit.
    losses = [np.mean(logsumexp(t * features, axis=1) - t * observed) for t in values]
    winner = int(np.argmin(losses))
    chosen = float(values[winner])
    tree = tree_at_temperature(p, chosen)
    return Fitted('temperature_random', partial(native_temperature_prediction, tree), dict(
        fit_calls=len(values), fit_labels=len(y), nll_evaluations=len(values), temperature=chosen,
        lower=lower, upper=upper, reference_nll=losses[0], selected_nll=losses[winner],
        sampled_temperatures=as_list(values), candidate_nll=as_list(losses), nodes=tree.nodes_count,
    ))


def stock_gp(p, y, val_p, val_y, seed):
    result = fit_suite_point(p, y, val_p, val_y, seed=seed, point_mutation=False, budgets=(GP_BUDGET,))[GP_BUDGET]
    assert result.calls == GP_BUDGET
    return Fitted('gp_stock', partial(predict_tree, result.fit_tree), dict(
        fit_calls=result.calls, fit_labels=len(y), validation_calls=result.n_trees, validation_labels=len(val_y),
        validation_selection_calls=0, archived_trees=result.n_trees, generation=result.generation,
        nodes=result.fit_tree.nodes_count, path_sha256=result.path_digest, tree=repr(result.fit_tree.structural_key()),
        selected_fit_brier=2 * result.fit_brier, best_archive_validation_brier=2 * result.validation_brier,
    ), gp_seed=seed)


def fit_cell(p, y, val_p, val_y, family, bank):
    methods = [timed(reference_methods, name, p, y) for name in ('mean', 'top_n', 'stack_logit', 'stack_clr')]
    methods.append(timed(fixed_temperature, p))
    solved = timed(fit_temperature, p, y)
    methods.append(solved)
    methods.append(timed(random_temperatures, p, y, family, bank, solved.cost['lower'], solved.cost['upper'], solved.cost['fit_calls']))
    methods.extend(timed(stock_gp, p, y, val_p, val_y, seed) for seed in GP_SEEDS)
    return methods


def score(fitted, p, y, q):
    wall, cpu = perf_counter(), process_time()
    prediction = np.asarray(fitted.predict(p), dtype=np.float64)
    wall, cpu = perf_counter() - wall, process_time() - cpu
    if prediction.shape != q.shape or not np.isfinite(prediction).all() or np.any(prediction < 0):
        raise ValueError('Invalid frozen prediction matrix')
    np.testing.assert_allclose(prediction.sum(axis=1), 1, rtol=0, atol=1e-10)
    targets = np.eye(p.shape[2])[y]
    loss = np.mean(np.sum((prediction - targets)**2, axis=1))
    result = dict(
        rows=len(y), providers=p.shape[1], classes=p.shape[2], brier=float(loss),
        conditional_excess=float(np.mean(np.sum((prediction - q)**2, axis=1))),
        nll=float(-np.mean(np.log(prediction[np.arange(len(y)), y]))), accuracy=float(np.mean(prediction.argmax(axis=1) == y)),
        prediction_sha256=sha(prediction), oracle_sha256=sha(q), labels_sha256=sha(y, '<i8'), prediction_wall_s=wall, prediction_cpu_s=cpu,
    )
    if 'temperature' in fitted.cost:
        logits = fitted.cost['temperature'] * mean_log(p)
        result.update(
            input_clipped_rows=int(np.any((p < LogitMeanNode._EPS) | (p > 1 - LogitMeanNode._EPS), axis=(1, 2)).sum()),
            output_clamped_rows=int(np.any(np.abs(logits) > LogitMeanNode._CLAMP, axis=1).sum()),
        )
    return result


def run(families, banks):
    set_backend('numpy')
    scores, budgets = [], []
    for family in families:
        for bank in banks:
            splits = data(family, bank)
            fit_p, fit_y, _ = splits['fit']
            val_p, val_y, _ = splits['validation']
            if len(fit_y) != max(FIT_SIZES) or len(val_y) != 256 or any(len(splits[k][1]) != 4096 for k in ('iid', 'shift')):
                raise ValueError('Unexpected development or test size')
            # Freeze both development sizes before scoring either one on a test split.
            models = {n: fit_cell(fit_p[:n], fit_y[:n], val_p, val_y, family, bank) for n in FIT_SIZES}
            for nfit, methods in models.items():
                for fitted in methods:
                    key = dict(family=family, bank=bank, nfit=nfit, method=fitted.method, gp_seed=fitted.gp_seed)
                    cost = dict(validation_calls=0, validation_labels=0, validation_selection_calls=0)
                    cost.update(fitted.cost)
                    budgets.append(dict(
                        **key, **cost, fit_row_scores=cost['fit_calls'] * nfit,
                        validation_row_scores=cost['validation_calls'] * len(val_y),
                    ))
                    for split in ('fit', 'validation', 'iid', 'shift'):
                        p, y, q = splits[split]
                        if split == 'fit':
                            p, y, q = p[:nfit], y[:nfit], q[:nfit]
                        metrics = score(fitted, p, y, q)
                        scores.append(dict(**key, split=split, **metrics))
            print(f'{family} {bank}', flush=True)
    return pd.DataFrame(scores), pd.DataFrame(budgets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bank-start', type=int, default=61200)
    parser.add_argument('--bank-stop', type=int, default=61216, help='Exclusive upper endpoint')
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    args = parser.parse_args()
    if not 61200 <= args.bank_start < args.bank_stop <= 61216 or len(set(args.families)) != len(args.families):
        parser.error('Choose a nonempty registered range and unique families')
    logger.remove()
    with threadpool_limits(limits=1):
        scores, budgets = run(args.families, range(args.bank_start, args.bank_stop))
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, records in [('scores.csv', scores), ('budgets.csv', budgets)]:
        target = args.out / filename
        temporary = target.with_suffix('.tmp')
        records.to_csv(temporary, index=False)
        temporary.replace(target)


if __name__ == '__main__':
    main()
