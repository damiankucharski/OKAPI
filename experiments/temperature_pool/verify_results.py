"""Check fitted NLL, archived temperature predictions and the exchangeable Gaussian oracle."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from fusionbench.scenarios.calibrated import mc_strength
from loguru import logger
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from experiments.temperature_pool.panel import FAMILIES, SIZES, data
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import set_backend, use_postprocessing_function
from okapi.node import LogitMeanNode, ValueNode
from okapi.tree import Tree

BANKS = range(61200, 61216)
FIT_SIZES = (128, 512)
METHODS = ('temperature_one', 'temperature_fit', 'temperature_random')


def read_rows(path, *, scores):
    records = {}
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['method'] not in METHODS:
                continue
            assert int(row['gp_seed']) == -1
            key = (row['family'], int(row['bank']), int(row['nfit']), row['method'])
            if scores:
                key += (row['split'],)
            if key in records:
                raise ValueError(f'Duplicate key in {path.name}: {key}')
            records[key] = row
    return records


def digest(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def log_features(p):
    # Explicit provider-axis accumulation independently of the fitter and the evaluation driver.
    return np.add.reduce([np.log(np.clip(p[:, j, :], 1e-6, 1 - 1e-6)) for j in range(p.shape[1])]) / p.shape[1]


def native_prediction(p, temperature):
    children = [ValueNode(None, p[:, j, :], f'm{j}.npy') for j in range(p.shape[1])]
    root = children[0]
    root.add_child(LogitMeanNode(children[1:], temperature=temperature, shift=0.0))
    with use_postprocessing_function(scale_vector_to_sum_1):
        return Tree(root).predict()


def verify_fit(cell, budgets, features, y):
    family, bank, nfit = cell
    one, fitted, random = (budgets[cell + (name,)] for name in METHODS)
    lower = .05
    upper = min(50.0, 30.0 / -features.min())
    np.testing.assert_allclose([float(fitted['lower']), float(fitted['upper'])], [lower, upper], rtol=2e-15, atol=0)
    assert float(one['temperature']) == 1 and int(one['fit_calls']) == 0

    def loss(t):
        logits = t * features
        return float((logsumexp(logits, axis=1) - logits[np.arange(nfit), y]).mean())

    t = float(fitted['temperature'])
    assert lower <= t <= np.nextafter(upper, np.inf)
    # Independent value-only solver also tests the endpoints the bounded solver only approaches.
    opt = minimize_scalar(loss, bounds=(lower, upper), method='bounded', options={'xatol': 1e-12})
    assert opt.success
    error = abs(min(loss(lower), loss(upper), opt.fun) - loss(t))
    assert error <= 1e-10, (cell, error)
    np.testing.assert_allclose([loss(1.0), loss(t)], [float(fitted['reference_nll']), float(fitted['selected_nll'])], rtol=0, atol=1e-12)
    calls = int(fitted['fit_calls'])
    assert calls == int(float(fitted['gradient_evaluations'])) + int(float(fitted['nll_evaluations'])) == int(random['fit_calls'])
    candidates = np.asarray(json.loads(random['sampled_temperatures']))
    rng = np.random.default_rng([bank, FAMILIES.index(family), nfit, 17])
    # Use the archived machine endpoints here, after checking them against the independently computed bound.
    expected = np.r_[1.0, rng.uniform(float(random['lower']), float(random['upper']), size=calls - 1)]
    np.testing.assert_array_equal(candidates, expected)
    losses = [loss(value) for value in candidates]
    np.testing.assert_allclose(losses, json.loads(random['candidate_nll']), rtol=0, atol=1e-12)
    assert candidates[np.argmin(losses)] == float(random['temperature'])
    return error


def verify_predictions(row, p, y, q, features, temperature):
    prediction = native_prediction(p, temperature)
    assert prediction.shape == q.shape and np.isfinite(prediction).all() and (prediction > 0).all()
    np.testing.assert_allclose(prediction.sum(axis=1), 1, rtol=0, atol=3e-15)
    assert int(row['rows']) == len(y) and int(row['classes']) == q.shape[1] and int(row['providers']) == p.shape[1]
    assert digest(prediction, '<f8') == row['prediction_sha256'], (row['family'], row['bank'], row['method'], row['split'], temperature)
    assert digest(q, '<f8') == row['oracle_sha256'] and digest(y, '<i8') == row['labels_sha256']
    onehot = np.zeros_like(prediction)
    onehot[np.arange(len(y)), y] = 1.0
    calculated = (
        np.einsum('ij,ij->', prediction - onehot, prediction - onehot) / len(y),
        np.einsum('ij,ij->', prediction - q, prediction - q) / len(y),
        -np.log(prediction[np.arange(len(y)), y]).mean(), (prediction.argmax(axis=1) == y).mean(),
    )
    saved = [float(row[name]) for name in ('brier', 'conditional_excess', 'nll', 'accuracy')]
    error = np.max(np.abs(calculated - np.asarray(saved)))
    assert error <= 2e-12, (row['method'], error)
    input_clipped = np.any((p < 1e-6) | (p > 1 - 1e-6), axis=(1, 2))
    output_clamped = np.any(np.abs(temperature * features) > 30., axis=1)
    assert input_clipped.sum() == int(float(row['input_clipped_rows']))
    assert output_clamped.sum() == int(float(row['output_clamped_rows']))
    return float(error)


def check_gaussian(family, split, p, q, features, totals):
    # In each class, the shift of the Gaussian mean is boost * ones across the K providers.
    k, classes = p.shape[1:]
    if family == 'multiclass_committee':
        accuracy, rho = .60, .55 if split == 'shift' else 0.0
    else:
        accuracy, rho = .95, .35 if split == 'shift' else .95
    covariance = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    slopes = np.linalg.solve(covariance, np.full(k, mc_strength(accuracy, classes)))
    # Subtract a reference class per provider to cancel its unknown softmax normalizer.
    log_odds = np.log(p) - np.log(p[:, :, :1])
    gauss = softmax(np.einsum('nmc,m->nc', log_odds, slopes), axis=1)
    error = float(np.max(np.abs(gauss - q)))
    assert error <= 2e-13
    temperature = float(slopes.sum())
    native = native_prediction(p, temperature)
    clipped = np.any((p < 1e-6) | (p > 1 - 1e-6), axis=(1, 2))
    clamped = np.any(np.abs(temperature * features) > 30, axis=1)
    rows = np.max(np.abs(native - q), axis=1)
    clean_error = float(rows[~(clipped | clamped)].max(initial=0))
    assert clean_error <= 2e-13
    totals['splits'] += 1
    totals['rows'] += len(p)
    totals['input_clipped_rows'] += int(clipped.sum())
    totals['output_clamped_rows'] += int(clamped.sum())
    totals['any_clipped_rows'] += int(np.count_nonzero(clipped | clamped))
    for key, value in [('gaussian_max_abs', error), ('native_all_max_abs', float(rows.max())), ('native_unclipped_max_abs', clean_error)]:
        totals[key] = max(totals[key], value)


def tail_witness():
    """Two points in the Gaussian logit support collapse to identical clipped native inputs."""
    k, classes = 8, 5
    raw = np.zeros((2, k, classes))
    raw[:, 0, 0] = [20., 34.]
    raw[:, 1:, 0] = -27. / (k - 1)
    inputs = softmax(raw, axis=2)
    np.testing.assert_array_equal(np.clip(inputs[0], 1e-6, 1 - 1e-6), np.clip(inputs[1], 1e-6, 1 - 1e-6))
    strength = mc_strength(.60, classes)
    native = native_prediction(inputs, k * strength)
    np.testing.assert_array_equal(native[0], native[1])
    # Under the independent committee law class zero's relative logit is boost * (-7 or +7).
    posterior = softmax(np.array([[-7., 0., 0., 0., 0.], [7., 0., 0., 0., 0.]]) * strength, axis=1)
    return dict(native_equal=True, native_class_zero=float(native[0, 0]), gaussian_class_zero=posterior[:, 0].tolist())


def verify(directory):
    budgets = read_rows(directory / 'budgets.csv.gz', scores=False)
    scores = read_rows(directory / 'scores.csv.gz', scores=True)
    set_backend('numpy')
    cells = predictions = 0
    fit_error = metric_error = 0.0
    oracle = dict(splits=0, rows=0, input_clipped_rows=0, output_clamped_rows=0, any_clipped_rows=0,
                  gaussian_max_abs=0.0, native_all_max_abs=0.0, native_unclipped_max_abs=0.0)
    for family in FAMILIES:
        for bank in BANKS:
            splits = data(family, bank)
            features = {split: log_features(p) for split, (p, _, _) in splits.items()}
            for split, (p, _, q) in splits.items():
                assert len(p) == SIZES[split]
                if family in ('multiclass_committee', 'saturated'):
                    check_gaussian(family, split, p, q, features[split], oracle)
            for nfit in FIT_SIZES:
                cell = (family, bank, nfit)
                fit_error = max(fit_error, verify_fit(cell, budgets, features['fit'][:nfit], splits['fit'][1][:nfit]))
                cells += 1
                for method in METHODS:
                    temperature = float(budgets[cell + (method,)]['temperature'])
                    for split, (p, y, q) in splits.items():
                        take = slice(nfit) if split == 'fit' else slice(None)
                        row = scores[cell + (method, split)]
                        error = verify_predictions(row, p[take], y[take], q[take], features[split][take], temperature)
                        metric_error = max(metric_error, error)
                        predictions += 1
    assert cells == 96 and len(budgets) == 3 * cells and predictions == len(scores) == 1152 and oracle['splits'] == 128
    return dict(fit_cells=cells, prediction_matrices=predictions, fit_nll_max_abs=fit_error, metric_max_abs=metric_error,
                exchangeable_oracle=oracle, gaussian_tail_witness=tail_witness())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).resolve().parent / 'results')
    logger.remove()
    print(json.dumps(verify(parser.parse_args().directory), sort_keys=True))
