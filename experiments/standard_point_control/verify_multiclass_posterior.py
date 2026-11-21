"""Check the multiclass posterior against full Gaussian densities on observed log ratios."""

import json
from statistics import NormalDist

import numpy as np
from fusionbench.scenarios.calibration import MulticlassCommittee
from fusionbench.scenarios.committee import Saturated
from scipy.special import softmax
from scipy.stats import multivariate_normal

from okapi.functions import scale_vector_to_sum_1
from okapi.globals import set_backend, use_postprocessing_function
from okapi.node import LogitMeanNode, ValueNode
from okapi.tree import Tree


def density_posterior(probabilities, strength, rho):
    """Compute each class density in the observable C−1 log ratios per provider."""
    if rho == 1:
        np.testing.assert_array_equal(probabilities, np.broadcast_to(probabilities[:, :1], probabilities.shape))
        probabilities = probabilities[:, :1]
    n, k, c = probabilities.shape
    z = (np.log(probabilities[:, :, :-1]) - np.log(probabilities[:, :, -1:])).reshape(n, -1)
    provider_covariance = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    contrast_covariance = np.eye(c - 1) + np.ones((c - 1, c - 1))
    covariance = np.kron(provider_covariance, contrast_covariance)
    likelihoods = []
    for event_class in range(c):
        mean = strength * np.eye(c)[event_class]
        observable_mean = np.tile(mean[:-1] - mean[-1], k)
        likelihoods.append(multivariate_normal.logpdf(z, mean=observable_mean, cov=covariance))
    return softmax(np.column_stack(likelihoods), axis=1)


def verify(scenario, seed, rho):
    p, _, _ = scenario._generate(192, np.random.default_rng(seed))
    c, k = scenario.n_classes, scenario.k
    strength_input = np.clip(min(scenario.accuracy, .999), .5 + 1e-6, 1 - 1e-9)
    strength = NormalDist().inv_cdf(strength_input) * (1 + .35 * np.log(c))
    weight = strength / (1 + (k - 1) * rho)
    exact = density_posterior(p, strength, rho)
    formula = softmax(weight * np.sum(np.log(p), axis=1), axis=1)
    error = float(np.max(np.abs(exact - formula)))
    np.testing.assert_allclose(formula, exact, rtol=0, atol=1e-12)
    native_temperature = k * weight
    assert np.min(p) > LogitMeanNode._EPS and np.max(p) < 1 - LogitMeanNode._EPS
    assert np.max(np.abs(np.log(p).mean(axis=1) * native_temperature)) < LogitMeanNode._CLAMP
    values = [ValueNode(None, p[:, i], i) for i in range(k)]
    values[0].add_child(LogitMeanNode(values[1:], temperature=native_temperature))
    set_backend('numpy')
    with use_postprocessing_function(scale_vector_to_sum_1):
        native = Tree(values[0]).predict()
    np.testing.assert_allclose(native, exact, rtol=0, atol=1e-12)
    previous_weight = 1 / k if isinstance(scenario, Saturated) else 1.0
    old = softmax(previous_weight * np.log(np.clip(p, 1e-12, 1)).sum(axis=1), axis=1)
    return dict(classes=c, k=k, effective_rho=rho, weight=weight, density_error=error,
                native_temperature=native_temperature, native_density_error=float(abs(native - exact).max()),
                old_max_error=float(abs(old - exact).max()), min_probability=float(p.min()))


if __name__ == '__main__':
    rows = [verify(MulticlassCommittee(), 88117, 0)]
    for k, coefficient in ((6, .95), (1, 0), (3, .37), (3, 1)):
        rows.append(verify(Saturated(k=k, corr=coefficient), 88373, coefficient))
    print(json.dumps(rows, indent=2))
