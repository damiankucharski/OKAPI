import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.functions import scale_vector_to_sum_1
from okapi.globals import use_postprocessing_function
from okapi.node import LogitMeanNode
from okapi.temperature import fit_multiclass_temperature_tree


@pytest.fixture(autouse=True)
def numpy_backend(monkeypatch):
    monkeypatch.setattr(Backend, '_current_backend', NumpyBackend)


def expected(providers, temperature, *, clamp=True):
    x = np.stack(tuple(providers.values()), axis=0).astype(np.float64)
    log_scores = np.log(np.clip(x, 1e-6, 1 - 1e-6)).mean(axis=0) * temperature
    if clamp:
        log_scores = np.clip(log_scores, -30, 30)
    return softmax(log_scores, axis=1)


def test_closed_form_three_class_solution_prediction_on_new_ids_and_cost():
    x = np.tile([.5, .25, .25], (6, 1))
    providers = {'teacher': x.astype(np.float32), 4: x.copy()}
    y = np.array([0, 0, 0, 0, 1, 2])
    fit = fit_multiclass_temperature_tree(providers, y[:, None])
    assert len(fit.tree.op_nodes) == 1 and type(fit.tree.op_nodes[0]) is LogitMeanNode
    assert fit.tree.op_nodes[0].temperature == fit.temperature
    assert [node.id for node in fit.tree.value_nodes] == list(providers)
    assert fit.temperature == pytest.approx(2, abs=1e-12)
    assert fit.lower == .05 and fit.upper == pytest.approx(30 / -np.log(.25))
    assert fit.reference_nll == pytest.approx(-np.log(x[np.arange(len(y)), y]).mean())
    assert fit.fit_nll == pytest.approx(-np.log(np.array([2 / 3, 1 / 6, 1 / 6])[y]).mean())
    assert fit.reference_nll > fit.fit_nll and 2 < fit.gradient_evaluations < 25 and fit.nll_evaluations == 2
    frozen = fit.predict()
    assert frozen.dtype == np.float64
    providers['teacher'][:] = [1, 0, 0]
    providers[4][:] = [0, 1, 0]
    np.testing.assert_array_equal(fit.predict(), frozen)
    # New data have a different length, reordered mapping and two numerical dtypes.
    heldout = {4: np.array([[.2, .5, .3], [.75, .25, 0]], dtype=np.float32), 'teacher': np.array([[.1, .2, .7], [0, 1, 0.]])}
    with use_postprocessing_function(lambda a: 0 * a):
        prediction = fit.predict(heldout)
    np.testing.assert_allclose(prediction, expected(heldout, fit.temperature), rtol=2e-15)
    with use_postprocessing_function(scale_vector_to_sum_1):
        native = fit.tree.do_pred_on_another_tensors(current_tensors={key: arr.astype(np.float64) for key, arr in heldout.items()})
    np.testing.assert_array_equal(native, prediction)


def test_objective_compares_to_independent_scalar_optimizer():
    rng = np.random.default_rng(713)
    providers = {j: rng.dirichlet(np.arange(1, 6), size=241) for j in range(4)}
    q = expected(providers, 1.7)
    y = np.array([rng.choice(5, p=p) for p in q])
    fit = fit_multiclass_temperature_tree(providers, y)
    logits = np.log(np.clip(np.stack(list(providers.values())), 1e-6, 1 - 1e-6)).mean(0)

    def objective(t):
        return (logsumexp(t * logits, axis=1) - t * logits[np.arange(len(y)), y]).mean()

    independent = minimize_scalar(objective, method='bounded', bounds=(fit.lower, fit.upper), options={'xatol': 1e-12})
    assert fit.lower < fit.temperature < fit.upper
    assert fit.temperature == pytest.approx(independent.x, abs=2e-7)
    assert fit.fit_nll == pytest.approx(independent.fun, abs=3e-15)
    assert fit.fit_nll == pytest.approx(-np.log(fit.predict()[np.arange(len(y)), y]).mean())


def test_boundary_optima_and_native_clamp_on_new_probabilities():
    p = np.array([[.9, .1], [.1, .9]])
    providers = {'one': p, 'two': p.copy()}
    upper = fit_multiclass_temperature_tree(providers, np.array([0, 1]))
    lower = fit_multiclass_temperature_tree(providers, np.array([1, 0]))
    assert upper.temperature == upper.upper and upper.gradient_evaluations == 2
    assert lower.temperature == lower.lower and lower.gradient_evaluations == 1
    assert lower.nll_evaluations == upper.nll_evaluations == 2
    assert upper.upper * np.log(.1) >= -30
    heldout = {key: np.eye(2) for key in providers}
    prediction = upper.predict(heldout)
    np.testing.assert_allclose(prediction, expected(heldout, upper.temperature), rtol=3e-15)
    assert prediction[0, 1] > 1e-14 and expected(heldout, upper.temperature, clamp=False)[0, 1] < 1e-60


def test_unidentified_temperature_uses_one_without_redundant_passes():
    # Opposing nonuniform providers yield constant pooled log scores in each row.
    p = np.array([[.2, .8], [.95, .05]])
    fit = fit_multiclass_temperature_tree({0: p, 1: p[:, ::-1]}, np.array([1, 1]))
    assert fit.temperature == 1 and fit.gradient_evaluations == 0 and fit.nll_evaluations == 1
    assert fit.fit_nll == fit.reference_nll == np.log(2)
    np.testing.assert_array_equal(fit.predict(), np.full_like(p, .5))


@pytest.mark.parametrize('providers, labels', [
    ({0: [[.4, .6]]}, [0]),
    ({0: [[1.]], 1: [[1.]]}, [0]),
    ({0: [[.4, .6]], 1: [[.4, .6], [.5, .5]]}, [0]),
    ({0: [[.4, .6]], 1: [[.4, .7]]}, [0]),
    ({0: [[.4, .6]], 1: [[np.nan, .7]]}, [0]),
    ({0: [[.4, .6]], 1: [[1.1, -.1]]}, [0]),
    ({0: [[.4, .6]], 1: [[.4, .6]]}, [2]),
    ({0: [[.4, .6]], 1: [[.4, .6]]}, [-1]),
    ({0: [[.4, .6]], 1: [[.4, .6]]}, [0.]),
    ({0: [[.4, .6]], 1: [[.4, .6]]}, [False]),
    ({0: [[.4, .6]], 1: [[.4, .6]]}, [0, 1]),
])
def test_invalid_inputs(providers, labels):
    with pytest.raises(ValueError):
        fit_multiclass_temperature_tree(providers, labels)


def test_new_predictions_require_fitted_provider_ids_and_classes():
    p = np.array([[.4, .6]])
    fit = fit_multiclass_temperature_tree({0: p, 1: p}, np.array([0]))
    with pytest.raises(ValueError):
        fit.predict({0: p, 2: p})
    with pytest.raises(ValueError):
        fit.predict({0: [[.2, .2, .6]], 1: [[.2, .3, .5]]})


def test_cpu_torch_uses_float64_native_values_and_scoped_normalization(monkeypatch):
    torch = pytest.importorskip('torch')
    from okapi.backend.pytorch import PyTorchBackend

    monkeypatch.setattr(Backend, '_current_backend', PyTorchBackend)
    p = np.array([[.2, .5, .3], [.6, .3, .1]], dtype=np.float32)
    data = {0: p, 1: p[:, ::-1].copy()}
    fit = fit_multiclass_temperature_tree(data, np.array([1, 0]))
    assert all(t.value.dtype == torch.float64 for t in fit.tree.value_nodes)
    fresh = {key: values[::-1].copy() for key, values in data.items()}
    with use_postprocessing_function(lambda a: torch.zeros_like(a)):
        probability = fit.predict(fresh)
    np.testing.assert_allclose(probability, expected(fresh, fit.temperature), atol=2e-15, rtol=2e-15)


def test_reported_root_search_cost_counts_distinct_full_gradients(monkeypatch):
    from scipy.optimize import brentq

    import okapi.temperature as module

    calls = []

    def count_samples(derivative, lower, upper, **options):
        def sample(t):
            calls.append(t)
            return derivative(t)

        return brentq(sample, lower, upper, **options)

    monkeypatch.setattr(module, 'brentq', count_samples)
    p = np.tile([.5, .25, .25], (6, 1))
    fit = fit_multiclass_temperature_tree({0: p, 1: p}, np.array([0, 0, 0, 0, 1, 2]))
    assert fit.gradient_evaluations == len({fit.lower, fit.upper, *calls})


@pytest.mark.parametrize(('source', 'target'), [('numpy', 'torch'), ('torch', 'numpy')])
def test_another_okapi_instance_can_switch_the_prediction_backend(tmp_path, source, target):
    torch = pytest.importorskip('torch')
    from okapi import Okapi

    p = np.array([[.5, .25, .25], [.6, .15, .25], [.1, .6, .3]])
    inputs = {0: p, 1: np.roll(p, 1, axis=1)}
    labels = np.array([0, 1, 2])
    Backend.set_backend(source)
    fit = fit_multiclass_temperature_tree(inputs, labels)
    original = fit.predict()
    fit_buffers = [node.value for node in fit.tree.value_nodes]
    # The second real optimizer chooses the other global backend.
    base = tmp_path / 'base'
    ground_truth = tmp_path / 'labels'
    if target == 'numpy':
        base, ground_truth = base.with_suffix('.npy'), ground_truth.with_suffix('.npy')
        np.save(base, p)
        np.save(ground_truth, labels)
    else:
        base, ground_truth = base.with_suffix('.pt'), ground_truth.with_suffix('.pt')
        torch.save(torch.from_numpy(p), base)
        torch.save(torch.from_numpy(labels), ground_truth)
    Okapi([base], ground_truth, 1, 1, 1, backend=target)
    current = Backend.get_backend()
    # Reversed-row views exercise negative strides when the destination is Torch.
    fresh = {key: value[::-1] for key, value in inputs.items()}
    np.testing.assert_allclose(fit.predict(), original, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(fit.predict(fresh), original[::-1], rtol=2e-14, atol=2e-15)
    assert Backend.get_backend() is current
    assert all(node.value is old for node, old in zip(fit.tree.value_nodes, fit_buffers, strict=True))
