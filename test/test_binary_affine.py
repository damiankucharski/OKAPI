from inspect import signature

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from okapi import Okapi
from okapi.backend.backend import Backend
from okapi.globals import postprocessing_function
from okapi.node import BinaryAffineNode, ValueNode
from okapi.operators import BINARY_AFFINE, CLOSE_THRESHOLD, FAR_THRESHOLD, MAX, MEAN, MIN, WEIGHTED_MEAN
from okapi.random_state import use_rng
from okapi.tree import Tree


@pytest.fixture(params=["numpy", "pytorch"])
def backend(request, monkeypatch):
    if request.param == "pytorch":
        pytest.importorskip("torch")
    monkeypatch.setattr(Backend, "_current_backend", Backend.get_backend())
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)
    Backend.set_backend(request.param)
    return Backend.get_backend()


def features(scores, transform):
    p = np.asarray(scores, dtype=np.float64)
    return logit(np.clip(p, 1e-6, 1 - 1e-6)) if transform == "logit" else p


def normalize(p):
    probabilities = np.maximum(p, 1e-12)
    return (probabilities / probabilities.sum(axis=-1, keepdims=True))[..., 1]


def reference(scores, coefficients, intercept, transform):
    linear = np.tensordot(coefficients, features(scores, transform), axes=(0, 0)) + intercept
    p = expit(linear)
    return normalize(np.stack([1 - p, p], axis=-1))


def tree_of(columns, weights, intercept, transform):
    root, *children = [ValueNode(None, col, i) for i, col in enumerate(columns)]
    root.add_child(BINARY_AFFINE(children, weights, intercept, transform))
    return Tree(root)


@pytest.mark.parametrize("transform", ["probability", "logit"])
@pytest.mark.parametrize("shape", [(6,), (6, 1), (2, 3, 1)])
def test_signed_affine_matches_reference_and_preserves_shape(backend, transform, shape):
    scores = np.array([[0., 1., .2, .25, .4, 1e-10], [1., 0., .7, .9, .5, 1 - 1e-10]]).reshape((2, *shape))
    coefficients = [.76543219, -1.86432198]
    intercept = .43
    tree = tree_of([backend.tensor(col) for col in scores], coefficients, intercept, transform)
    direct = BINARY_AFFINE(None, coefficients, intercept, transform).op(backend.tensor(scores))
    expected = reference(scores, coefficients, intercept, transform)
    for result in (direct, tree.predict()):
        out = backend.to_numpy(result)
        assert out.shape == shape
        np.testing.assert_allclose(out, expected, atol=2e-15, rtol=2e-13)


@pytest.mark.parametrize("transform", ["probability", "logit"])
def test_finite_extremes_match_two_class_probability_floor(backend, transform):
    scores = np.array([[0., 1.], [1., 0.]])
    tree = tree_of([backend.tensor(col) for col in scores], [1e4, -1e4], 1.3, transform)
    observed = backend.to_numpy(tree.predict())
    expected = normalize(np.eye(2))
    np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-16)
    np.testing.assert_allclose(observed[0], expected[0], rtol=2e-14, atol=0)


def test_float32_logit_precedes_float64_affine_accumulation(backend):
    # At 1 the float32 upper clip is 0.9999989867, visibly below the
    # float64 clip. Repeating the feature with nearly opposite coefficients
    # also exposes a lost fractional weight in a float32 accumulation.
    p = np.array([1., .9999999, .9999, .5, 0.], dtype=np.float32)
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    stock_logit = np.log(clipped / (1 - clipped)).astype(np.float64)
    weights, intercept = [2**26 + .25, -(2**26)], -3.4
    probability = expit(weights[0] * stock_logit + weights[1] * stock_logit + intercept)
    expected = normalize(np.stack((1 - probability, probability), axis=-1))
    early_promotion = reference(np.stack((p, p)), weights, intercept, "logit")
    assert abs(early_promotion[0] - expected[0]) > .0005
    tree = tree_of([backend.tensor(p), backend.tensor(p)], weights, intercept, "logit")
    np.testing.assert_allclose(backend.to_numpy(tree.predict()), expected, rtol=3e-7, atol=2e-9)


@pytest.mark.parametrize("dtype", [np.float16, np.int32])
def test_half_precision_and_integer_endpoints_have_finite_logits(backend, dtype):
    p = np.array([0, 1], dtype=dtype)
    tree = tree_of([backend.tensor(p)], [.5], .75, "logit")
    expected = reference(p[None, :], [.5], .75, "logit")
    actual = backend.to_numpy(tree.predict())
    assert np.all(np.isfinite(actual)) and np.all((actual > 0) & (actual < 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-13)


@pytest.mark.parametrize("transform", ["probability", "logit"])
def test_explicit_two_column_inputs_reproduce_sklearn_pipeline(backend, transform):
    rng = np.random.default_rng(71)
    # Two explicit class scores per provider. Noncomplementary scores expose
    # conversions that silently discard the first class of each provider.
    raw = rng.uniform(.01, .99, size=(160, 3, 2))
    x = raw.reshape(len(raw), 6)
    y = (x @ np.array([-2., 1., .75, -1., 1.25, -.5]) > -.2).astype(int)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(features(x, transform), y)
    scaler, estimator = pipeline
    coefs = estimator.coef_[0] / scaler.scale_
    intercept = estimator.intercept_[0] - coefs @ scaler.mean_
    test = rng.uniform(size=(23, 6))
    test[0] = [0, 1, .2, .8, .6, .9]
    expected = normalize(pipeline.predict_proba(features(test, transform)))
    tree = tree_of([backend.tensor(col) for col in test.T], coefs.tolist(), intercept, transform)
    np.testing.assert_allclose(backend.to_numpy(tree.predict()), expected, atol=2e-15, rtol=2e-13)


def test_consumes_preceding_parent_evaluation_and_postprocesses_once(backend, monkeypatch):
    root, a, b = [ValueNode(None, backend.tensor(np.array([v])), name) for v, name in ((.2, "r"), (.8, "a"), (.3, "b"))]
    seen = []

    def postprocess(x):
        seen.append(backend.to_numpy(x).item())
        return .5 * x + .1

    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", postprocess)
    root.add_child(MAX([a]))
    root.add_child(BINARY_AFFINE([b], [2., -1.], .5, "probability"))
    expected = expit(2 * .5 - .3 + .5)
    np.testing.assert_allclose(backend.to_numpy(Tree(root).predict()), [.5 * expected + .1])
    np.testing.assert_allclose(seen, [.8, expected])


@pytest.mark.parametrize("shapes", [((4,), (4, 1)), ((3, 1), (4, 1)), ((4, 2), (4, 2))])
def test_tree_rejects_multiclass_and_broadcasting(backend, shapes):
    columns = [backend.tensor(np.full(shape, .5)) for shape in shapes]
    with pytest.raises(ValueError):
        tree_of(columns, [1., 1.], 0., "logit").predict()
    if shapes[0] == shapes[1]:
        with pytest.raises(ValueError):
            BINARY_AFFINE(None, [1., 1.]).op(backend.tensor(np.full((2, *shapes[0]), .5)))


def test_parameter_keys_copy_and_opt_in(backend):
    cols = [backend.tensor(np.array([v])) for v in (.2, .4)]
    tree = tree_of(cols, [1., -.5], .25, "logit")
    copied = tree.copy()
    key = tree.structural_key()
    assert key == copied.structural_key()
    for ps in (([np.nextafter(1., 2.), -.5], .25, "logit"), ([1., -.5], np.nextafter(.25, 1.), "logit"),
               ([1., -.5], .25, "probability")):
        assert key != tree_of(cols, *ps).structural_key()
    with use_rng(np.random.RandomState(17)):
        assert tree.op_nodes[0].mutate_params(mutation_strength=.5)
    assert tree.structural_key() != key and copied.structural_key() == key
    assert BINARY_AFFINE is BinaryAffineNode
    assert signature(Okapi).parameters["allowed_ops"].default == (MEAN, MIN, MAX, WEIGHTED_MEAN, FAR_THRESHOLD, CLOSE_THRESHOLD)


def test_signed_generation_mutation_and_child_parameter_mapping():
    created, mutated, added = [], [], []
    with use_rng(np.random.RandomState(124)):
        for _ in range(32):
            created.extend(BINARY_AFFINE.create_node([]).coefficients)
            node = BINARY_AFFINE([], [0.])
            node.mutate_params(mutation_strength=1.)
            mutated.extend(node.coefficients)
            a, b, c = [ValueNode(None, np.ones(2), i) for i in range(3)]
            node = BINARY_AFFINE([a, b], [-.25, .75, .625])
            new = ValueNode(None, np.ones(2), 9)
            node.add_child(c)
            extra = node.coefficients[-1]
            added.append(extra)
            assert node.coefficients == [-.25, .75, .625, extra]
            node.remove_child(a)
            node.replace_child(b, new)
            assert node.children == [new, c] and node.coefficients == [-.25, .625, extra]
    for values in (created, mutated, added):
        assert min(values) < 0 < max(values)


@pytest.mark.parametrize("coeffs,bias,transform", [([np.nan], 0., "logit"), ([1.], np.inf, "logit"),
                                                  ([1.], 0., "other"), ([1., .4], 0., "logit")])
def test_invalid_construction(coeffs, bias, transform):
    with pytest.raises(ValueError):
        BINARY_AFFINE([], coeffs, bias, transform)
