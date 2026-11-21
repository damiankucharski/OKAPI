import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from okapi.affine import fit_stacking_tree, stacking_tree
from okapi.backend.backend import Backend
from okapi.backend.numpy_backend import NumpyBackend
from okapi.globals import postprocessing_function
from okapi.node import MaxNode, MeanNode, ValueNode
from okapi.tree import Tree


@pytest.fixture(autouse=True)
def binary_numpy(monkeypatch):
    monkeypatch.setattr(Backend, "_current_backend", NumpyBackend)
    monkeypatch.setattr(postprocessing_function, "_postprocessing_function", lambda x: x)


def features(columns, transform):
    if transform == "probability":
        return columns
    p = np.clip(columns, 1e-6, 1 - 1e-6)
    return np.log(p / (1-p))


def public_proba(model, columns, transform):
    pred = np.maximum(model.predict_proba(features(columns, transform)), 1e-12)
    return pred[:, 1] / pred.sum(axis=1)


def channels(positive):
    return np.stack((1-positive, positive), axis=-1).reshape(len(positive), -1)


def leaves(matrix):
    return [ValueNode(None, matrix[:, j:j+1], j) for j in range(matrix.shape[1])]


@pytest.mark.parametrize("transform", ["probability", "logit"])
def test_signed_coefficients_and_intercept_on_complementary_channels(transform):
    rng = np.random.default_rng(300)
    fit = rng.uniform(.05, .95, (300, 3))
    fit[:, 2] = .73  # StandardScaler uses scale 1 for a constant feature.
    y = fit[:, 0] - 1.4*fit[:, 1] + rng.normal(0, .16, len(fit)) > -.04
    pipe = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs")).fit(features(channels(fit), transform), y.astype(int))
    query = rng.uniform(.05, .95, (48, 3))
    query[:2] = [[0, 1, .73], [1, 0, .73]]
    original_coef = pipe[-1].coef_.copy()
    unscaled = pipe[-1].coef_[0]/pipe[0].scale_
    signed = unscaled[1::2]-unscaled[::2]
    assert signed[0] > 0 and signed[1] < 0
    assert abs(pipe[-1].intercept_[0]) > .1
    expected = public_proba(pipe, channels(query), transform)
    folded = stacking_tree(leaves(query), pipe, transform=transform)
    explicit = stacking_tree(leaves(channels(query)), pipe, transform=transform, complementary=False)
    np.testing.assert_allclose(folded.predict().ravel(), expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(explicit.predict().ravel(), expected, atol=2e-12, rtol=0)
    np.testing.assert_array_equal(pipe[-1].coef_, original_coef)


@pytest.mark.parametrize("transform", ["probability", "logit"])
@pytest.mark.parametrize(("center", "scale"), [(False, True), (True, False), (False, False)])
def test_explicit_independent_channels_and_optional_scaling(transform, center, scale):
    rng = np.random.default_rng(311)
    fit, query = rng.uniform(.03, .97, (200, 4)), rng.uniform(.03, .97, (48, 4))
    y = (fit[:, 0] - fit[:, 3] - .1*fit[:, 1] > -.1).astype(int)
    pipe = make_pipeline(StandardScaler(with_mean=center, with_std=scale),
                         LogisticRegression(solver="lbfgs", fit_intercept=False)).fit(features(fit, transform), y)
    expected = public_proba(pipe, query, transform)
    actual = stacking_tree(leaves(query), pipe, transform=transform, complementary=False).predict().ravel()
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    wrong_assumption = stacking_tree(leaves(query[:, 1::2]), pipe, transform=transform).predict().ravel()
    assert np.max(np.abs(wrong_assumption-expected)) > .1


def test_nested_repeated_sources_are_copied_and_rebind_independently():
    rng = np.random.default_rng(330)
    a, b, c = rng.random((3, 90, 1))
    mean = (a+b)/2
    data = np.column_stack((mean, c, mean))
    y = (mean[:, 0]-c[:, 0] > -.07).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(channels(data), y)
    root = ValueNode(None, a, "a")
    root.add_child(MeanNode([ValueNode(None, b, "b")]))
    other = ValueNode(None, c, "c")
    original = root.get_nodes() + other.get_nodes()
    tree = stacking_tree([root, other, root], pipe, transform="probability")
    assert set(original).isdisjoint(tree.root.get_nodes())
    assert len({id(node) for node in tree.root.get_nodes()}) == len(tree.root.get_nodes())
    assert len(root.children) == 1 and root.parent is None and other.parent is None
    np.testing.assert_allclose(tree.predict().ravel(), public_proba(pipe, channels(data), "probability"), atol=2e-12)
    new = {name: rng.random((22, 1)) for name in "abc"}
    new_mean = (new["a"]+new["b"])/2
    new_columns = channels(np.column_stack((new_mean, new["c"], new_mean)))
    np.testing.assert_allclose(tree.do_pred_on_another_tensors(current_tensors=new).ravel(),
                               public_proba(pipe, new_columns, "probability"), atol=2e-12)
    np.testing.assert_allclose(Tree(root).predict(), mean)


def test_reject_incompatible_or_unfitted_pipeline_and_features():
    matrix = np.array([[.2, .8], [.8, .2], [.3, .7], [.7, .3]])
    roots = leaves(matrix[:, 1:])
    y = [0, 1, 0, 1]
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    with pytest.raises(NotFittedError):
        stacking_tree(roots, pipe, transform="logit")
    with pytest.raises(ValueError, match="pipeline"):
        stacking_tree(roots, LogisticRegression(), transform="logit")
    pipe.fit(matrix, y)
    with pytest.raises(ValueError, match="transform"):
        stacking_tree(roots, pipe, transform="sigmoid")
    with pytest.raises(ValueError, match="roots"):
        stacking_tree(roots, pipe, transform="logit", complementary=False)
    with pytest.raises(ValueError, match="roots"):
        stacking_tree([matrix], pipe, transform="logit")
    with pytest.raises(ValueError, match="boolean"):
        stacking_tree(roots, pipe, transform="logit", complementary="yes")
    pipe[-1].classes_ = np.array([1, 0])
    with pytest.raises(ValueError, match="ordered binary"):
        stacking_tree(roots, pipe, transform="logit")
    pipe.fit(matrix, [0, 1, 2, 0])
    with pytest.raises(ValueError, match="ordered binary"):
        stacking_tree(roots, pipe, transform="logit")
    pipe.fit(matrix[:, :1], y)
    with pytest.raises(ValueError, match="2M"):
        stacking_tree(roots, pipe, transform="logit")


@pytest.mark.parametrize(("method", "transform"), [("StackingLR", "probability"), ("StackingLogit", "logit")])
def test_public_fusionbench_model_integration(method, transform):
    stacking = pytest.importorskip("fusionbench.methods.stacking")
    rng = np.random.default_rng(345)
    p, query = rng.uniform(.02, .98, (180, 4)), rng.uniform(.02, .98, (40, 4))
    y = (p[:, 0]-p[:, 1]+p[:, 2] > .62).astype(int)
    model = getattr(stacking, method)().fit(channels(p).reshape(len(p), 4, 2), y, task="binary", n_classes=2)
    expected = model.predict(channels(query).reshape(len(query), 4, 2))[:, 1]
    actual = stacking_tree(leaves(query), model.model, transform=transform).predict().ravel()
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("transform", ["probability", "logit"])
@pytest.mark.parametrize("folded", [True, False])
def test_native_fit_matches_independent_fit_and_transfer_on_nonlinear_roots(transform, folded):
    rng = np.random.default_rng(350)
    source, query = rng.uniform(.02, .98, (250, 5)), rng.uniform(.02, .98, (85, 5))

    def program_columns(raw):
        nonlinear = np.maximum(raw[:, 0], raw[:, 1])
        values = np.column_stack([nonlinear, raw[:, 2]]) if folded else np.column_stack([nonlinear, raw[:, 2:]])
        return channels(values) if folded else values

    first = ValueNode(None, source[:, 0:1], 0)
    first.add_child(MaxNode([ValueNode(None, source[:, 1:2], 1)]))
    indices = [2] if folded else [2, 3, 4]
    roots = [first, *(ValueNode(None, source[:, j:j+1], j) for j in indices)]
    y = (np.maximum(source[:, 0], source[:, 1])-source[:, 2] > .13).astype(int)
    initial_nodes = first.get_nodes()
    initial_values = source.copy()
    native = fit_stacking_tree(roots, y[:, None], transform=transform, complementary=folded, c_value=.42, max_iter=275)
    assert all(node.evaluation is None for node in first.get_nodes() if isinstance(node, ValueNode))
    assert first.get_nodes() == initial_nodes
    np.testing.assert_array_equal(initial_values, source)

    # Independent parameter fit on the analytically evaluated max; only data,
    # settings and the sklearn solver are shared with native training.
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=.42, solver="lbfgs", max_iter=275))
    pipe.fit(features(program_columns(source), transform), y)
    transferred = stacking_tree(roots, pipe, transform=transform, complementary=folded)
    tensor_map = {i: query[:, i:i+1] for i in range(query.shape[1])}
    expected = public_proba(pipe, program_columns(query), transform)
    for tree in (native, transferred):
        actual = tree.do_pred_on_another_tensors(current_tensors=tensor_map).ravel()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("dtype", ["float64", "bfloat16"])
def test_native_fit_evaluates_cpu_torch_roots_and_labels(monkeypatch, dtype):
    torch = pytest.importorskip("torch")
    from okapi.backend.pytorch import PyTorchBackend

    monkeypatch.setattr(Backend, "_current_backend", PyTorchBackend)
    rng = np.random.default_rng(355)
    x = rng.uniform(.02, .98, (120, 2))
    y = (x[:, 0] - x[:, 1] > .1).astype(int)
    values = torch.tensor(x, dtype=getattr(torch, dtype), requires_grad=True)
    roots = [ValueNode(None, values[:, j:j+1], j) for j in range(2)]
    labels = torch.tensor(y, dtype=getattr(torch, dtype))
    actual = fit_stacking_tree(roots, labels, transform="logit").predict().detach().cpu().numpy().ravel()
    observed = (values.float() if dtype == "bfloat16" else values).detach().numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=2000)).fit(features(channels(observed), "logit"), y)
    tolerance = 2e-6 if dtype == "bfloat16" else 2e-12
    np.testing.assert_allclose(actual, public_proba(pipe, channels(observed), "logit"), atol=tolerance, rtol=0)


def test_native_fit_rejects_invalid_labels_roots_and_outputs():
    p = np.array([.1, .4, .6, .9])
    roots = [ValueNode(None, p, "p")]
    y = np.array([0, 0, 1, 1])
    for bad_labels in ([1, 1, 1, 1], [0, 0, 2, 2], [[0, 1], [1, 0]], [0, 1, np.nan, 1]):
        with pytest.raises(ValueError, match="binary classes"):
            fit_stacking_tree(roots, bad_labels, transform="probability")
    for bad in (np.column_stack((p, p)), p[:2], np.array(1.)):
        with pytest.raises(ValueError, match="one probability per label"):
            fit_stacking_tree([ValueNode(None, bad, "p")], y, transform="probability")
    for bad in ([0, np.nan, 1, .2], [0, 2, 1, .2], [0, np.inf, 1, .2], [-.1, 1, 0, 1]):
        with pytest.raises(ValueError, match="finite probabilities"):
            fit_stacking_tree([ValueNode(None, np.array(bad), "p")], y, transform="logit")
    with pytest.raises(ValueError, match="2M"):
        fit_stacking_tree(roots, y, complementary=False, transform="probability")
    for bad_roots in ([], [p]):
        with pytest.raises(ValueError, match="ValueNode roots"):
            fit_stacking_tree(bad_roots, y, transform="probability")
