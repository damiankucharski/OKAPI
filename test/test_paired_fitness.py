import numpy as np
import pytest

from okapi import Okapi
from okapi.globals import _passthrough, set_backend, use_postprocessing_function
from okapi.node import ValueNode
from okapi.paired_fitness import PairedGainFitness, arithmetic_mean_tree
from okapi.pattern_logit import fit_pattern_tree


def test_noisy_fitness_winner_yields_to_repeatable_gain_without_holdout():
    y = np.array([0, 1] * 8)
    baseline = np.full(16, .5)
    stable = np.full(16, .44)
    volatile = baseline - .09 + np.tile([-.20, .20], 8)
    # Treat supplied values as per-row losses to exercise a general objective.
    raw = PairedGainFitness(baseline, y, lambda p, y: p, z=0)
    paired = PairedGainFitness(baseline, y, lambda p, y: p)
    assert raw(volatile, y) > raw(stable, y)
    assert paired(volatile, y) == pytest.approx(.09 - .20 / np.sqrt(15))
    assert paired(volatile, y) < paired(stable, y)
    assert paired(baseline, y) == 0
    with pytest.raises(ValueError, match="order"):
        paired(stable, y[::-1])


def test_group_weighting_repeated_rows_and_one_unit():
    rows = np.repeat(np.arange(3), [12, 1, 2])
    b = np.ones(len(rows))
    y = rows % 2
    improvements = np.array([.3, -.2, .1])
    value = PairedGainFitness(b, y, lambda p, _: p, groups=rows)(b - improvements[rows], y)
    assert value == pytest.approx(improvements.mean() - improvements.std(ddof=1) / np.sqrt(3))
    assert PairedGainFitness(b, y, lambda p, _: p, groups=np.zeros_like(rows))(b - .2, y) == 0.
    assert PairedGainFitness(b[:1], y[:1], lambda p, _: p, z=0)(b[:1] - .2, y[:1]) == pytest.approx(.2)


def test_native_head_fits_on_same_rows_with_per_trial_penalty():
    set_backend("numpy")
    rng = np.random.default_rng(17)
    labels = rng.integers(2, size=14)
    tensors = {"a": rng.random((14, 1)), "b": rng.random((14, 1))}
    roots = [ValueNode(None, value, key) for key, value in tensors.items()]
    with use_postprocessing_function(_passthrough):
        baseline = arithmetic_mean_tree(tensors).predict()
        fits = []
        for z in (0, 1):
            calls = []

            def row_loss(p, y, calls=calls):
                calls.append(1)
                return (p[:, 0] - y) ** 2

            score = PairedGainFitness(baseline, labels, row_loss, z=z)
            tree, info = fit_pattern_tree(roots, labels, objective=score)
            assert len(calls) == info["evaluations"] + 1  # one cached baseline
            assert score(tree.predict(), labels) == pytest.approx(info["score"])
            fits.append(tree.predict())
        assert not np.allclose(*fits)


@pytest.mark.parametrize("reuse", [False, True])
def test_standard_solver_scores_every_generation_and_returns_native_winner(tmp_path, reuse):
    rng = np.random.default_rng(55)
    folder = tmp_path / "providers"
    folder.mkdir()
    y = rng.integers(2, size=(32, 1))
    for i in range(4):
        np.save(folder / f"{i}.npy", rng.random((32, 1)))
    gt = tmp_path / "labels.npy"
    np.save(gt, y)
    calls = []

    def row_loss(p, y):
        calls.append(1)
        return (p[:, 0] - y[:, 0]) ** 2

    with use_postprocessing_function(_passthrough):
        model = Okapi(folder, gt, 4, 2, 2, backend="numpy", paired_row_loss=row_loss,
                      paired_z=1., paired_groups=np.repeat(np.arange(16), 2),
                      reuse_survivor_fitness=reuse, selection_fitness_decimals=None)
        assert len(calls) == 1
        assert model.paired_fitness_.n_units == 16
        assert model.objective_functions == (model.paired_fitness_,)
        for step in range(3):
            before = len(calls)
            model.run_iteration()
            if not reuse or step == 0:
                assert len(calls) > before
            raw = model._calculate_fitnesses(model.population)
            np.testing.assert_allclose(model.fitnesses, .5 + np.arctan(raw) / np.pi)
        fresh = {k: np.zeros((12, 1)) for k in model.train_tensors}
        chosen, result = model.select_tree_against_mean(fresh, np.zeros((12, 1)), row_loss)
        assert result.index is None and chosen.root.value is model.models[0]
        np.testing.assert_allclose(chosen.do_pred_on_another_tensors(current_tensors=fresh), 0.)


def test_torch_callback_and_missing_or_invalid_inputs():
    torch = pytest.importorskip("torch")
    y = torch.tensor([1, 0, 1, 0])
    p = torch.full((4,), .5, requires_grad=True)
    loss = lambda prediction, target: (prediction - target) ** 2  # noqa: E731
    score = PairedGainFitness(p, y, loss)
    assert score(y, y) == pytest.approx(.25)
    with pytest.raises(ValueError, match="shape"):
        PairedGainFitness(p, y, lambda prediction, target: loss(prediction, target).mean())
    with pytest.raises(ValueError, match="Group IDs"):
        PairedGainFitness(p, y, loss, groups=[1, 2, np.nan, 3])
    with pytest.raises(ValueError, match="nonnegative"):
        PairedGainFitness(p, y, loss, z=-.5)
    with pytest.raises(ValueError, match="finite"):
        score(torch.full((4,), np.nan), y)
