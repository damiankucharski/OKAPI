"""Cross-validated fitness (k-fold evaluation averaging of the objective).

OKAPI operators are pointwise across samples, so a tree's prediction is computed once
on the full split and sliced per fold; each objective becomes the mean of its per-fold
scores (minus ``cv_penalty * std``, in the objective's worsening direction). ``cv=None``
must reproduce the single-split fitness exactly. These tests pin: the default no-op, fold
construction (int -> KFold, splitter object as-is, disjoint cover, determinism), the
mean/std maths, the penalty sign for maximise vs minimise, and an end-to-end train.
"""

import numpy as np
import pytest

from okapi.okapi import Okapi
from okapi.pareto import maximize, minimize


def _identity(x):
    return x


def _acc(prediction, gt):
    p = np.asarray(prediction)
    y = np.asarray(gt).ravel().astype(int)
    yhat = (p[:, 0] > 0.5).astype(int) if p.shape[1] == 1 else p.argmax(axis=1)
    return float((yhat == y).mean())


def _err(prediction, gt):  # a minimise objective
    return 1.0 - _acc(prediction, gt)


def _make_engine(tmp_path, *, cv=None, cv_penalty=0.0, objective=_acc, objectives=(maximize,),
                 seed=0, n=200, n_models=6, pop=6):
    rng = np.random.default_rng(seed)
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir(parents=True)
    for k in range(n_models):
        np.save(fit_dir / f"m{k}.npy", rng.random((n, 1)).astype(np.float64))
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, (rng.random(n) > 0.5).astype(np.int64))
    return Okapi(
        preds_source=fit_dir,
        gt_path=gt_path,
        population_size=pop,
        population_multiplier=3,
        tournament_size=4,
        minimize_node_count=False,
        objective_functions=(objective,),
        objectives=objectives,
        backend="numpy",
        seed=seed,
        postprocessing_function=_identity,
        cv=cv,
        cv_penalty=cv_penalty,
    )


# --------------------------------------------------------------------------- #
# default no-op
# --------------------------------------------------------------------------- #


def test_cv_none_is_full_split_fitness(tmp_path):
    ok = _make_engine(tmp_path, cv=None)
    assert ok._cv_folds is None
    trees = ok.population
    fits = ok._calculate_fitnesses(trees)
    for tree, fit in zip(trees, fits, strict=True):
        pred = tree.predict()
        assert fit[0] == pytest.approx(_acc(pred, ok.gt_tensor), abs=1e-12)
    assert ok.fitness_stds is None  # no std tracked when cv is off


def test_cv_one_is_treated_as_off(tmp_path):
    assert _make_engine(tmp_path, cv=1)._cv_folds is None


# --------------------------------------------------------------------------- #
# fold construction
# --------------------------------------------------------------------------- #


def test_cv_int_builds_disjoint_cover(tmp_path):
    ok = _make_engine(tmp_path, cv=4, n=200)
    assert ok._cv_folds is not None and len(ok._cv_folds) == 4
    allidx = np.concatenate(ok._cv_folds)
    assert sorted(allidx.tolist()) == list(range(200))  # partition of all rows


def test_cv_folds_deterministic_for_seed(tmp_path):
    ok1 = _make_engine(tmp_path / "a", cv=5, seed=7)
    ok2 = _make_engine(tmp_path / "b", cv=5, seed=7)
    for a, b in zip(ok1._cv_folds, ok2._cv_folds, strict=True):
        np.testing.assert_array_equal(a, b)


def test_cv_accepts_sklearn_splitter_object(tmp_path):
    from sklearn.model_selection import StratifiedKFold

    ok = _make_engine(tmp_path, cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=0))
    assert ok._cv_folds is not None and len(ok._cv_folds) == 4
    y = np.asarray(ok.gt_tensor).ravel()
    overall = y.mean()
    for idx in ok._cv_folds:  # stratified -> each fold's class balance near overall
        assert abs(y[idx].mean() - overall) < 0.15


# --------------------------------------------------------------------------- #
# fitness maths
# --------------------------------------------------------------------------- #


def test_cv_fitness_equals_manual_fold_mean(tmp_path):
    ok = _make_engine(tmp_path, cv=5, cv_penalty=0.0, seed=1)
    trees = ok.population
    fits = ok._calculate_fitnesses(trees)
    obj = ok.objective_functions[0]
    for tree, fit in zip(trees, fits, strict=True):
        pred = tree.predict()
        manual = np.mean([obj(pred[idx], ok.gt_tensor[idx]) for idx in ok._cv_folds])
        assert fit[0] == pytest.approx(manual, abs=1e-12)
    assert ok.fitness_stds is not None and ok.fitness_stds.shape == (len(trees), 1)


def test_cv_penalty_worsens_maximize_by_std(tmp_path):
    ok = _make_engine(tmp_path, cv=5, seed=2)
    trees = ok.population
    ok.cv_penalty = 0.0
    f0 = ok._calculate_fitnesses(trees)
    s0 = ok.fitness_stds.copy()
    ok.cv_penalty = 1.0
    f1 = ok._calculate_fitnesses(trees)
    np.testing.assert_allclose(f1[:, 0], f0[:, 0] - s0[:, 0], atol=1e-12)
    assert (f1[:, 0] <= f0[:, 0] + 1e-12).all()  # maximise: penalty lowers fitness


def test_cv_penalty_direction_is_objective_aware(tmp_path):
    # For a minimise objective the penalty ADDS the std (worsening = larger).
    ok = _make_engine(tmp_path, cv=5, seed=3, objective=_err, objectives=(minimize,))
    trees = ok.population
    ok.cv_penalty = 0.0
    f0 = ok._calculate_fitnesses(trees)
    s0 = ok.fitness_stds.copy()
    ok.cv_penalty = 1.0
    f1 = ok._calculate_fitnesses(trees)
    np.testing.assert_allclose(f1[:, 0], f0[:, 0] + s0[:, 0], atol=1e-12)


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #


def test_engine_trains_with_cv(tmp_path):
    ok = _make_engine(tmp_path, cv=4, cv_penalty=0.5, seed=0)
    ok.train(15)
    assert ok.population
    assert ok.fitnesses.shape[0] == len(ok.population)
    # fitness_stds tracks the most recent _calculate_fitnesses call (pre-trim inside
    # run_iteration); recompute on the current population for a population-aligned view.
    fits = ok._calculate_fitnesses(ok.population)
    assert ok.fitness_stds is not None and ok.fitness_stds.shape == fits.shape


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
