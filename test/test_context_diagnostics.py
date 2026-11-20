import numpy as np

from okapi.context_diagnostics import assess_context_mixture


def test_deterministic_contradiction_and_in_segment():
    ref = np.zeros((30, 2))
    alt = np.tile([1.0, 0.0], (30, 1))
    compatible = assess_context_mixture(np.tile([0.4, 0.0], (40, 1)), ref, alt, 0.1)
    assert compatible.status == "compatible"
    assert abs(compatible.prevalence - 0.4) < 1e-10
    incompatible = assess_context_mixture(np.tile([0.4, 1.0], (40, 1)), ref, alt, 0.1)
    assert incompatible.status == "incompatible"
    assert incompatible.p_value <= 0.05


def test_unidentified_and_missing_group():
    x = np.ones((10, 1))
    assert assess_context_mixture(x, x, x, 0.2).status == "unidentified"
    assert assess_context_mixture(x, x, x[:0], 0.2).status == "insufficient_source"


def test_scalar_interior_cannot_check_another_moment():
    a = np.tile([0.1], (40, 1))
    b = np.tile([0.9], (40, 1))
    x = np.linspace(0.2, 0.8, 80)[:, None]
    fit = assess_context_mixture(x, a, b, 0.1)
    assert fit.status == "compatible"
    assert fit.p_value == 1.0


def test_permutation_invariant_check():
    from okapi.context_diagnostics import invariant_mean_pvalue

    a = np.zeros((30, 5))
    b = np.zeros((40, 5))
    assert invariant_mean_pvalue(a, b, draws=99) == 1.0
    b[:, 2] = 1.0
    assert invariant_mean_pvalue(a, b, draws=99) == 0.01
