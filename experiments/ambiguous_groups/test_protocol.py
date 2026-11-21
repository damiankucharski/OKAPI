import numpy as np

from experiments.ambiguous_groups import run


def test_training_receives_only_own_labels_and_finishes_before_test_draws(monkeypatch):
    original = run.sample
    generated = {}
    fits = []

    def draw(design, seed, split_id, n, sigma):
        if split_id >= 2:
            assert len(fits) == 2 * len(run.ARMS)
        generated[split_id] = original(design, seed, split_id, n, sigma)
        return generated[split_id]

    def fit(arm, fit_x, fit_y, val_x, val_y, *, seed):
        world = "A" if len(fits) < len(run.ARMS) else "H"
        assert set(generated) == {0, 1}
        assert fit_x is generated[0][0] and fit_y is generated[0][1][world]
        assert val_x is generated[1][0] and val_y is generated[1][1][world]
        assert not fit_x.flags.writeable and not val_x.flags.writeable
        fits.append((world, arm))
        return lambda x: np.full(len(x), .5), dict(offered_matchings="01-23-45-67", selected_matching="01-23-45-67")

    monkeypatch.setattr(run, "sample", draw)
    monkeypatch.setattr(run, "fit_method", fit)
    _, _, risks = run.run_bank(13)
    assert len(set(fits)) == 12 and len(risks) == 28
