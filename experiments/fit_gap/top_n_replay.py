"""Add the ranked-prefix comparator to the archived matched learning curves."""

from pathlib import Path

import numpy as np
import pandas as pd

from fusionbench.eval.search_stability import _scenario_pair
from fusionbench.methods.averaging import TopN
from run import brier, draw


def score_bank(family, seed):
    source, shift = _scenario_pair(family)
    fit_x, fit_y, _ = draw(source, seed, 0, 1536)
    fitted = {}
    for size in (96, 384, 1536):
        fitted[size] = TopN(metric="brier").fit(fit_x[:size], fit_y[:size],
                                                task="binary", n_classes=2)
    held_out = {
        "validation": draw(source, seed, 1, 256)[:2],
        "iid": draw(source, seed, 2, 4096)[:2],
        "shift": draw(shift, seed, 3, 4096)[:2],
    }
    rows = []
    for size, model in fitted.items():
        fit = (fit_x[:size], fit_y[:size])
        for name, (x, y) in {"fit": fit, **held_out}.items():
            prediction = model.predict(x)[:, 1]
            rows.append(dict(family=family, seed=seed, n=size, split=name,
                             n_selected=model.n_selected_, providers=x.shape[1],
                             mean_replay=brier(x[:, :, 1].mean(axis=1), y),
                             top_n=brier(prediction, y)))
    return rows


if __name__ == "__main__":
    results = Path(__file__).resolve().parent / "results"
    archive = pd.read_csv(results / "measurements.csv")
    keys = ["family", "seed", "n", "split"]
    controls = (archive[archive.method.isin(["mean", "stack", "gp48"])]
                .pivot(index=keys, columns="method", values="brier"))
    banks = archive[["family", "seed"]].drop_duplicates().itertuples(index=False, name=None)
    replay = pd.DataFrame(row for family, seed in banks for row in score_bank(family, seed))
    replay = replay.set_index(keys)
    pd.testing.assert_index_equal(replay.index.sort_values(), controls.index.sort_values())
    comparison = replay.join(controls, validate="one_to_one")
    np.testing.assert_allclose(comparison.pop("mean_replay"), comparison["mean"], atol=1e-12, rtol=0)
    comparison.sort_index().to_csv(results / "top_n_replay.csv")
