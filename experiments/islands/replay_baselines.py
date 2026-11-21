"""Post-hoc fit and test audit of the published single-population island control.

Run this file beside run.py, with OKAPI and FusionBench on PYTHONPATH.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from threadpoolctl import threadpool_limits

from fusionbench.methods.stacking import StackingLR, StackingLogit
from run import SCENARIOS, _scenario_pair, _tensor_map, brier, search, select_candidate, split


def rescore(scenario, seed):
    source, shifted = _scenario_pair(scenario)
    fit = split(source, seed, 0, 384)
    selection = split(source, seed, 1, 256)
    checkpoint = search("panmixia", *fit, seed, period=2)[-1]
    tree, _, _ = select_candidate(checkpoint, selection)
    predictions = {
        "OKAPI": lambda x: tree.do_pred_on_another_tensors(current_tensors=_tensor_map(x)),
        "mean": lambda x: x[:, :, 1].mean(axis=1),
    }
    for name, model in (("stack_prob384", StackingLR()), ("stack_logit384", StackingLogit()),
                        ("stack_logit640", StackingLogit())):
        rows = fit
        if name == "stack_logit640":
            rows = tuple(np.concatenate((a, b)) for a, b in zip(fit, selection))
        model.fit(*rows, task="binary", n_classes=2)
        predictions[name] = lambda x, model=model: model.predict(x)[:, 1]
    samples = {"fit": fit, "selection": selection, "iid": split(source, seed, 2, 4096),
               "shift": split(shifted, seed, 3, 4096)}
    return [dict(scenario=scenario, seed=seed, domain=domain, method=method, brier=brier(predict(x), y))
            for domain, (x, y) in samples.items() for method, predict in predictions.items()]


def verify_archive(replay, results_dir):
    archive = pd.read_csv(results_dir / "confirmation.csv")
    archive = archive[archive.budget.isin(["evaluations", "baseline"])].copy()
    archive["method"] = archive.arm.map({"panmixia": "OKAPI", "stack": "stack_logit640", "mean": "mean"})
    keys = ["scenario", "seed", "domain", "method"]
    expected = archive.dropna(subset=["method"]).set_index(keys).brier.sort_index()
    observed = replay[replay.method.isin(["OKAPI", "stack_logit640", "mean"]) &
                      replay.domain.isin(["iid", "shift"])].set_index(keys).brier.sort_index()
    pd.testing.assert_index_equal(expected.index, observed.index)
    np.testing.assert_allclose(expected, observed, atol=1e-12, rtol=0)


if __name__ == "__main__":
    logger.disable("okapi")
    results_dir = Path(__file__).resolve().parent / "results"
    with threadpool_limits(1):
        frame = pd.DataFrame(row for scenario in SCENARIOS for seed in range(21200, 21212)
                             for row in rescore(scenario, seed))
    verify_archive(frame, results_dir)
    frame.to_csv(results_dir / "fit-test-replay.csv", index=False)
