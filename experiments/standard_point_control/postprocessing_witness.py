"""Reproduce prediction changes when the current task's postprocessor switches."""

import json

import numpy as np
from fusionbench.eval.metrics import brier_score
from fusionbench.methods.okapi import OKAPIMethod, _identity
from fusionbench.scenarios.registry import SCENARIOS

from okapi.globals import postprocessing_function, set_postprocessing_function


def draw(name, size, seed):
    scenario = SCENARIOS[name]()
    x, y, _ = scenario._generate(size, np.random.default_rng(seed))
    return scenario, x, y


def verify():
    binary, x, y = draw("committee", 96, 487)
    _, test, labels = draw("committee", 192, 488)
    multi, other_x, other_y = draw("multiclass_committee", 96, 487)
    models = [OKAPIMethod(seed=1207, n_restarts=1, generations=3, fitness="brier") for _ in range(2)]
    first, second = models
    original = postprocessing_function._postprocessing_function
    try:
        first.fit(x, y, task=binary.task, n_classes=binary.n_classes)
        before = first.predict(test)
        second.fit(other_x, other_y, task=multi.task, n_classes=multi.n_classes)
        after = first.predict(test)
        set_postprocessing_function(_identity)
        restored = first.predict(test)
        np.testing.assert_array_equal(restored, before)
        return {
            "nodes": first.best_nodes,
            "test_rows": len(test),
            "changed_rows": int(np.any(before != after, axis=1).sum()),
            "max_probability_change": float(np.abs(before - after).max()),
            "full_brier_before": brier_score(labels, before, binary.n_classes),
            "full_brier_after": brier_score(labels, after, binary.n_classes),
        }
    finally:
        set_postprocessing_function(original)
        for model in models:
            model._tmp.cleanup()


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
