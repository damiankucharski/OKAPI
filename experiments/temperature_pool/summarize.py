"""Summarize paired outcomes by independent data bank and check scored cost and scope."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from experiments.temperature_pool.panel import FAMILIES
from experiments.temperature_pool.run import FIT_SIZES, GP_BUDGET, GP_SEEDS

METRICS = ("brier", "conditional_excess", "nll", "accuracy")
ARMS = ("mean", "top_n", "stack_logit", "stack_clr", "temperature_one", "temperature_fit", "temperature_random", "gp_stock")
KEY = ["family", "bank", "nfit", "method", "gp_seed"]


def interval(a):
    a = np.asarray(a, dtype=float)
    if len(a) != 16 or not np.isfinite(a).all():
        raise ValueError(f"Expected 16 finite independent-bank means, received {a!r}")
    mean = a.mean()
    width = t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return {"banks": len(a), "mean": mean, "lo95": mean - width, "hi95": mean + width}


def verify(s, b):
    assert set(s.family) == set(FAMILIES) and set(s.nfit) == set(FIT_SIZES)
    assert set(s.method) == set(ARMS) and set(s.bank) == set(range(61200, 61216))
    assert not s.duplicated(KEY + ["split"]).any() and not b.duplicated(KEY).any()
    assert len(b) == 3 * 16 * 2 * 9 and len(s) == 4 * len(b)
    assert s.groupby(KEY)["split"].nunique().eq(4).all()
    assert set(b.loc[b.method == "gp_stock", "gp_seed"]) == set(GP_SEEDS)
    assert b.loc[b.method != "gp_stock", "gp_seed"].eq(-1).all()
    gb = b.query("method == 'gp_stock'")
    assert (gb.fit_calls == GP_BUDGET).all() and (gb.validation_calls == gb.archived_trees).all()
    assert (b.validation_selection_calls == 0).all()
    z = b.pivot(index=["family", "bank", "nfit"], columns="method", values="fit_calls")
    assert (z.temperature_fit == z.temperature_random).all() and (z.temperature_one == 0).all()
    assert (b.loc[b.method == "temperature_fit", "nll_evaluations"] <= 2).all()
    losses = b.pivot(index=["family", "bank", "nfit"], columns="method", values="selected_nll")
    assert (losses.temperature_fit <= losses.temperature_random + 1e-13).all()
    assert (s.groupby(["family", "bank", "nfit", "split"])[["oracle_sha256", "labels_sha256"]].nunique() == 1).all().all()
    assert (s[s.split != "fit"].groupby(["family", "bank", "split"])[["oracle_sha256", "labels_sha256"]].nunique() == 1).all().all()


def summarize(source):
    s, b = (pd.read_csv(source / name, float_precision="round_trip") for name in ("scores.csv.gz", "budgets.csv.gz"))
    verify(s, b)
    m = s.groupby(["family", "bank", "nfit", "split", "method"])[list(METRICS)].mean().reset_index()
    means = []
    for keys, group in m.groupby(["family", "nfit", "split", "method"]):
        for metric in METRICS:
            means.append(dict(zip(["family", "nfit", "split", "method"], keys, strict=True), metric=metric, **interval(group[metric])))
    pd.DataFrame(means).to_csv(source / "means.csv", index=False)
    diffs = []
    for metric in METRICS:
        scores = m.pivot(index=["family", "bank", "nfit", "split"], columns="method", values=metric)
        for reference in (x for x in ARMS if x != "temperature_fit"):
            d = (scores.temperature_fit - scores[reference]).rename("contrast").reset_index()
            # Equal family weights inside each independent bank, only for the two exchangeable laws.
            e = d[d.family != "heterogeneous"].groupby(["bank", "nfit", "split"]).contrast.mean().reset_index().assign(family="exchangeable")
            for keys, group in pd.concat([d, e]).groupby(["family", "nfit", "split"]):
                diffs.append(
                    dict(zip(["family", "nfit", "split"], keys, strict=True), reference=reference, metric=metric, **interval(group.contrast))
                )
    pd.DataFrame(diffs).to_csv(source / "paired.csv", index=False)
    fields = ["fit_calls", "fit_row_scores", "validation_calls", "fit_wall_s", "fit_cpu_s", "temperature", "n_selected"]
    costs = b.groupby(["family", "nfit", "method"])[fields].mean()
    costs.to_csv(source / "costs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    summarize(parser.parse_args().path)
