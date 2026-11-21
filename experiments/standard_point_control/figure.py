"""Plot preregistered point-on versus point-off Brier contrasts at the final fit budget."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MAIN = (
    "committee", "deadweight", "needle", "saturated", "correlated", "hardveto", "hardrescue",
    "weakest_link", "two_pathways", "parity", "compound", "gated_experts", "miscalibration", "multiclass_committee",
)
ADDITIONS = ("calibrated_suppressor", "polarity_fault", "native_switch", "native_close_switch", "compositional_switch")
SCENARIOS = MAIN + ADDITIONS


def contrasts(directory):
    rows = pd.read_csv(directory / "runs.csv.gz")
    panel = rows.loc[
        rows.method.isin(["point_off", "point_on"]) & rows.checkpoint.astype(str).eq("1200")
        & rows.selection.eq("fit") & rows.split.isin(["iid", "shift"])
    ]
    arms = panel.pivot(index=["scenario", "split", "data_seed", "gp_seed"], columns="method", values="brier")
    if arms.isna().any().any():
        raise ValueError("unpaired GP observation")
    runs = (arms.point_on - arms.point_off).groupby(["scenario", "split", "data_seed"])
    assert runs.size().eq(2).all()
    banks = runs.mean().groupby(["scenario", "split"])
    assert banks.count().eq(8).all()
    mean = banks.mean()
    half = t.ppf(.975, 7) * banks.sem()
    points = pd.DataFrame({"difference": mean, "low_95": mean - half, "high_95": mean + half})
    assert set(points.xs("iid", level="split").index) == set(SCENARIOS)
    assert len(points.xs("shift", level="split")) == 9
    published = pd.read_csv(directory / "paired.csv")
    check = published.loc[
        published.checkpoint.astype(str).eq("1200") & published.selection.eq("fit") & published.metric.eq("brier")
    ].set_index(["scenario", "split"]).loc[points.index, points.columns]
    np.testing.assert_allclose(points, check, atol=2e-15, rtol=0)
    return points


def plot(points, output):
    names = [s.replace("_", " ").capitalize() for s in SCENARIOS]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(9, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.225, right=.97, top=.83, bottom=.125, wspace=.20)
    lim = 1000 * np.max(np.abs(points[["low_95", "high_95"]].to_numpy())) * 1.09
    for ax, split, title in zip(axes, ("iid", "shift"), ("Unseen IID", "Predefined shift"), strict=True):
        data = points.xs(split, level="split").reindex(SCENARIOS)
        chosen = data.difference.notna().to_numpy()
        d, lo, hi = (1000 * data.loc[data.difference.notna(), k].to_numpy() for k in points.columns)
        for i in y[::2]:
            ax.axhspan(i - .5, i + .5, color="#f0f3f4", lw=0, zorder=0)
        ax.axvline(0, color="#777c80", lw=.9, zorder=1)
        ax.axhline(len(MAIN) - .5, color="#829098", lw=.8)
        ax.errorbar(d, y[chosen], xerr=[d - lo, hi - d], fmt="o", ms=4.5, color="#205e70",
                    ecolor="#457887", lw=1.25, capsize=2.4, zorder=4)
        ax.set(xlim=(-lim, lim), ylim=(len(y) - .5, -.5))
        ax.set_title(title, fontsize=12, fontweight="semibold", pad=11)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#a3a9ac")
        ax.tick_params(axis="both", labelsize=9, length=0, pad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, symmetric=True))
    axes[0].set_yticks(y, names)
    fig.text(.045, .951, "Does ordinary point mutation improve ensemble Brier?", fontsize=16, weight="bold")
    fig.text(.045, .913, "Fit-selected programs after 1,200 evaluations. Each interval pairs eight banks and averages two starts.",
             fontsize=10, color="#42484d")
    fig.text(.60, .067, "Full Brier change × 1,000  (enabled minus stock)", ha="center", fontsize=10)
    fig.text(.045, .023, "Values below zero favor point mutation. Intervals are pointwise 95%. Blank rows have no registered shift.",
             fontsize=8.7, color="#42484d")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="published FusionBench full-suite-point directory")
    parser.add_argument("output", type=Path, help="output figure file")
    args = parser.parse_args()
    plot(contrasts(args.source), args.output)
