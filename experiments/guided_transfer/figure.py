"""Render IID Brier across the original and transfer families from published measurements."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t


def main():
    directory = Path(__file__).resolve().parent
    original = pd.read_csv(directory.parent / "guided/results/runs.csv")
    transfer = pd.read_csv(directory / "results/runs.csv")
    data = pd.concat([original, transfer], ignore_index=True).query("split == 'iid'")
    families = (("two_pathways", "TwoPathways"), ("correlated", "CorrelatedRedundancy"),
                ("weakest_link", "WeakestLink"), ("parity", "Parity"))
    arms = (("guided", "Correlation-guided", "#00766f", "o"),
            ("placebo", "Random membership", "#c65d13", "s"),
            ("native", "Singleton GP", "#525f81", "^"))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.1))
    for ax, (family, title) in zip(axes.flat, families, strict=True):
        cell = data[data.family == family]
        for name, label, color, marker in arms:
            by_bank = cell[cell.arm == name].groupby(["bank", "budget"]).brier.mean().unstack()
            margin = t.ppf(.975, len(by_bank) - 1)*by_bank.sem()
            ax.errorbar([0, 1], by_bank.mean(), yerr=margin, label=label, color=color,
                        marker=marker, markersize=5, linewidth=1.8, capsize=3)
        for name, label, linestyle in (("mean", "Arithmetic mean", ":"), ("stack", "Logit stacker", "--")):
            ax.axhline(cell[cell.arm == name].brier.mean(), color="#303030", lw=1.3, ls=linestyle, label=label)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks([0,1], ["160","640"])
        ax.set_xlim(-.2,1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=.2)
        ax.set_axisbelow(True)
    for ax in axes[:,0]:
        ax.set_ylabel("IID Brier")
    for ax in axes[1]:
        ax.set_xlabel("Fit assessments")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center",
               ncols=3, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.19, left=.11, right=.97, top=.95, wspace=.32, hspace=.36)
    fig.savefig(directory / "convergence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
