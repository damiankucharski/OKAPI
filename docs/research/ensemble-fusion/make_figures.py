"""Render two article figures from the frozen FusionBench CSVs, without fitting models."""

import argparse
import subprocess
from io import BytesIO
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

REV = "5f9254e0c6e4559d7c0d19f28dc91641fba09d57"
BASE = "docs/ensemble-study/replacement-experiment"
TEAL, BRICK, SLATE = "#00736E", "#A44425", "#566679"
PALE, LINE, INK = "#EAF0F1", "#CFD8D8", "#17282C"
FAMILY = {"flat": "#3675A4", "augmented": "#D68442", "gp": "#716491"}
CANVAS = 7.12


def read_csv(checkout, path):
    source = subprocess.check_output(["git", "-C", str(checkout), "show", f"{REV}:{BASE}/{path}"])
    return pd.read_csv(BytesIO(source), keep_default_na=False)


def save(fig, path, title):
    fig.savefig(
        path,
        metadata={
            "Title": title,
            "Author": "FusionBench and OKAPI study",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def comparison(checkout, output):
    data = read_csv(checkout, "panel/results/matched-variants.csv")
    assert len(data) == 144 and not data.duplicated(["scenario", "rep"]).any()
    assert data.groupby("scenario").rep.apply(lambda v: set(v) == set(range(6))).all()
    assert set(data.chosen) == set(FAMILY)
    for family in FAMILY:
        selected = data.loc[data.chosen == family]
        np.testing.assert_allclose(selected.native, selected[family], rtol=0, atol=1e-12)
    for method in ("native", "gp", "ordinary_full"):
        data[f"gap_{method}"] = data[method] - data.prob_lr
    means = data.groupby("scenario")[["gap_native", "gap_gp", "gap_ordinary_full"]].mean()
    means = means.sort_values("gap_native")
    assert len(means) == 24
    counts = pd.crosstab(data.scenario, data.chosen).reindex(means.index)
    y = np.arange(len(means))

    fig = plt.figure(figsize=(CANVAS, 6.40))
    ax = fig.add_axes([0.245, 0.098, 0.525, 0.783])
    families = fig.add_axes([0.813, 0.098, 0.164, 0.783], sharey=ax)
    fig.text(0.015, 0.975, "A", weight="bold", size=11, va="top")
    fig.text(0.052, 0.975, "Paired test performance", weight="bold", size=11, va="top")
    fig.text(0.792, 0.975, "B", weight="bold", size=11, va="top")
    fig.text(0.829, 0.975, "Tree chosen", weight="bold", size=10, va="top")

    for i in y[::2]:
        for panel in (ax, families):
            panel.axhspan(i - 0.5, i + 0.5, color="#F1F4F4", lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=0.75, zorder=1)
    ax.hlines(y, means.gap_native, means.gap_gp, color="#AEBDBC", lw=0.85, zorder=2)
    ax.plot(
        means.gap_ordinary_full,
        y,
        ls="none",
        marker="^",
        markersize=4.3,
        markeredgewidth=0.6,
        markerfacecolor="white",
        markeredgecolor=SLATE,
        zorder=3,
    )
    ax.plot(
        means.gap_gp,
        y,
        ls="none",
        marker="D",
        markersize=4.0,
        markeredgewidth=0.6,
        markerfacecolor="white",
        markeredgecolor=BRICK,
        zorder=4,
    )
    ax.plot(
        means.gap_native,
        y,
        ls="none",
        marker="o",
        markersize=4.4,
        markeredgewidth=0.5,
        markerfacecolor=TEAL,
        markeredgecolor="white",
        zorder=5,
    )
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.set_yticks(y, [s.replace("_", " ") for s in means.index], fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlim(-0.275, 0.28)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f}".replace("-", "−") if v else "0"))
    ax.set_xlabel("Test Brier minus probability stacking", labelpad=6)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=TEAL,
            ls="none",
            markersize=4.7,
            label="Selected native tree",
        ),
        Line2D(
            [],
            [],
            marker="D",
            markerfacecolor="white",
            color=BRICK,
            ls="none",
            markersize=4.0,
            label="Same search, GP only",
        ),
        Line2D(
            [],
            [],
            marker="^",
            markerfacecolor="white",
            color=SLATE,
            ls="none",
            markersize=4.5,
            label="Short GP",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.043, 0.937),
        ncol=3,
        handlelength=0.85,
        handletextpad=0.35,
        columnspacing=1.35,
        borderaxespad=0,
    )

    # A column of six cells encodes six choices per law, grouped by family.
    left = np.zeros(len(means))
    for family, color in FAMILY.items():
        families.barh(
            y,
            counts[family],
            left=left,
            height=0.58,
            color=color,
            edgecolor="none",
            zorder=2,
        )
        left += counts[family].to_numpy()
    for i in range(1, 6):
        families.axvline(i, color="white", lw=0.75, zorder=3)
    families.set(xlim=(0, 6), xticks=[0, 3, 6], xlabel="Banks")
    families.tick_params(axis="y", left=False, labelleft=False)
    families.xaxis.labelpad = 6
    for a in (ax, families):
        for side in ("left", "right", "top"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(LINE)
    for k, (family, label) in enumerate((("flat", "Provider head"), ("augmented", "Head + GP"), ("gp", "GP only"))):
        fig.text(0.813, 0.942 - k * 0.018, "■", color=FAMILY[family], size=8.0, va="center")
        fig.text(0.835, 0.942 - k * 0.018, label, size=7.3, va="center")
    save(
        fig,
        output / "native-performance.pdf",
        "Same-bank native tree selection and test performance",
    )
    print(
        "Full-class test Brier",
        data[["native", "prob_lr", "gp", "ordinary_full"]].mean().to_dict(),
    )
    print("Native family counts", data.chosen.value_counts().to_dict())
    print(
        "Lower native scenario means",
        int((means.gap_native < 0).sum()),
        "/",
        len(means),
    )


def event_count(series, denominator):
    values = series.map({True: True, False: False, "True": True, "False": False})
    assert len(values) == denominator and not values.isna().any()
    return int(values.sum())


def mechanisms(checkout, output):
    searches = read_csv(checkout, "controlled/results/search.csv")
    witness = read_csv(checkout, "controlled/results/direct-witness.csv")
    supplied = read_csv(checkout, "barrier/supplied.csv")
    live = read_csv(checkout, "barrier/live.csv")
    assert not searches.duplicated(["bank", "law", "policy"]).any()
    assert not witness.duplicated(["bank", "law", "gate", "slots"]).any()
    assert not supplied.duplicated(["bank", "generation", "policy"]).any()
    assert not live.duplicated(["bank", "method"]).any()
    certification, crossing = [], []
    for policy, gate in (("open_raw", "raw"), ("open_kl", "kl")):
        matched = searches.loc[searches.policy == policy]
        null = matched.loc[matched.law == "null"]
        signal = matched.loc[matched.law == "signal"]
        direct = witness.loc[(witness.law == "signal") & (witness.slots == 12) & (witness.gate == gate)]
        assert set(null.bank) == set(signal.bank) == set(direct.bank) == set(range(48))
        certification.append(
            [
                event_count(v, 48)
                for v in (
                    null.ever_harmful_move,
                    null.exact_excess.gt(1e-12),
                    signal.final_better_than_mean,
                    direct.accept,
                )
            ]
        )
    for policy in ("open_kl", "strict_kl"):
        matched = supplied.loc[supplied.policy == policy]
        offer = matched.loc[matched.generation == 2]
        deployment = matched.loc[matched.generation == 6]
        organic = live.loc[live.method == policy]
        assert set(offer.bank) == set(deployment.bank) == set(organic.bank) == set(range(32))
        assert pd.to_numeric(organic.proposal_count).eq(72).all()
        crossing.append(
            [
                event_count(v, 32)
                for v in (
                    offer.offered_1.eq("xor") | offer.offered_2.eq("xor"),
                    deployment.champion.eq("xor"),
                    organic.found_exact,
                    organic.exact_excess.lt(0.28125 - 1e-12),
                )
            ]
        )

    fig = plt.figure(figsize=(CANVAS, 3.75))
    specs = [
        (
            0.037,
            "A",
            "Certifying a replacement",
            "48 banks per law",
            [
                "Null  ·  Ever adopted a worse rule",
                "Null  ·  Ended worse than the mean",
                "Signal  ·  Ended better than the mean",
                "Signal  ·  Accepted supplied optimum",
            ],
            certification,
            48,
            ["Empirical", "KL bound"],
            [BRICK, TEAL],
        ),
        (
            0.541,
            "B",
            "Crossing a neutral intermediate",
            "32 banks per route",
            [
                "Supplied  ·  Offered exact XOR",
                "Supplied  ·  Deployed exact XOR",
                "Free search  ·  Offered exact XOR",
                "Free search  ·  Ended better than the mean",
            ],
            crossing,
            32,
            ["Open breeding", "Strict breeding"],
            ["#3675A4", "#716491"],
        ),
    ]
    for left, letter, title, sub, labels, counts, denominator, legend, colors in specs:
        fig.text(left - 0.016, 0.976, letter, weight="bold", size=10, va="top")
        fig.text(left + 0.014, 0.976, title, weight="bold", size=9.6, va="top")
        fig.text(left + 0.014, 0.921, sub, color=SLATE, size=8, va="top")
        ax = fig.add_axes([left + 0.009, 0.134, 0.417, 0.632])
        ax.set(xlim=(0, 100), ylim=(3.63, -0.57), xticks=[0, 25, 50, 75, 100], yticks=[])
        ax.grid(axis="x", color="#E0E7E6", linewidth=0.6, zorder=0)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.set_xlabel("Banks with the stated event (%)", fontsize=8.1)
        for index, label in enumerate(labels):
            ax.text(0, index - 0.35, label, va="center", fontsize=8.1, weight="medium")
            for k, offset in enumerate((-0.025, 0.215)):
                count = counts[k][index]
                pct = count * 100 / denominator
                pos = index + offset
                ax.barh(pos, 100, height=0.175, color=PALE, linewidth=0, zorder=1)
                ax.barh(pos, pct, height=0.175, color=colors[k], linewidth=0, zorder=2)
                if count == 0:
                    ax.plot(
                        0,
                        pos,
                        marker="|",
                        markersize=4.8,
                        color=colors[k],
                        clip_on=False,
                        zorder=3,
                    )
                inside = pct >= 76
                ax.text(
                    pct + (-1.8 if inside else 1.8),
                    pos,
                    f"{count}/{denominator}",
                    fontsize=7.1,
                    va="center",
                    ha="right" if inside else "left",
                    color="white" if inside else INK,
                    weight="medium",
                    zorder=4,
                )
        fig.legend(
            handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, legend, strict=True)],
            loc="upper left",
            bbox_to_anchor=(left + 0.006, 0.861),
            ncol=2,
            borderaxespad=0,
            handlelength=1,
            handleheight=0.65,
            handletextpad=0.4,
            columnspacing=1.25,
        )
    save(
        fig,
        output / "replacement-mechanisms.pdf",
        "Replacement safety, power and the supply of XOR trees",
    )
    print("Certification counts (empirical, KL)", certification)
    print("XOR counts (open, strict)", crossing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusionbench",
        type=Path,
        required=True,
        help=f"FusionBench checkout containing {REV[:7]}",
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory for the two vector PDFs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "xtick.major.size": 2.5,
            "xtick.color": SLATE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "legend.fontsize": 7.45,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    comparison(args.fusionbench, args.out)
    mechanisms(args.fusionbench, args.out)


if __name__ == "__main__":
    main()
