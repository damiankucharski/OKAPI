"""Replay public fit-only Top-N on both previously published guided GP studies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.methods.averaging import TopN
from fusionbench.methods.stacking import StackingLogit
from scipy.stats import t
from threadpoolctl import threadpool_limits

from experiments.guided.run import SIZES, data

EXPERIMENTS = Path(__file__).resolve().parents[1]
STUDIES = {"guided": (("two_pathways", "correlated"), range(25100, 25112)),
           "guided_transfer": (("weakest_link", "parity"), range(28100, 28112))}
PUBLISHED_GP = {"native", "guided", "placebo"}


def score(pred, y):
    return dict(brier=float(np.mean((pred-y)**2)), accuracy=float(np.mean((pred>.5)==y)))


def fitted_controls():
    """Freeze every fit before the first validation or test draw."""
    fitted = {}
    for study, (families, banks) in STUDIES.items():
        for family in families:
            for bank in banks:
                x, y = data(family, bank, "fit")
                start = perf_counter()
                top = TopN(metric="brier").fit(x, y, task="binary", n_classes=2)
                top_s = perf_counter()-start
                start = perf_counter()
                stack = StackingLogit(c_value=1.).fit(x, y, task="binary", n_classes=2)
                stack_s = perf_counter()-start
                m = x.shape[1]
                if sorted(top.ranking_) != list(range(m)) or len(top.prefix_scores_) != m:
                    raise ValueError("Incomplete public Top-N diagnostics")
                fitted[study, family, bank] = (top, stack, top_s, stack_s)
    return fitted


def replay(models):
    rows, fits = [], []
    for (study, family, bank), (top, stack, top_s, stack_s) in models.items():
        m = len(top.ranking_)
        fits.append(dict(study=study, family=family, bank=bank, providers=m, selected=int(top.n_selected_),
                         ranking=json.dumps(np.asarray(top.ranking_).tolist()),
                         prefix_scores=json.dumps(np.asarray(top.prefix_scores_).tolist()),
                         fit_labels=SIZES["fit"], selection_labels=0, individual_scores=m, prefix_scores_count=m,
                         individual_row_scores=m*SIZES["fit"], prefix_row_scores=m*SIZES["fit"], fit_seconds=top_s))
        for split in SIZES:
            x, y = data(family, bank, split)
            for method, estimator, fit_seconds in (("mean", None, 0.), ("topn", top, top_s), ("stack", stack, stack_s)):
                start = perf_counter()
                pred = x[:, :, 1].mean(axis=1) if estimator is None else estimator.predict(x)[:, 1]
                seconds = perf_counter()-start
                rows.append(dict(study=study, family=family, bank=bank, split=split, method=method, n=len(y),
                                 fit_seconds=fit_seconds, predict_seconds=seconds, **score(pred, y)))
    return pd.DataFrame(rows), pd.DataFrame(fits)


def verify_archives(controls):
    prior, checks = [], []
    for study in STUDIES:
        archive = pd.read_csv(EXPERIMENTS / study / "results/runs.csv", float_precision="round_trip")
        keys = ["family", "bank", "split", "method"]
        archived = archive[archive.arm.isin(["mean", "stack"])].rename(columns={"arm": "method"})
        reproduced = controls[(controls.study==study) & controls.method.isin(["mean", "stack"])]
        check = archived[keys + ["brier", "accuracy"]].merge(reproduced[keys + ["brier", "accuracy"]], on=keys,
                                                            validate="one_to_one", how="outer", indicator=True,
                                                            suffixes=("_published", "_replay"))
        assert len(check) == 2 * 12 * 4 * 2 and check._merge.eq("both").all()
        check = check.drop(columns="_merge")
        check["brier_delta"] = check.brier_replay - check.brier_published
        check["accuracy_delta"] = check.accuracy_replay - check.accuracy_published
        np.testing.assert_allclose(check.brier_published, check.brier_replay, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(check.accuracy_published, check.accuracy_replay)
        checks.append(check.assign(study=study))
        gp = archive[archive.arm.isin(PUBLISHED_GP)].copy()
        assert gp.groupby(["family", "bank", "split", "arm", "budget"]).gp_seed.nunique().eq(2).all()
        gp["method"] = gp.arm + "_" + gp.budget.astype(str)
        pooled = gp.groupby(keys, as_index=False)[["brier", "accuracy"]].mean()
        prior.append(pooled.assign(study=study))
    return pd.concat(prior, ignore_index=True), pd.concat(checks, ignore_index=True)


def paired(frame):
    rows = []
    for (study, family, split), part in frame.groupby(["study", "family", "split"]):
        for metric in ("brier", "accuracy"):
            banks = part.pivot(index="bank", columns="method", values=metric)
            for method in banks:
                diff = banks[method]-banks.topn
                width = t.ppf(.975, len(diff)-1)*diff.sem()
                rows.append(dict(study=study, family=family, split=split, metric=metric, method=method, n=len(diff),
                                 mean=banks[method].mean(), topn=banks.topn.mean(), delta=diff.mean(),
                                 lo=diff.mean()-width, hi=diff.mean()+width,
                                 lower_banks=int((diff<0).sum()), higher_banks=int((diff>0).sum())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXPERIMENTS / "guided_topn/results")
    args = parser.parse_args()
    cpu = None
    if hasattr(os, "sched_getaffinity"):
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
    with threadpool_limits(1):
        controls, fits = replay(fitted_controls())
    prior, checks = verify_archives(controls)
    scores = pd.concat([controls[prior.columns], prior], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    controls.assign(cpu=cpu).to_csv(args.out / "controls.csv", index=False)
    fits.to_csv(args.out / "selection.csv", index=False)
    checks.to_csv(args.out / "archive_checks.csv", index=False)
    paired(scores).to_csv(args.out / "paired.csv", index=False)


if __name__ == "__main__":
    main()
