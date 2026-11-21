"""Replay conversion of the published two-provider suppressor models to native trees."""
import argparse
from itertools import chain
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.eval import calibrated_suppressor as panel
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from okapi.affine import stacking_tree
from okapi.globals import set_backend, set_postprocessing_function
from okapi.node import ValueNode

ARCHIVE = Path(panel.__file__).resolve().parents[2] / "docs/ensemble-study/calibrated-suppressor/results"
SCORE_KEY = ["seed", "rho", "split", "method"]
FIT_KEY = ["seed", "rho", "method"]


def split_stream(seed, rho):
    """Mirror run_bank's five streams, including its shared provider permutation."""
    streams = np.random.SeedSequence(seed).spawn(5)
    order = np.random.default_rng(streams[0]).permutation(2)
    source = panel.CalibratedSuppressor(corr=rho)
    shifted = panel.CalibratedSuppressor(corr=0.0 if rho == .85 else .85)

    def draw():
        for stream, (name, law, n) in zip(streams[1:], (
            ("fit", source, 384), ("val", source, 256), ("iid", source, 8192), ("shift", shifted, 8192),
        ), strict=True):
            p, y, _ = law._generate(n, np.random.default_rng(stream))
            yield name, p[:, order], y

    return order, draw()


def roots(x, folded):
    channels = (1,) if folded else (0, 1)
    return [ValueNode(None, x[:, i, c:c+1], 2*i+c) for i in range(x.shape[1]) for c in channels]


def rebound_tree(tree, x):
    bound = tree.copy()
    for leaf in bound.value_nodes:
        i, c = divmod(leaf.id, 2)
        leaf.value = x[:, i, c:c+1]
    return bound


def replay(seed, rho):
    order, draws = split_stream(seed, rho)
    fit_split = next(draws)
    _, fit_x, fit_y = fit_split
    heads, costs, models = [], [], []
    for transform, factory in (("probability", panel.StackingLR), ("logit", panel.StackingLogit)):
        start = perf_counter()
        model = factory(c_value=1.).fit(fit_x, fit_y, task="binary", n_classes=2)
        fit_seconds = perf_counter()-start
        pair = []
        for form, folded in (("explicit", False), ("folded", True)):
            start = perf_counter()
            head = stacking_tree(roots(fit_x, folded), model.model, transform=transform, complementary=folded)
            transfer_seconds = perf_counter()-start
            pair.append((form, head))
            costs.append(dict(seed=seed, rho=rho, method=model.name, transform=transform,
                              form=form, fit_rows=len(fit_y), fits_per_source=1,
                              source_fit_seconds=fit_seconds, solver_iterations=int(model.model[-1].n_iter_.sum()),
                              transfer_seconds=transfer_seconds, transfer_fit_calls=0, validation_choices=0,
                              nodes=head.nodes_count, slots=len(head.value_nodes)))
        models.append((model, pair))
    # Strong/weak identities are used only to interpret already-frozen weights.
    for (model, pair) in models:
        original = panel.coefficients(model)[np.argsort(order)]
        for form, tree in pair:
            head = tree.root.children[-1]
            w = np.array(head.coefficients)
            effective = (w if form == "folded" else w[1::2] - w[::2])[np.argsort(order)]
            rec = next(row for row in costs if (row["method"], row["form"]) == (model.name, form))
            rec.update(intercept=head.intercept, strong_weight=effective[0], weak_weight=effective[1],
                       source_strong_weight=original[0], source_weak_weight=original[1],
                       first_source=int(order[0]), max_weight_error=np.max(np.abs(effective-original)))
    for split, x, y in chain((fit_split,), draws):
        for model, pair in models:
            start = perf_counter()
            source_prediction = model.predict(x)[:, 1]
            source_infer_seconds = perf_counter()-start
            for form, tree in pair:
                start = perf_counter()
                bound = rebound_tree(tree, x)
                bind_seconds = perf_counter()-start
                start = perf_counter()
                prediction = bound.predict().ravel()
                predict_seconds = perf_counter()-start
                assert prediction.shape == source_prediction.shape == y.shape
                heads.append(dict(seed=seed, rho=rho, method=model.name, form=form, split=split,
                                  rows=len(y), max_abs_probability_error=float(np.max(abs(prediction-source_prediction))),
                                  brier=float(np.mean((prediction-y)**2)),
                                  refit_brier=float(np.mean((source_prediction-y)**2)),
                                  refit_infer_seconds=source_infer_seconds, bind_seconds=bind_seconds,
                                  native_predict_seconds=predict_seconds))
    # A separate pass runs the unmodified public driver, including all five
    # fits, its two privileged references, source ordering and scoring methods.
    control_scores, control_fits = panel.run_bank(seed, rho)
    return heads, costs, control_scores, control_fits


def paired(heads, controls):
    cap = controls.query("method == 'nonnegative_logit'")[["seed", "rho", "split", "brier"]]
    signed = heads.query("method == 'stacking_logit' and rho == .85 and split in ('iid','shift')")
    data = signed.merge(cap, on=["seed", "rho", "split"], suffixes=("_signed", "_nonnegative"), validate="many_to_one")
    rows = []
    for (split, form), cell in data.groupby(["split", "form"]):
        delta = cell.brier_signed - cell.brier_nonnegative
        margin = t.ppf(.975, len(delta)-1)*delta.sem() if len(delta) > 1 else np.nan
        rows.append(dict(split=split, form=form, banks=len(delta), signed=cell.brier_signed.mean(),
                         nonnegative=cell.brier_nonnegative.mean(), difference=delta.mean(),
                         lower=delta.mean()-margin, upper=delta.mean()+margin, signed_wins=int((delta < 0).sum())))
    return pd.DataFrame(rows)


def archive_frame(file, seeds):
    frame = pd.read_csv(ARCHIVE / file, float_precision="round_trip")
    return frame[frame.seed.isin(seeds)]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--seeds", nargs="+", type=int, default=list(range(41000, 41024)))
    args = cli.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(range(41000, 41024)):
        cli.error("Use distinct seeds from the original 41000–41023 panel")
    logger.disable("okapi")
    set_backend("numpy")
    set_postprocessing_function(lambda p: p)
    raw = [[], [], [], []]
    with threadpool_limits(1):
        for seed in args.seeds:
            for rho in (0., .5, .85):
                for target, rows in zip(raw, replay(seed, rho), strict=True):
                    target.extend(rows)
    heads, costs, controls, fits = [pd.DataFrame(rows) for rows in raw]
    archived = archive_frame("scores.csv", args.seeds).rename(columns={"brier": "archived_brier"})
    controls = controls.merge(archived, on=SCORE_KEY, validate="one_to_one", how="outer", indicator=True)
    assert controls._merge.eq("both").all()
    controls = controls.drop(columns="_merge")
    controls["archived_difference"] = controls.brier - controls.archived_brier
    heads = heads.merge(archived, on=SCORE_KEY, validate="many_to_one")
    heads["archived_difference"] = heads.brier - heads.archived_brier
    recorded = archive_frame("fits.csv", args.seeds).drop(columns="fit_seconds")
    fits = fits.merge(recorded, on=FIT_KEY, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_archive"))
    assert fits._merge.eq("both").all()
    fits = fits.drop(columns="_merge")
    assert (fits.first_source == fits.first_source_archive).all()
    assert fits.n_selected.fillna(-1).equals(fits.n_selected_archive.fillna(-1))
    for name in ("strong_weight", "weak_weight"):
        np.testing.assert_allclose(fits[name], fits[f"{name}_archive"], atol=1e-9, rtol=0, equal_nan=True)
    assert costs.max_weight_error.max() <= 1e-12
    assert len(heads) == len(args.seeds)*3*2*2*4
    assert heads.max_abs_probability_error.max() <= 1e-9
    assert controls.archived_difference.abs().max() <= 1e-9
    assert heads.archived_difference.abs().max() <= 1e-9
    args.out.mkdir(exist_ok=True, parents=True)
    for name, frame in (("scores", heads), ("heads", costs), ("source_scores", controls), ("source_fits", fits),
                        ("paired", paired(heads, controls))):
        frame.to_csv(args.out / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
