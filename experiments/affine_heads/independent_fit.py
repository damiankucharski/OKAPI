"""Compare separately trained native and public logistic heads on the same fixed roots."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.methods.okapi import _identity
from fusionbench.methods.stacking import StackingLR, StackingLogit
from loguru import logger
from threadpoolctl import threadpool_limits

from experiments.affine_heads.run import all_channels, append_features, feature_plan, leaves, tree_predictions
from experiments.guided.run import FAMILIES, SIZES, data, providers
from okapi.affine import fit_stacking_tree
from okapi.globals import set_backend, set_postprocessing_function
from okapi.tree import Tree


def fit_all(banks):
    frozen = []
    for family in FAMILIES:
        for bank in banks:
            source, y = data(family, bank, "fit")
            for precision in ("float64", "float32"):
                x = source.astype(precision, copy=False)
                original = leaves(providers(x))
                for form in ("flat_explicit", "flat_folded", "grouped_folded"):
                    extra = feature_plan(x, family, bank, -1)[0] if form == "grouped_folded" else []
                    expanded = append_features(x, [Tree(r.copy_subtree()).predict().ravel() for r in extra])
                    explicit = form == "flat_explicit"
                    roots = leaves(all_channels(x)) if explicit else [*original, *extra]
                    for transform, training in (("logit", StackingLogit), ("probability", StackingLR)):
                        ref = training(c_value=1.0, max_iter=2000).fit(expanded, y, task="binary", n_classes=2)
                        begin = perf_counter()
                        native = fit_stacking_tree(roots, y, transform=transform, complementary=not explicit, c_value=1.0, max_iter=2000)
                        fit_s = perf_counter() - begin
                        frozen.append((family, bank, precision, form, transform, ref, native, extra, fit_s))
    return frozen


def score(frozen, scores):
    indexed = scores.query("start==-1").set_index(["family", "bank", "split", "method"])
    rows = []
    for family, bank, precision, form, transform, ref, native, extra, fit_s in frozen:
        for split in SIZES:
            x, y = data(family, bank, split)
            x = x.astype(precision, copy=False)
            positive = providers(x)
            inputs = append_features(x, [tree_predictions(Tree(r.copy_subtree()), positive) for r in extra])
            expected = ref.predict(inputs)[:, 1]
            prediction = tree_predictions(native, all_channels(x) if form == "flat_explicit" else positive)
            difference = np.abs(prediction - expected)
            brier = np.mean((prediction - y) ** 2)
            method = ("grouped_" if extra else "flat_") + transform
            archived = indexed.loc[(family, bank, split, method), "brier"]
            rows.append(
                dict(
                    family=family,
                    bank=bank,
                    precision=precision,
                    form=form,
                    transform=transform,
                    split=split,
                    n=len(y),
                    native_fit_seconds=fit_s,
                    max_abs=difference.max(),
                    mean_abs=difference.mean(),
                    native_brier=brier,
                    public_brier=np.mean((expected - y) ** 2),
                    diff_from_float64_archive=brier - archived,
                )
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("experiments/affine_heads/results/scores.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--banks", nargs="+", type=int, default=list(range(31700, 31712)))
    args = parser.parse_args()
    logger.disable("okapi")
    set_backend("numpy")
    set_postprocessing_function(_identity)
    with threadpool_limits(1):
        result = score(fit_all(args.banks), pd.read_csv(args.archive))
    result.to_csv(args.out, index=False)
