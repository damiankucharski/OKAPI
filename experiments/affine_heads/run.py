"""Fitted parameter transfer and nonlinear feature controls on fixed, anonymous providers."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.methods.averaging import TopN
from fusionbench.methods.okapi import _identity
from fusionbench.methods.stacking import StackingLR, StackingLogit
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from experiments.guided.run import AVAILABLE_FAMILIES, FAMILIES, SIZES, data, providers
from experiments.parity_operator.run import native_control
from okapi.affine import stacking_tree
from okapi.globals import set_backend, set_postprocessing_function
from okapi.guided import _average_link_cuts, _correlation_distances
from okapi.node import ValueNode
from okapi.operators import BINARY_AFFINE, MAX, MEAN, MIN
from okapi.tree import Tree


def layer(operator, roots):
    parent, *children = (root.copy_subtree() for root in roots)
    parent.add_child(operator(children))
    return parent


def leaves(mapping):
    return [ValueNode(None, column, name) for name, column in mapping.items()]


def all_channels(x):
    return {f"m{i}c{j}": x[:, i, j : j + 1] for i in range(x.shape[1]) for j in (0, 1)}


def feature_plan(x, family, bank, seed):
    base = leaves(providers(x))
    cut = _average_link_cuts(_correlation_distances(x[:, :, 1]))[4]
    order = np.arange(len(base))
    if seed != -1:
        rng = np.random.default_rng([bank, AVAILABLE_FAMILIES.index(family), seed, 919])
        order = rng.permutation(len(base))
    groups = [[int(order[j]) for j in block] for block in cut]
    roots = []
    for group in groups:
        inputs = [base[j] for j in group]
        roots.append(layer(MEAN, inputs) if len(inputs) > 1 else inputs[0].copy_subtree())
    features = []
    for j, k, l in ((1, 2, 3), (2, 1, 3), (3, 1, 2)):
        a = layer(MIN, [roots[0], roots[j]])
        b = layer(MIN, [roots[k], roots[l]])
        features.append(layer(MAX, [a, b]))
    return features, groups


def append_features(x, values):
    if not values:
        return x
    positive = np.column_stack(values)
    return np.concatenate([x, np.stack([1.0 - positive, positive], axis=-1)], axis=1)


def tree_predictions(tree, bindings):
    return tree.do_pred_on_another_tensors(current_tensors=bindings).ravel()


def objective(z, y, coef, c):
    return np.mean(np.logaddexp(0.0, z) - y * z) + np.dot(coef, coef) / (2 * c * len(y))


@dataclass
class Fitted:
    method: str
    seed: int
    transform: str
    estimator: object
    folded: Tree
    explicit: Tree | None
    embedded: Tree | None
    features: list


def fit_bank(family, bank):
    x, y = data(family, bank, "fit")
    n, m, _ = x.shape
    original = leaves(providers(x))
    architectures = []
    for seed in (-1, 311, 751):
        begin = perf_counter()
        roots, groups = feature_plan(x, family, bank, seed)
        feature_counts = [Tree(r.copy_subtree()) for r in roots]
        matrix = append_features(x, [tr.predict().ravel() for tr in feature_counts])
        architectures.append((seed, roots, groups, matrix, perf_counter() - begin))
    fits, costs = [], []
    for transform, cls in (("probability", StackingLR), ("logit", StackingLogit)):
        base = None
        for seed, roots, groups, inputs, feature_s in [(-1, [], [], x, 0.0), *architectures]:
            method = ("grouped_" if seed == -1 else "shuffled_") + transform if roots else "flat_" + transform
            begin = perf_counter()
            model = cls(c_value=1.0, max_iter=2000).fit(inputs, y, task="binary", n_classes=2)
            fit_s = perf_counter() - begin
            begin = perf_counter()
            folded = stacking_tree([*original, *roots], model.model, transform=transform)
            explicit = None
            embedded = None
            if not roots:
                explicit = stacking_tree(leaves(all_channels(x)), model.model, transform=transform, complementary=False)
            else:
                old = base.folded.root.children[-1]
                root, *children = (node.copy_subtree() for node in [*original, *roots])
                coeff = [*old.coefficients, *([0.0] * len(roots))]
                root.add_child(BINARY_AFFINE(children, coeff, intercept=old.intercept, transform=transform))
                embedded = Tree(root)
            convert_s = perf_counter() - begin
            candidate = Fitted(method, seed, transform, model, folded, explicit, embedded, roots)
            if not roots:
                base = candidate
            fits.append(candidate)
            classifier = model.model.steps[-1][1]
            scaler = model.model.steps[0][1]
            standardized = scaler.transform(model._features(inputs))
            w = classifier.coef_.ravel()
            z = classifier.decision_function(standardized)
            flat_classifier = base.estimator.model.steps[-1][1]
            w0 = np.pad(flat_classifier.coef_.ravel(), (0, len(w) - flat_classifier.coef_.size))
            z0 = standardized @ w0 + flat_classifier.intercept_[0]
            flat_z = base.estimator.model.decision_function(base.estimator._features(x))
            np.testing.assert_allclose(z0, flat_z, rtol=2e-14, atol=2e-14)
            costs.append(
                dict(
                    family=family,
                    bank=bank,
                    method=method,
                    start=seed,
                    fit_labels=n,
                    solver_iterations=classifier.n_iter_[0],
                    C=model.c_value,
                    scalar_features=len(w),
                    negative_folded_coefficients=sum(a < 0 for a in folded.root.children[-1].coefficients),
                    folded_intercept=folded.root.children[-1].intercept,
                    nodes=folded.nodes_count,
                    initial_objective=objective(z0, y, w0, 1.0),
                    final_objective=objective(z, y, w, 1.0),
                    feature_evaluations=len(roots),
                    feature_row_scores=n * len(roots),
                    groups=str(groups),
                    feature_seconds=feature_s,
                    fit_seconds=fit_s,
                    transfer_seconds=convert_s,
                    validation_choices=0,
                )
            )
    begin = perf_counter()
    top = TopN(metric="brier").fit(x, y, task="binary", n_classes=2)
    costs.append(
        dict(
            family=family,
            bank=bank,
            method="topn",
            start=-1,
            fit_labels=n,
            selected_n=top.n_selected_,
            feature_evaluations=2 * m,
            feature_row_scores=2 * m * n,
            fit_seconds=perf_counter() - begin,
            validation_choices=0,
        )
    )
    native = []
    for seed in (311, 751):
        tree, cost = native_control(x, y, seed)
        native.append((seed, tree))
        costs.append(
            dict(
                family=family,
                bank=bank,
                method="native_160",
                start=seed,
                fit_labels=n,
                nodes=tree.nodes_count,
                tree_calls=160,
                tree_row_scores=160 * n,
                fit_seconds=cost["fit_seconds"],
                validation_choices=0,
            )
        )
    costs.append(dict(family=family, bank=bank, method="mean", start=-1, fit_labels=0, fit_seconds=0.0, validation_choices=0))
    return (fits, top, native), costs


def record(rows, common, method, seed, p, y, seconds):
    p = np.asarray(p).ravel()
    rows.append(
        dict(
            **common,
            method=method,
            start=seed,
            n=len(y),
            brier=np.mean((p - y) ** 2),
            logloss=-np.mean(y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))),
            accuracy=np.mean((p > 0.5) == y),
            predict_seconds=seconds,
        )
    )


def evaluate(frozen):
    rows, errors = [], []
    for (family, bank), (fits, top, native) in frozen.items():
        for split in SIZES:
            x, y = data(family, bank, split)
            bindings = providers(x)
            common = dict(family=family, bank=bank, split=split)
            begin = perf_counter()
            p = x[:, :, 1].mean(axis=1)
            record(rows, common, "mean", -1, p, y, perf_counter() - begin)
            begin = perf_counter()
            p = top.predict(x)[:, 1]
            record(rows, common, "topn", -1, p, y, perf_counter() - begin)
            for seed, tree in native:
                begin = perf_counter()
                p = tree_predictions(tree, bindings)
                record(rows, common, "native_160", seed, p, y, perf_counter() - begin)
            flat_predictions = {}
            for fit in fits:
                begin = perf_counter()
                p = tree_predictions(fit.folded, bindings)
                record(rows, common, fit.method, fit.seed, p, y, perf_counter() - begin)
                ref_inputs = append_features(x, [tree_predictions(Tree(r.copy_subtree()), bindings) for r in fit.features])
                begin = perf_counter()
                ref = fit.estimator.predict(ref_inputs)[:, 1]
                reference_s = perf_counter() - begin
                pairs = [("folded_vs_public", p, ref)]
                if fit.explicit is not None:
                    explicit = tree_predictions(fit.explicit, all_channels(x))
                    pairs.append(("explicit_vs_public", explicit, ref))
                    flat_predictions[fit.transform] = ref
                    record(rows, common, "public_" + fit.transform, fit.seed, ref, y, reference_s)
                if fit.embedded is not None:
                    retained = tree_predictions(fit.embedded, bindings)
                    pairs.append(("zero_extension_vs_flat", retained, flat_predictions[fit.transform]))
                for comparison, left, right in pairs:
                    d = np.abs(left - right)
                    errors.append(
                        dict(**common, method=fit.method, start=fit.seed, comparison=comparison, n=len(y), max_abs=d.max(), mean_abs=d.mean())
                    )
    return pd.DataFrame(rows), pd.DataFrame(errors)


def paired(frame):
    out = []
    for (family, split), block in frame.groupby(["family", "split"]):
        by_bank = block.pivot_table(index="bank", columns="method", values="brier")
        for transform in ("probability", "logit"):
            arm = "grouped_" + transform
            for ref in ("public_" + transform, "shuffled_" + transform, "mean", "topn", "native_160"):
                delta = by_bank[arm] - by_bank[ref]
                margin = t.ppf(0.975, len(delta) - 1) * delta.sem()
                out.append(
                    dict(
                        family=family,
                        split=split,
                        method=arm,
                        reference=ref,
                        banks=len(delta),
                        delta=delta.mean(),
                        lower=delta.mean() - margin,
                        upper=delta.mean() + margin,
                        wins=(delta < 0).sum(),
                    )
                )
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--banks", type=int, nargs="+", default=list(range(31700, 31712)))
    opts = parser.parse_args()
    logger.disable("okapi")
    set_backend("numpy")
    set_postprocessing_function(_identity)
    frozen, costs = {}, []
    with threadpool_limits(1):
        for family in FAMILIES:
            for bank in opts.banks:
                row, c = fit_bank(family, bank)
                frozen[family, bank] = row
                costs.extend(c)
        scores, errors = evaluate(frozen)
    opts.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("scores", scores), ("transfer", errors), ("cost", pd.DataFrame(costs)), ("paired", paired(scores))):
        table.to_csv(opts.out / (name + ".csv"), index=False)
