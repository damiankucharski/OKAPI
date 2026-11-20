"""Audit equivalent OKAPI programs on FusionBench's Compound task.

Run from the OKAPI checkout with FusionBench installed:
    python -m experiments.operator_equivalence --output audit.csv
"""
import argparse
import csv
from itertools import permutations, product
from pathlib import Path

import numpy as np
from fusionbench.scenarios.gates import Compound

from okapi.globals import set_backend
from okapi.interventions import binary_decision_changes, predict_with_bypass, predict_with_substitution
from okapi.node import MaxNode, MeanNode, MinNode, ValueNode
from okapi.tree import Tree

OPS = {"max": MaxNode, "min": MinNode, "mean": MeanNode}


def programs(names):
    # Up to two sequential updates with distinct provider IDs. Include the
    # unmodified detectors and all shorter programs in model selection.
    for n_updates in range(3):
        for order in permutations(names, n_updates + 1):
            # First binary update commutes, so avoid scoring both serializations.
            if n_updates and order[0] > order[1]:
                continue
            for ops in product(OPS, repeat=n_updates):
                yield order, ops


def tree_for(order, ops, scores):
    root = ValueNode(None, scores[order[0]], order[0])
    for name, op in zip(order[1:], ops, strict=True):
        root.add_child(OPS[op]([ValueNode(None, scores[name], name)]))
    return Tree(root)


def fit_program(dataset):
    scores = dict(zip(dataset.model_ids, dataset.fit_preds[:, :, 1].T, strict=True))
    def criterion(program):
        order, ops = program
        prob = tree_for(order, ops, scores).predict()
        return np.mean((prob - dataset.fit_y) ** 2), len(ops), order, ops
    return min(programs(dataset.model_ids), key=criterion)


def describe(tree, paths, y, intact):
    bypass = predict_with_bypass(tree, paths)
    diff = binary_decision_changes(intact, bypass, y)
    return int(diff.repairs.sum()), int(diff.harms.sum())


def audit_order(order, ops, scores, y, permutation):
    full_tree = tree_for(order, ops, scores)
    prob = full_tree.predict()
    metrics = {"accuracy": float(np.mean((prob >= 0.5) == y)), "brier": float(np.mean((prob - y) ** 2))}
    if not ops:
        return [dict(form="single", intervention="none", target="none", repairs=0, harms=0, **metrics)]
    results = []
    if ops[0] in ("min", "max"):
        # Duplicating the first idempotent update leaves every score unchanged.
        repeat = tree_for((order[0], order[1], *order[1:]), (ops[0], *ops), scores)
        np.testing.assert_array_equal(prob, repeat.predict())
        equality = predict_with_bypass(full_tree, [(0,)])
        np.testing.assert_array_equal(equality, predict_with_bypass(repeat, [(0,), (1,)]))
        variants = {"original": (full_tree, [[(0,)]]),
                    "duplicate": (repeat, [[(0,)], [(1,)], [(0,), (1,)]])}
    else:
        variants = {"original": (full_tree, [[(0,)]])}
    for form, (tree, groups) in variants.items():
        for paths in groups:
            repairs, harms = describe(tree, paths, y, prob)
            results.append(dict(form=form, intervention="bypass", target="+".join(str(p[0]) for p in paths),
                                repairs=repairs, harms=harms, **metrics))
        for provider in order:
            swapped = predict_with_substitution(tree, {provider: scores[provider][permutation]})
            intact_substitution = predict_with_substitution(full_tree, {provider: scores[provider][permutation]})
            np.testing.assert_array_equal(swapped, intact_substitution)
            changes = binary_decision_changes(prob, swapped, y)
            results.append(dict(form=form, intervention="permutation", target=provider,
                                repairs=int(changes.repairs.sum()), harms=int(changes.harms.sum()), **metrics))
    return results


def run(seeds, n_fit, n_test):
    set_backend("numpy")
    rows = []
    for seed in seeds:
        scenario = Compound(rep=1)
        fit = scenario.build(n_fit, seed=seed)
        order, ops = fit_program(fit)
        for domain, task in [("iid", scenario), ("weaker_rescue", Compound(rep=1, acc_rescue=0.55))]:
            # Same random draws across domains, changing only the rescue channel strength.
            test = task.build(n_test, seed=100_000 + seed)
            scores = dict(zip(test.model_ids, test.test_preds[:, :, 1].T, strict=True))
            orders = [order]
            if ops:
                swapped = (order[1], order[0], *order[2:])
                np.testing.assert_array_equal(tree_for(order, ops, scores).predict(), tree_for(swapped, ops, scores).predict())
                orders.append(swapped)
            permutation = np.random.default_rng(200_000 + seed).permutation(n_test)
            for variant in orders:
                for result in audit_order(variant, ops, scores, test.test_y, permutation):
                    rows.append(dict(seed=seed, domain=domain, n_fit=n_fit, n_test=n_test,
                                     providers=",".join(variant), operators=",".join(ops),
                                     oracle_accuracy=float(np.mean((test.test_oracle[:, 1] >= 0.5) == test.test_y)),
                                     **result))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12, help="number of seeds, starting at zero")
    parser.add_argument("--n-fit", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seeds, args.n_fit, args.n_test) < 1:
        parser.error("all sample and seed counts must be positive")
    rows = run(range(args.seeds), args.n_fit, args.n_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
