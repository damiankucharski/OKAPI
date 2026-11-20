"""Run the fixed-aggregator experiment in trimmed_pool_protocol.md."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from fusionbench.methods.stacking import StackingLogit, StackingLR
from fusionbench.scenarios.committee import Committee
from scipy.stats import t

from okapi.globals import set_backend
from okapi.node import ValueNode
from okapi.operators import MEAN, TRIMMED_MEAN
from okapi.tree import Tree


def brier(predictions, labels):
    return float(np.mean((predictions - labels) ** 2))


def evaluate_tree(operator, predictions):
    columns = predictions[:, :, 1].T
    root = ValueNode(None, columns[0], 0)
    root.add_child(operator([ValueNode(None, column, i) for i, column in enumerate(columns[1:], 1)]))
    return Tree(root).predict()


def split(bank, tag, n):
    rng = np.random.default_rng([bank, tag])
    predictions, labels, _ = Committee(k=7, accuracy=.75)._generate(n, rng)
    return predictions, labels


def fault_conditions(clean, bank):
    rng = np.random.default_rng([bank, 4])
    n, k = clean.shape[:2]
    rows = np.arange(n)
    active = rows[rng.random(n) < .5]
    order = rng.random((n, k)).argsort(axis=1)
    stuck = rng.choice([.001, .999], size=(len(active), 1))
    yield "shift_clean", clean, 0
    for name, count in (("minority", 1), ("majority", 4)):
        indices = order[active, :count]
        damaged = clean.copy()
        damaged[active[:, None], indices, 1] = stuck
        damaged[active[:, None], indices, 0] = 1 - stuck
        yield name, damaged, len(active)
    specialist = np.full_like(clean, .5)
    specialist[rows, order[:, 0]] = clean[rows, order[:, 0]]
    yield "singleton", specialist, n


def select_stacker(bank):
    fit, fit_y = split(bank, 0, 512)
    val, val_y = split(bank, 1, 256)
    costs = []
    winner = None
    best_loss = np.inf
    for family in (StackingLR, StackingLogit):
        for regularization in (.1, 1., 10.):
            model = family(c_value=regularization)
            started = perf_counter()
            model.fit(fit, fit_y, task="binary", n_classes=2)
            elapsed = perf_counter() - started
            started = perf_counter()
            loss = brier(model.predict(val)[:, 1], val_y)
            validation_seconds = perf_counter() - started
            costs.append(dict(bank=bank, features=model.name, c=regularization,
                              fit_labels=len(fit_y), validation_labels=len(val_y),
                              solver_iterations=int(model.model[-1].n_iter_.sum()),
                              fit_seconds=elapsed, validation_seconds=validation_seconds,
                              validation_brier=loss))
            if loss < best_loss:
                winner, best_loss = model, loss
    for row in costs:
        row["selected"] = int((row["features"], row["c"]) == (winner.name, winner.c_value))
    return winner, costs


def run_bank(bank):
    stacker, costs = select_stacker(bank)
    methods = {
        "mean": lambda p: evaluate_tree(MEAN, p),
        "trim_one": lambda p: evaluate_tree(TRIMMED_MEAN, p),
        "median": lambda p: np.median(p[:, :, 1], axis=1),
        "stacker": lambda p: stacker.predict(p)[:, 1],
    }
    iid, iid_y = split(bank, 2, 4096)
    shift, shift_y = split(bank, 3, 4096)
    conditions = [("iid", iid, 0, iid_y)]
    conditions.extend((name, preds, affected, shift_y) for name, preds, affected in fault_conditions(shift, bank))
    results = []
    for condition, preds, affected, labels in conditions:
        for name, predict in methods.items():
            started = perf_counter()
            probabilities = predict(preds)
            elapsed = perf_counter() - started
            results.append(dict(bank=bank, condition=condition, method=name,
                                cases=len(labels), affected=affected, providers=preds.shape[1],
                                brier=brier(probabilities, labels),
                                accuracy=float(np.mean((probabilities >= .5) == labels)),
                                inference_seconds=elapsed))
    return results, costs


def summarize(results):
    per_cell = defaultdict(list)
    losses = {}
    for row in results:
        per_cell[row["condition"], row["method"]].append(row)
        losses[row["bank"], row["condition"], row["method"]] = row["brier"]
    summaries = []
    for (condition, method), rows in per_cell.items():
        delta = np.array([r["brier"] - losses[r["bank"], condition, "mean"] for r in rows])
        margin = t.ppf(.975, len(rows) - 1) * np.std(delta, ddof=1) / np.sqrt(len(rows))
        degradation = ""
        if condition != "iid":
            degradation = np.mean([r["brier"] - losses[r["bank"], "shift_clean", method] for r in rows])
        summaries.append(dict(condition=condition, method=method, banks=len(rows),
                              brier=np.mean([r["brier"] for r in rows]),
                              accuracy=np.mean([r["accuracy"] for r in rows]),
                              delta_vs_mean=delta.mean(), lower=delta.mean() - margin, upper=delta.mean() + margin,
                              degradation_vs_shift_clean=degradation,
                              mean_inference_ms=1000 * np.mean([r["inference_seconds"] for r in rows])))
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_backend("numpy")
    results, costs = [], []
    for bank in range(2300, 2316):
        bank_results, bank_costs = run_bank(bank)
        results.extend(bank_results)
        costs.extend(bank_costs)
    write_csv(args.out / "trimmed_pool_measurements.csv", results)
    write_csv(args.out / "trimmed_pool_stacker.csv", costs)
    write_csv(args.out / "trimmed_pool_summary.csv", summarize(results))


if __name__ == "__main__":
    main()
