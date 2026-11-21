"""Evaluate bounded template search when cross-provider correlation can be misleading."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy.special import expit, logit, ndtri
from scipy.stats import t
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from okapi.globals import set_backend, set_postprocessing_function
from okapi.guided import _average_link_cuts, _correlation_distances, _templates_from_cuts


def schema(bank):
    rng = np.random.default_rng([bank, 77])
    truth = np.repeat(np.arange(4), 2)
    while True:
        pairs = rng.permutation(8).reshape(4, 2)
        if np.all(truth[pairs[:, 0]] != truth[pairs[:, 1]]):
            break
    nuisance = np.empty(8, dtype=int)
    for j, pair in enumerate(pairs):
        nuisance[pair] = j
    order = rng.permutation(8)
    return truth[order], nuisance[order]


def draw(bank, stream, count, rho, assignments):
    truth, nuisance = assignments
    rng = np.random.default_rng([bank, stream, 881])
    signs = 2 * rng.integers(2, size=(count, 4)) - 1
    labels = ((np.minimum(signs[:, 0], signs[:, 1]) > 0) | (np.minimum(signs[:, 2], signs[:, 3]) > 0)).astype(int)
    noise = rng.normal(size=(count, 4))
    independent = rng.normal(size=(count, 8))
    s = ndtri(.78)
    observations = s*signs[:, truth] + np.sqrt(rho)*noise[:, nuisance] + np.sqrt(1-rho)*independent
    return expit(2*s*observations), labels


def tensor_map(x):
    return {f"anonymous{i}": x[:, i:i + 1] for i in range(x.shape[1])}


def stack_input(x):
    return np.stack((1 - x, x), axis=2)


def cuts_for(x, method):
    if method in ("pearson_p", "random", "oracle4"):
        return _average_link_cuts(_correlation_distances(x))
    log_odds = logit(x)
    if method == "pearson_logit":
        return _average_link_cuts(_correlation_distances(log_odds))
    z = (log_odds - log_odds.mean(axis=0)) / log_odds.std(axis=0)
    correlation = z.T @ z / len(z)
    square = z**2
    negative_cumulant = 1 + 2*correlation**2 - square.T @ square / len(z)
    return _average_link_cuts(1 - negative_cumulant)


@dataclass
class Policy:
    predict: Callable[[np.ndarray], np.ndarray]
    template_count: int = 0
    grouping_s: float = 0.
    fit_s: float = 0.
    partition: np.ndarray | None = None


def fit_library(method, x, y, seed, oracle_groups=None):
    values = tensor_map(x)
    start = perf_counter()
    cuts = cuts_for(x, method)
    if oracle_groups is not None:
        cuts[4] = tuple(tuple(np.flatnonzero(oracle_groups == j)) for j in range(4))
    random_seed = seed if method == "random" else None
    order = np.arange(x.shape[1]) if random_seed is None else np.random.default_rng(seed).permutation(x.shape[1])
    groups = np.empty(x.shape[1], dtype=int)
    for j, group in enumerate(cuts[4]):
        groups[order[list(group)]] = j
    grouping_s = perf_counter() - start
    library = _templates_from_cuts(list(values), list(values.values()), cuts, random_seed)
    if oracle_groups is not None:
        library = library[15:]
    losses = [np.mean((tree.predict().ravel() - y)**2) for tree in library]
    winner = library[int(np.argmin(losses))]
    elapsed = perf_counter() - start
    return Policy(lambda p: winner.do_pred_on_another_tensors(current_tensors=tensor_map(p)).ravel(), len(library),
                  grouping_s, elapsed, groups)


def run_bank(bank):
    assignments = schema(bank)
    policies = {}
    fits = {}
    for rho in (0., .90):
        fit_x, fit_y = draw(bank, 0, 1536, rho, assignments)
        for n in (384, 1536):
            x, y = fit_x[:n], fit_y[:n]
            fits[rho, n] = x, y
            items = {method: fit_library(method, x, y, bank) for method in ("pearson_p", "pearson_logit", "cumulant", "random")}
            start = perf_counter()
            stack = StackingLogit(c_value=1.).fit(stack_input(x), y, task="binary", n_classes=2)
            items["stack"] = Policy(lambda p, model=stack: model.predict(stack_input(p))[:, 1], fit_s=perf_counter()-start)
            items["mean"] = Policy(lambda p: p.mean(axis=1))
            policies[rho, n] = items
    # Only the diagnostic oracle gets the true grouping, after learners are frozen.
    rows = []
    for (rho, n), items in policies.items():
        x, y = fits[rho, n]
        items["oracle4"] = fit_library("oracle4", x, y, bank, oracle_groups=assignments[0])
        tests = {"fit": (x, y), "validation": draw(bank, 1, 256, rho, assignments),
                 "iid": draw(bank, 2, 4096, rho, assignments), "shift": draw(bank, 3, 4096, .90-rho, assignments)}
        for method, model in items.items():
            ari = None if model.partition is None else adjusted_rand_score(assignments[0], model.partition)
            for split, (p, labels) in tests.items():
                start = perf_counter()
                pred = model.predict(p)
                predict_s = perf_counter() - start
                rows.append(dict(bank=bank, rho=rho, fit_n=n, method=method, split=split, examples=len(labels),
                                 brier=np.mean((pred-labels)**2), accuracy=np.mean((pred >= .5)==labels),
                                 assessments=model.template_count, candidate_fit_rows=n*model.template_count, validation_choices=0,
                                 adjusted_rand=ari, exact_partition=ari == 1 if ari is not None else None,
                                 grouping_s=model.grouping_s, fit_s=model.fit_s, infer_s=predict_s))
    return rows


def summarize(frame):
    out = []
    for (rho, n, split), group in frame.groupby(["rho", "fit_n", "split"]):
        p = group.pivot(index="bank", columns="method", values="brier")
        for other in p.columns.drop("cumulant"):
            d = p.cumulant - p[other]
            half = t.ppf(.975, len(d)-1)*d.sem()
            out.append(dict(rho=rho, fit_n=n, split=split, reference=other, banks=len(d), delta=d.mean(),
                            lower=d.mean()-half, upper=d.mean()+half, wins=int((d < 0).sum())))
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--banks", type=int, nargs="+", default=list(range(25300, 25312)))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    logger.disable("okapi")
    set_backend("numpy")
    set_postprocessing_function(lambda p: p)
    with threadpool_limits(1):
        frame = pd.DataFrame([row for bank in args.banks for row in run_bank(bank)])
    frame.to_csv(args.out / "runs.csv", index=False)
    summarize(frame).to_csv(args.out / "paired.csv", index=False)
