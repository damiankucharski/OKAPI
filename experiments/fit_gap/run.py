"""Four matched splits for GP search, arithmetic pooling and regularized stacking."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from fusionbench.eval.search_stability import _scenario_pair, _tensor_map
from fusionbench.methods.okapi import _identity
from fusionbench.methods.stacking import StackingLogit
from loguru import logger
from scipy.stats import t
from threadpoolctl import threadpool_limits

from okapi.node import MaxNode, MeanNode, MinNode, ValueNode, WeightedMeanNode
from okapi.okapi import Okapi
from okapi.pareto import minimize
from okapi.tree import Tree


def brier(p, y):
    return float(np.mean((np.ravel(p) - np.ravel(y)) ** 2))


def draw(generator, seed, stream, size):
    return generator._generate(size, np.random.default_rng([seed, stream]))


def tensor_predictor(tree):
    def predict(x):
        return tree.do_pred_on_another_tensors(current_tensors=_tensor_map(x)).ravel()
    return predict


def feasible_tree(family, x):
    tensors = _tensor_map(x)
    nodes = [ValueNode(None, value, provider_id) for provider_id, value in tensors.items()]
    if family == "two_pathways":
        for start in (0, 2, 4, 6):
            nodes[start].add_child(MeanNode([nodes[start + 1]]))
        nodes[0].add_child(MinNode([nodes[2]]))
        nodes[4].add_child(MinNode([nodes[6]]))
        nodes[0].add_child(MaxNode([nodes[4]]))
    else:
        weights = np.array([1 / (1 + 7 * .85)] * 8 + [1., 1.])
        nodes[0].add_child(WeightedMeanNode(nodes[1:], (weights / weights.sum()).tolist()))
    return Tree(nodes[0])


class CountedOkapi(Okapi):
    calls = 0

    def _calculate_fitnesses(self, trees=None):
        self.calls += len(self.population if trees is None else trees)
        return super()._calculate_fitnesses(trees)


@dataclass
class Policy:
    predict: Callable[[np.ndarray], np.ndarray]
    calls: int = 0
    val_calls: int = 0
    fit_s: float = 0.
    selection_s: float = 0.
    nodes: int = 0
    chosen_seed: int = -1


def gp_checkpoints(x, y, seed):
    current = perf_counter()
    checkpoints = {}
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        paths = []
        for key, p in _tensor_map(x).items():
            path = directory / key
            np.save(path, p)
            paths.append(path)
        np.save(directory / "y.npy", y)
        gp = CountedOkapi(paths, directory / "y.npy", population_size=len(paths), population_multiplier=2,
                          tournament_size=2, objective_functions=(brier,), objectives=(minimize,),
                          seed=seed, backend="numpy", postprocessing_function=_identity)
        elapsed = perf_counter() - current
        for i in range(1, 49):
            current = perf_counter()
            gp.run_iteration()
            elapsed += perf_counter() - current
            if i in (8, 48):
                current = perf_counter()
                j = int(np.argmin(gp.pareto_fitnesses[:, 0]))
                tree = gp.pareto_trees[j].copy()
                extraction = perf_counter() - current
                checkpoints[i] = Policy(tensor_predictor(tree), gp.calls, fit_s=elapsed + extraction,
                                        nodes=tree.nodes_count, chosen_seed=seed)
    return checkpoints


def fit_policies(family, fit_x, fit_y, val_x, val_y, seed):
    seeds = [int(np.random.SeedSequence([seed, i, 82]).generate_state(1)[0]) for i in range(2)]
    a, b = [gp_checkpoints(fit_x, fit_y, search_seed) for search_seed in seeds]
    start = perf_counter()
    choices = [a[48], b[48]]
    losses = [brier(item.predict(val_x), val_y) for item in choices]
    champion = choices[int(np.argmin(losses))]
    selected = Policy(champion.predict, sum(p.calls for p in choices), 2,
                      sum(p.fit_s for p in choices), perf_counter() - start, champion.nodes, champion.chosen_seed)
    start = perf_counter()
    stack = StackingLogit(c_value=1.).fit(fit_x, fit_y, task="binary", n_classes=2)
    stack_s = perf_counter() - start
    witness = feasible_tree(family, fit_x)
    return {
        "mean": Policy(lambda x: x[:, :, 1].mean(axis=1)),
        "stack": Policy(lambda x: stack.predict(x)[:, 1], fit_s=stack_s),
        "gp8": a[8], "gp48": a[48], "gp48_seed2": b[48], "gp48_selected": selected,
        "feasible_witness": Policy(tensor_predictor(witness), nodes=witness.nodes_count),
    }


def one_bank(family, seed):
    source, shift = _scenario_pair(family)
    train_x, train_y, _ = draw(source, seed, 0, 1536)
    val_x, val_y, _ = draw(source, seed, 1, 256)
    by_size = {n: fit_policies(family, train_x[:n], train_y[:n], val_x, val_y, seed)
               for n in (96, 384, 1536)}
    # The tests and the generator oracle are opened only after all choices.
    tests = {}
    for name, generator, stream in (("iid", source, 2), ("shift", shift, 3)):
        x, y, latent = draw(generator, seed, stream, 4096)
        q = generator.bayes_oracle(x, latent)[:, 1]
        tests[name] = x, y, q
    rows = []
    for n, policies in by_size.items():
        splits = {"fit": (train_x[:n], train_y[:n], None), "validation": (val_x, val_y, None), **tests}
        for split, (x, y, q) in splits.items():
            lo, hi = x[:, :, 1].min(axis=1), x[:, :, 1].max(axis=1)
            floor = None if q is None else brier(np.clip(q, lo, hi), q)
            for method, policy in policies.items():
                start = perf_counter()
                p = policy.predict(x)
                predict_s = perf_counter() - start
                if method != "stack":
                    assert np.all((p >= lo - 1e-12) & (p <= hi + 1e-12))
                rows.append(dict(family=family, seed=seed, n=n, split=split, method=method,
                                 brier=brier(p, y), accuracy=np.mean((p >= .5) == y.ravel()),
                                 gp_fit_calls=policy.calls, selection_calls=policy.val_calls, chosen_seed=policy.chosen_seed,
                                 nodes=policy.nodes, fit_s=policy.fit_s, selection_s=policy.selection_s, predict_s=predict_s,
                                 rows=len(y), oracle_excess=None if q is None else brier(p, q), interval_floor=floor,
                                 outside_interval=np.mean((p < lo - 1e-12) | (p > hi + 1e-12))))
    return rows


def interval(d):
    m = float(t.ppf(.975, len(d) - 1) * d.sem())
    return dict(delta=d.mean(), lower=d.mean()-m, upper=d.mean()+m, banks=len(d), wins=int((d < 0).sum()))


def paired(frame):
    out = []
    for (family, n), f in frame.groupby(["family", "n"]):
        block = f.pivot(index="seed", columns=["method", "split"], values="brier")
        for a, b in (("gp48", "stack"), ("gp48", "mean"), ("gp48", "gp8"), ("gp48_selected", "gp48"),
                     ("feasible_witness", "gp48"), ("stack", "mean")):
            diffs = {}
            for split in ("fit", "validation", "iid", "shift"):
                diffs[split] = block[a, split] - block[b, split]
            diffs["iid_gap_minus_fit_gap"] = diffs["iid"] - diffs["fit"]
            for split, d in diffs.items():
                out.append(dict(family=family, n=n, first=a, second=b, split=split, **interval(d)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(24100, 24112)))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    logger.disable("okapi")
    with threadpool_limits(1):
        frame = pd.DataFrame([row for family in ("two_pathways", "correlated") for seed in args.seeds for row in one_bank(family, seed)])
    frame.to_csv(args.out / "measurements.csv", index=False)
    means = frame.groupby(["family", "n", "method", "split"], as_index=False)[["brier", "accuracy", "gp_fit_calls", "selection_calls", "nodes", "fit_s", "selection_s"]].mean()
    means.to_csv(args.out / "means.csv", index=False)
    paired(frame).to_csv(args.out / "paired.csv", index=False)
