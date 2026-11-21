"""Verify two archived live populations via native crossover on four-input Boolean corners."""

import argparse
import csv
import gzip
import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

from experiments.donor_opportunity.native_kernel import kernel
from okapi.globals import set_backend, use_postprocessing_function
from okapi.node import MaxNode, MinNode, ValueNode
from okapi.tree import Tree

CORNERS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)


def signature(tree):
    with use_postprocessing_function(lambda values: values):
        predicted = tree.predict().ravel()
    assert np.all((predicted == 0) | (predicted == 1))
    return sum(int(value) << i for i, value in enumerate(predicted))


def leaf(name):
    index = int(name.removeprefix("p").removesuffix(".npy"))
    return ValueNode(None, CORNERS[:, index : index + 1], name)


def decode(record):
    children = [decode(c) for c in record["children"]]
    if record["type"] == "ValueNode":
        node = leaf(record["id"])
        for child in children:
            node.add_child(child)
        return node
    return {"MinNode": MinNode, "MaxNode": MaxNode}[record["type"]](children)


@lru_cache(maxsize=1)
def canonical_signatures():
    # Enumerate output bit patterns; monotonicity can be checked along the hypercube's edges.
    edges = [(i, i | (1 << bit)) for i in range(16) for bit in range(4) if not i & (1 << bit)]
    rules = []
    for truth in range(1, (1 << 16) - 1):
        if any((truth >> a) & 1 and not (truth >> b) & 1 for a, b in edges):
            continue
        minimal = [k for k in range(1, 16) if (truth >> k) & 1 and all(not ((truth >> a) & 1) for a, b in edges if b == k)]
        terms = tuple(sorted(tuple(i for i in range(4) if k & (1 << i)) for k in minimal))
        rules.append((sum(map(len, terms)), terms, truth))
    assert len(rules) == 166
    return tuple(truth for _, _, truth in sorted(rules))


def stats(parents, fitnesses, target, **caps):
    result = kernel(parents, np.asarray(fitnesses).reshape(len(parents), -1), signature=signature, target=target, **caps)
    return dict(
        target=target,
        parent_signatures=list(result.parent_signatures),
        parent_weights=[str(p) for p in result.parent_probs],
        tournament_productive=str(result.productive),
        uniform_productive=str(result.uniform_productive),
        productive_parent_pairs=[list(pair) for pair, branch in result.pairs.items() if branch.productive],
        pair_probabilities=[[str(result.pairs[i, j].productive) for j in range(len(parents))] for i in range(len(parents))],
        physical_draw_paths=sum(branch.n_draw_paths for branch in result.pairs.values()),
        distinct_ordered_child_signatures=len({pair for branch in result.pairs.values() for pair in branch.child_probs}),
    )


def fixture():
    a, b, c, d = [leaf(f"p{i}.npy") for i in range(4)]
    a.add_child(MinNode([b]))
    a.add_child(MaxNode([leaf("p2.npy")]))
    c.add_child(MinNode([d]))
    parents = [Tree(a), Tree(c)]
    truth = np.maximum(np.minimum(CORNERS[:, 0], CORNERS[:, 1]), np.minimum(CORNERS[:, 2], CORNERS[:, 3]))
    target = sum(int(v) << i for i, v in enumerate(truth))
    uncapped = stats(parents, [0.9, 0.9], target)
    unequal = stats(parents, [0.90, 0.91], target)
    small = stats(parents, [0.9, 0.9], target, max_nodes=5)
    assert uncapped["tournament_productive"] == uncapped["uniform_productive"] == "1/24"
    assert unequal["tournament_productive"] == "0" and unequal["uniform_productive"] == "1/24"
    assert small["tournament_productive"] == small["uniform_productive"] == "0"
    return {"tied_uncapped": uncapped, "unequal_uncapped": unequal, "tied_cap5": small}


def archived(directory):
    rules = canonical_signatures()
    focal = {}
    with gzip.open(directory / "traces.jsonl.gz", "rt") as inp:
        for line in inp:
            entry = json.loads(line)
            if (entry["data_seed"], entry["gp_seed"]) != (6603, 821):
                continue
            (stage,) = [v for v in entry["generations"] if v["generation"] == 15]
            parents = [Tree(decode(entry["programs"][i])) for i in stage["parents"]]
            target = rules[entry["goals"]["population_rule"]]
            for tree, i in zip(parents, stage["parents"], strict=True):
                assert signature(tree) == rules[entry["rule_ids"][i]]
            if entry["law"] == "matched":
                roles = np.random.default_rng([6603, 903]).permutation(4)
                a, b, c, d = CORNERS[:, roles].T
                generated = np.maximum(np.minimum(a, b), np.minimum(c, d))
                assert target == sum(int(v) << i for i, v in enumerate(generated))
            observed = stats(parents, stage["parent_fitness"], target)
            assert 0 <= Fraction(observed["tournament_productive"]) <= 1
            assert target not in observed["parent_signatures"]
            observed["fit_calls_before"] = stage["calls_before"]
            focal[entry["law"]] = observed
    assert set(focal) == {"matched", "additive"}
    with gzip.open(directory / "focal.csv.gz", "rt") as inp:
        reported = [
            r for r in csv.DictReader(inp) if (r["data_seed"], r["gp_seed"], r["generation"], r["target"]) == ("6603", "821", "15", "population")
        ]
    assert len(reported) == 2
    for row in reported:
        result = focal[row["law"]]
        assert result["parent_signatures"] == json.loads(row["parent_signatures"])
        np.testing.assert_allclose([float(Fraction(v)) for v in result["parent_weights"]], json.loads(row["parent_weights"]), rtol=0, atol=2e-15)
        probabilities = [[float(Fraction(v)) for v in branch] for branch in result["pair_probabilities"]]
        np.testing.assert_allclose(probabilities, json.loads(row["per_pair"]), rtol=0, atol=2e-15)
        assert abs(float(Fraction(result["tournament_productive"])) - float(row["p_tournament_new"])) < 2e-15
        assert abs(float(Fraction(result["uniform_productive"])) - float(row["p_uniform_new"])) < 2e-15
    return focal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    args = parser.parse_args()
    logger.remove()
    set_backend("numpy")
    result = {"fixture": fixture()}
    if args.directory:
        result["archive_6603_821_G15"] = archived(args.directory)
    print(json.dumps(result, indent=2))
