"""Audit target-free joins of intermediate native MIN/MAX values in archived programs."""

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from experiments.donor_opportunity.check import canonical_signatures

SETS = ("root", "subtree", "raw", "prefix")
FOCAL = (1, 5, 10, 15)
LEAVES = {f"p{i}.npy": sum(1 << corner for corner in range(16) if corner & (1 << i)) for i in range(4)}


@dataclass
class Program:
    root: int
    banks: dict
    has_proper: bool


def explain(tree):
    completed, inputs, prefixes = {}, {}, {}

    def value(node, path):
        if node["type"] != "ValueNode":
            raise ValueError("Expected alternating value and operator nodes")
        current = LEAVES[node["id"]]
        inputs.setdefault(current, (path, 0))
        operations = node["children"]
        for j, op in enumerate(operations):
            kind = op["type"]
            if kind not in ("MinNode", "MaxNode"):
                raise ValueError("Only MIN/MAX are supported")
            for k, operand in enumerate(op["children"]):
                child = value(operand, (*path, j, k))
                current = current & child if kind == "MinNode" else current | child
            if j + 1 < len(operations):
                prefixes.setdefault(current, (path, j + 1))
        completed.setdefault(current, (path, len(operations)))
        return current

    root = value(tree, ())
    with_inputs = inputs | completed
    banks = {
        "root": {root: ((), len(tree["children"]))},
        "subtree": completed,
        "raw": with_inputs,
        "prefix": prefixes | with_inputs,
    }
    return Program(root, banks, bool(prefixes))


def exposure(programs, source, target, halves):
    available = {kind: {} for kind in SETS}
    proper_sources = 0
    for i in set(source):
        p = programs[i]
        if p.root == target:
            continue
        proper_sources += int(p.has_proper)
        for kind in SETS:
            for mask in p.banks[kind]:
                if mask != target:
                    available[kind].setdefault(mask, set()).add(i)
    earlier = available["raw"]
    extended = available["prefix"]
    new_functions = len(extended.keys() - earlier.keys())
    new_sources = sum(len(extended[mask] - earlier[mask]) for mask in earlier)
    result = dict(proper_sources=proper_sources, novel_functions=new_functions, extra_sources_for_known=new_sources)
    if halves:
        for kind, donors in available.items():
            first, second = (donors.get(h, set()) for h in halves)
            result[kind + "_halves"] = int(bool(first)) + int(bool(second))
            result[kind + "_distinct"] = any(i != j for i in first for j in second)
    return result


def capabilities(programs, source, target):
    result = {}
    for kind in SETS:
        providers = {}
        for index in source:
            p = programs[index]
            if p.root != target:
                for mask, spot in p.banks[kind].items():
                    if mask != target:
                        providers.setdefault(mask, {}).setdefault(index, spot)
        first = None
        for a, b in combinations(sorted(providers), 2):
            if (a | b) != target and (a & b) != target:
                continue
            parents = next(((i, j) for i in providers[a] for j in providers[b] if i != j), None)
            if parents is not None:
                i, j = parents
                first = dict(op="MAX" if a | b == target else "MIN", left=[i, a, *providers[a][i]], right=[j, b, *providers[b][j]])
                break
        result[kind] = first
    flags = [w is not None for w in result.values()]
    assert flags == sorted(flags)
    return result


def recorded(directory):
    with gzip.open(directory / "focal.csv.gz", "rt") as inp:
        records = [row for row in csv.DictReader(inp) if row["target"] == "population"]
    return {(r["law"], int(r["data_seed"]), int(r["gp_seed"]), int(r["generation"])): r for r in records}


def audit(directory):
    locked, rules = recorded(directory), canonical_signatures()
    cases, evidence, exposures = [], [], []
    with gzip.open(directory / "traces.jsonl.gz", "rt") as inp:
        for line in inp:
            trace = json.loads(line)
            law, bank, seed = trace["law"], trace["data_seed"], trace["gp_seed"]
            programs = [explain(record) for record in trace["programs"]]
            assert [p.root for p in programs] == [rules[i] for i in trace["rule_ids"]]
            target = rules[trace["goals"]["population_rule"]]
            halves = None
            if law == "matched":
                roles = np.random.default_rng([bank, 903]).permutation(4)
                a, b, c, d = (LEAVES[f"p{i}.npy"] for i in roles)
                halves = (a & b, c & d)
                assert target == halves[0] | halves[1]
            archive = list(trace["initial_scores"])
            for g in trace["generations"]:
                generation = g["generation"]
                if generation > max(FOCAL):
                    break
                assert len(archive) == g["calls_before"]
                if generation in FOCAL:
                    ref = locked[law, bank, seed, generation]
                    for scope, ids in (("live", g["parents"]), ("history", list(dict.fromkeys(archive)))):
                        joins = capabilities(programs, ids, target)
                        key = dict(law=law, bank=bank, gp_seed=seed, generation=generation, scope=scope)
                        exposures.append(dict(**key, **exposure(programs, ids, target, halves)))
                        flags = {kind: joins[kind] is not None for kind in SETS}
                        cases.append(
                            dict(
                                **key,
                                target=target,
                                programs=len(ids),
                                target_free_programs=sum(programs[i].root != target for i in ids),
                                calls_before=len(archive),
                                **flags,
                            )
                        )
                        if scope == "live":
                            for kind in SETS[:2]:
                                base = kind + "_joins_parents_free_strict_"
                                assert flags[kind] == bool(int(ref[base + "union_pairs"]) + int(ref[base + "intersection_pairs"]))
                        for kind in ("raw", "prefix"):
                            if flags[kind] and not flags[SETS[SETS.index(kind) - 1]]:
                                witness = joins[kind]
                                ids_used = (witness["left"][0], witness["right"][0])
                                evidence.append(
                                    dict(**key, kind=kind, target=target, join=witness, source_programs=[trace["programs"][i] for i in ids_used])
                                )
                archive.extend(g["scored"])
    assert len(cases) == 2 * 16 * 2 * 4 * 2
    return pd.DataFrame(cases), evidence, pd.DataFrame(exposures)


def exposure_summary(rows):
    out = []
    for (law, generation, scope), block in rows.groupby(["law", "generation", "scope"]):
        row = dict(law=law, generation=generation, scope=scope, starts=len(block))
        for metric in ("proper_sources", "novel_functions", "extra_sources_for_known"):
            row[metric + "_present"] = int(block[metric].gt(0).sum())
        if law == "matched":
            for kind in SETS:
                for count in (0, 1, 2):
                    row[f"{kind}_halves_{count}"] = int(block[kind + "_halves"].eq(count).sum())
                row[kind + "_both_distinct"] = int(block[kind + "_distinct"].sum())
        out.append(row)
    return pd.DataFrame(out)


def summarise(cases):
    out = []
    for (law, generation, scope), block in cases.groupby(["law", "generation", "scope"]):
        key = dict(law=law, generation=generation, scope=scope, starts=len(block))
        counts = {label: int(block[label].sum()) for label in SETS}
        extra = block.prefix & ~block.subtree
        banks = extra.groupby(block.bank).mean()
        assert len(banks) == 16
        half_width = t.ppf(0.975, 15) * banks.std(ddof=1) / 4
        out.append(
            dict(
                **key,
                **counts,
                extra=int(extra.sum()),
                raw_extra=int((block.raw & ~block.subtree).sum()),
                proper_extra=int((block.prefix & ~block.raw).sum()),
                mean_extra=banks.mean(),
                low=banks.mean() - half_width,
                high=banks.mean() + half_width,
                banks_with_extra=int(banks.gt(0).sum()),
            )
        )
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows, witnesses, exposures = audit(args.directory)
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "availability.csv", index=False)
    summarise(rows).to_csv(args.output / "summary.csv", index=False)
    exposures.to_csv(args.output / "exposure.csv", index=False)
    exposure_summary(exposures).to_csv(args.output / "exposure_summary.csv", index=False)
    (args.output / "witnesses.json").write_text(json.dumps(witnesses, indent=2) + "\n")
