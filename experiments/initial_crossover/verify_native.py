"""Replay every possible physical uncapped crossover on the archived eight-provider G0."""

import argparse
import ast
import csv
import gzip
import json
import re
from collections import defaultdict
from fractions import Fraction
from itertools import combinations_with_replacement
from math import comb
from pathlib import Path
from time import perf_counter

import numpy as np
from fusionbench.eval.lattice_signature import lattice_signature, provider_signatures
from loguru import logger

from experiments.donor_opportunity.native_kernel import child_distribution
from okapi.globals import set_backend, use_postprocessing_function
from okapi.node import MaxNode, MinNode, ValueNode
from okapi.tree import Tree

NAMES = tuple(f'p{i}.npy' for i in range(8))
CORNERS = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
BINDINGS = {name: CORNERS[:, i:i + 1] for i, name in enumerate(NAMES)}
MASKS = provider_signatures(NAMES)


def physical_signature(tree):
    prediction = tree.predict().ravel()
    assert prediction.shape == (256,) and np.isin(prediction, (0, 1)).all()
    mask = int.from_bytes(np.packbits(prediction, bitorder='little').tobytes(), 'little')
    assert mask == lattice_signature(tree, MASKS)
    return mask


def archived_tree(text):
    # Native structural keys are preorder tuples with a concrete class, child count
    # and exact parameter tuple. Parse just these literals, never evaluate source code.
    record = iter(ast.literal_eval(re.sub(r"<class '([^']+)'>", r"'\1'", text)))

    def unfold():
        kind, arity, parameters = next(record)
        children = [unfold() for _ in range(arity)]
        if kind == 'okapi.node.ValueNode':
            ((dtype, name),) = parameters
            assert dtype == 'str' and name in BINDINGS
            return ValueNode(children, BINDINGS[name], name)
        assert not parameters
        return {'okapi.node.MinNode': MinNode, 'okapi.node.MaxNode': MaxNode}[kind](children)

    tree = Tree(unfold())
    assert next(record, None) is None and repr(tree.structural_key()) == text
    return tree


def parents_from(record):
    inverse = {hex(signature): name for name, signature in MASKS.items()}
    singletons = [Tree(ValueNode([], BINDINGS[inverse[bits]], inverse[bits])) for bits in record['singleton_signatures']]
    seeded = [archived_tree(text) for text in record['seed_structures']]
    parents = singletons + seeded
    assert len(parents) == record['initial_calls'] == 38
    assert [hex(physical_signature(t)) for t in parents] == record['singleton_signatures'] + record['seed_signatures']
    return parents


def weights_from_fit(record, parents):
    bank, law = record['data_seed'], record['law']
    roles = np.random.default_rng([bank, 8, 401]).choice(8, 4, replace=False)
    assert roles.tolist() == json.loads(record['roles'])
    generator = np.random.default_rng([bank, 8, 0, 8093])
    digits = generator.choice(4, size=(256, 8))
    ticks = np.array([2, 7, 13, 18])[digits]
    a, b, c, d = ticks[:, roles].T
    target = np.maximum(np.minimum(a, b), np.minimum(c, d)) if law == 'branch' else np.minimum(np.maximum(np.minimum(a, b), c), d)
    y = 20 * (generator.random(256) < target / 20)
    bindings = {name: ticks[:, i:i + 1] for i, name in enumerate(NAMES)}
    # Native trees on integer ticks give exact 400*rows Brier numerators.
    scores = []
    for tree in parents:
        fit = tree.do_pred_on_another_tensors(current_tensors=bindings).ravel()
        scores.append(1 - np.sum((fit - y)**2) / (400 * len(y)))
    stored = np.round(scores, 3)
    weights = []
    for score in stored:
        worse = int((stored < score).sum())
        same_other = int((stored == score).sum()) - 1
        # Condition on this candidate appearing and j other equally good candidates.
        # Its random position wins one of j+1 ties; any remaining sampled parent is worse.
        weight = sum((Fraction(comb(same_other, j) * comb(worse, 4-j), (j+1) * comb(38, 5)) for j in range(5)), Fraction())
        weights.append(weight)
    assert sum(weights) == 1
    return weights


def verify(source, measurements):
    start = perf_counter()
    set_backend('numpy')
    logger.disable('okapi')
    with gzip.open(source / 'focal-seeds.jsonl.gz', 'rt') as stream:
        archives = [r for r in map(json.loads, stream) if r['law'] in ('branch', 'chain') and r['width'] == 8]
    with measurements.open() as stream:
        expected = {(r['law'], int(r['data_seed']), r['method'], int(r['gp_seed'])): r for r in csv.DictReader(stream)}
    by_geometry = defaultdict(list)
    for r in archives:
        by_geometry[r['method'], r['gp_seed']].append(r)
    assert len(by_geometry) == 6 and len(expected) == len(archives) == 144
    totals = dict(populations=0, unordered_parent_pairs=0, physical_swap_outcomes=0, child_truth_tables=0, individual_cells=0)
    with use_postprocessing_function(np.asarray):
        for records in by_geometry.values():
            first = records[0]
            for r in records:
                for key in ('singleton_signatures', 'seed_structures', 'seed_signatures'):
                    assert r[key] == first[key]
            parents = parents_from(first)
            original = [physical_signature(p) for p in parents]
            targets = {int(r['target_hex'], 16) for r in records}
            productive = defaultdict(list)
            for i, j in combinations_with_replacement(range(38), 2):
                outcomes, leaves = child_distribution(parents[i], parents[j], signature=physical_signature)
                strengths = defaultdict(Fraction)
                for children, probability in outcomes.items():
                    for hit in set(children).intersection(targets).difference((original[i], original[j])):
                        strengths[hit] += probability
                for target, chance in strengths.items():
                    productive[target].append((i, j, chance))
                totals['unordered_parent_pairs'] += 1
                totals['physical_swap_outcomes'] += leaves
                totals['child_truth_tables'] += 2 * leaves
            totals['populations'] += 1
            for r in records:
                cell = (r['law'], r['data_seed'], r['method'], r['gp_seed'])
                measured = expected[cell]
                target = int(r['target_hex'], 16)
                weights = weights_from_fit(r, parents)
                uniform = tournament = Fraction()
                legal = selected = 0
                for i, j, gain in productive[target]:
                    orders = 1 + int(i != j)
                    uniform += orders * gain / 38**2
                    tournament += orders * gain * weights[i] * weights[j]
                    legal += orders
                    if weights[i] and weights[j]:
                        selected += orders
                assert uniform == Fraction(measured['exact_uniform']) and tournament == Fraction(measured['exact_tournament']), cell
                assert float(uniform) == float(measured['p_uniform']) and float(tournament) == float(measured['p_tournament']), cell
                assert legal == int(measured['ordered_legal_pairs']) and selected == int(measured['positive_weight_pairs']), cell
                assert (target in original) == (measured['initial_hit'] == 'True'), cell
                totals['individual_cells'] += 1
    assert totals['unordered_parent_pairs'] == 6 * 741 and totals['individual_cells'] == 144
    totals['wall_seconds'] = round(perf_counter() - start, 3)
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='FusionBench docs/ensemble-study/seed-geometry directory')
    parser.add_argument('measurements', type=Path, nargs='?', default=Path(__file__).resolve().parent / 'results' / 'opportunities.csv')
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.measurements), sort_keys=True))
