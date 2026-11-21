"""Replay every distinct pruned prefix as a native tree on corners and four levels."""

import argparse
import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import t

from experiments.donor_opportunity.check import canonical_signatures, decode, signature
from okapi.globals import set_backend, use_postprocessing_function
from okapi.tree import Tree

LEVELS = np.array([.125, .375, .625, .875])
GRID = np.array(list(product(LEVELS, repeat=4)))
GRID_INPUTS = {f'p{i}.npy': GRID[:, i:i+1] for i in range(4)}
THRESHOLD_CORNERS = (GRID[:, None, :] >= LEVELS[None, :, None]) @ (1 << np.arange(4))
FAMILIES = ('root', 'final', 'raw', 'all')


def serialized(node):
    return json.dumps(node, sort_keys=True, separators=(',', ':'))


@lru_cache(maxsize=None)
def native_mask(program):
    tree = Tree(decode(json.loads(program)))
    mask = signature(tree)
    with use_postprocessing_function(lambda x: x):
        actual = tree.do_pred_on_another_tensors(current_tensors=GRID_INPUTS).ravel()
    expected = ((mask >> THRESHOLD_CORNERS) & 1) @ np.diff(np.r_[0., LEVELS])
    np.testing.assert_array_equal(actual, expected)
    return mask


def occurrences(node):
    """Physically shorten this ValueNode; preserve all operands of its retained operations."""
    assert node['type'] == 'ValueNode'
    operations = node['children']
    for j in range(len(operations) + 1):
        cut = {**node, 'children': operations[:j]}
        yield j, len(operations), serialized(cut)
    for op in operations:
        assert op['type'] in ('MinNode', 'MaxNode')
        for operand in op['children']:
            yield from occurrences(operand)


@dataclass(frozen=True)
class Donors:
    root_mask: int
    modes: tuple[frozenset, ...]


def donor_sets(program, proper):
    final, raw, intermediate = set(), set(), set()
    root = native_mask(serialized(program))
    for j, n, key in occurrences(program):
        value = native_mask(key)
        if j == n:
            final.add(value)
        if j == 0:
            raw.add(value)
        if 0 < j < n:
            proper.add(key)
            intermediate.add(value)
    return Donors(root, tuple(map(frozenset, ({root}, final, final | raw, final | raw | intermediate))))


@lru_cache(maxsize=None)
def partners(target):
    # Direct comparison of 16-element Boolean vectors for all 166×166 pairs.
    # Root and donor target exclusions are enforced separately at the source level.
    lattice = np.array(canonical_signatures())
    values = ((lattice[:, None] >> np.arange(16)) & 1).astype(bool)
    truth = ((target >> np.arange(16)) & 1).astype(bool)
    min_ok = np.all(np.minimum(values[:, None], values[None]) == truth, axis=2)
    max_ok = np.all(np.maximum(values[:, None], values[None]) == truth, axis=2)
    return {int(mask): set(lattice[min_ok[i] | max_ok[i]].tolist()) for i, mask in enumerate(lattice)}


def available(programs, source_indices, target):
    seen = [set() for _ in FAMILIES]
    hit = [False for _ in FAMILIES]
    matches = partners(target)
    for index in dict.fromkeys(source_indices):
        donor = programs[index]
        if donor.root_mask == target:
            continue
        for k, subset in enumerate(donor.modes):
            retained = subset - {target}
            if any(seen[k].intersection(matches[mask]) for mask in retained):
                hit[k] = True
            # Update only after querying, so two donors from the same source cannot join.
            seen[k].update(retained)
    assert hit == sorted(hit)
    return hit


def fixture():
    def plain(i):
        return dict(type='ValueNode', id=f'p{i}.npy', children=[])

    def construct(i, j):
        root = plain(i)
        root['children'] = [dict(type='MinNode', children=[plain(j)]), dict(type='MaxNode', children=[plain(i)])]
        return donor_sets(root, set())

    roots = [construct(0, 1), construct(2, 3)]
    target = np.minimum(GRID[:, 0], GRID[:, 1])
    expected = np.maximum(target, np.minimum(GRID[:, 2], GRID[:, 3]))
    corners = np.array(list(product((0., 1.), repeat=4)))[:, ::-1]
    mask = sum(int(max(min(a, b), min(c, d))) << j for j, (a, b, c, d) in enumerate(corners))
    np.testing.assert_array_equal(expected, ((mask >> THRESHOLD_CORNERS) & 1) @ np.diff(np.r_[0., LEVELS]))
    assert available(roots, (0, 1), mask) == [False, False, False, True]
    assert available(roots, (0, 0), mask) == [False] * 4


def verify(trace_file):
    logger.remove()
    set_backend('numpy')
    fixture()
    native_mask.cache_clear()
    proper, program_keys = set(), set()
    snapshots = []
    archive_count = recorded_evaluations = program_count = 0
    rule_masks = canonical_signatures()
    with gzip.open(trace_file, 'rt') as records:
        for line in records:
            r = json.loads(line)
            archive_count += 1
            programs = [donor_sets(p, proper) for p in r['programs']]
            keys = [serialized(p) for p in r['programs']]
            assert len(set(keys)) == len(keys)
            program_keys.update(keys)
            program_count += len(programs)
            assert [p.root_mask for p in programs] == [rule_masks[j] for j in r['rule_ids']]
            target = rule_masks[r['goals']['population_rule']]
            chronology = list(r['initial_scores'])
            for stage in r['generations']:
                assert len(chronology) == stage['calls_before']
                assert set(stage['past_rules']) == {r['rule_ids'][idx] for idx in chronology}
                if stage['generation'] in (1, 5, 10, 15):
                    for scope, indices in (('live', stage['parents']), ('history', chronology)):
                        hit = available(programs, indices, target)
                        source_ids = set(indices)
                        snapshots.append(dict(law=r['law'], data_seed=r['data_seed'], gp_seed=r['gp_seed'],
                                              generation=stage['generation'], scope=scope, target=target, programs=len(source_ids),
                                              target_free_programs=sum(programs[i].root_mask != target for i in source_ids),
                                              calls_before=len(chronology), **dict(zip(FAMILIES, hit, strict=True))))
                chronology.extend(stage['scored'])
            assert chronology == r['archive'] and len(chronology) == r['calls']
            recorded_evaluations += len(chronology)
    df = pd.DataFrame(snapshots)
    assert archive_count == 64 and len(df) == 512
    focal = df.query("generation == 15 and scope == 'live' and law == 'matched'").sort_values(['data_seed', 'gp_seed']).copy()
    assert len(focal) == 32 and focal.groupby('data_seed').gp_seed.nunique().eq(2).all()
    gains = focal['all'].astype(int) - focal['final'].astype(int)
    bank_means = gains.groupby(focal['data_seed']).mean()
    half = float(t.ppf(.975, len(bank_means) - 1) * bank_means.sem())
    report = dict(archives=archive_count, chronological_calls=recorded_evaluations, program_dict_entries=program_count,
                  unique_source_genotypes=len(program_keys), unique_native_prefixes=native_mask.cache_info().currsize,
                  unique_strict_intermediate_genotypes=len(proper), grid_rows_per_prefix=len(GRID), corners_per_prefix=16,
                  snapshots=len(df), independent_booleans=len(df) * len(FAMILIES),
                  matched_live_G15={k: int(focal[k].sum()) for k in FAMILIES},
                  added_G15_starts=int(gains.sum()), added_G15_banks=int((bank_means > 0).sum()),
                  paired_mean=float(bank_means.mean()), paired_t95=[float(bank_means.mean() - half), float(bank_means.mean() + half)])
    if not gains.any():
        report['one_sided_95_upper_bank_pair_hit'] = 1 - .05 ** (1 / len(bank_means))
    return df, report


def compare(snapshot, reference):
    paired = pd.read_csv(reference)
    converted = snapshot.rename(columns={'data_seed': 'bank', 'final': 'subtree', 'all': 'prefix'})
    keys = ['law', 'bank', 'gp_seed', 'generation', 'scope']
    observed = converted.set_index(keys).sort_index()
    stored = paired.set_index(keys).sort_index()
    assert observed.index.is_unique and stored.index.is_unique
    fields = ['target', 'programs', 'target_free_programs', 'calls_before', 'root', 'subtree', 'raw', 'prefix']
    pd.testing.assert_frame_equal(observed[fields], stored[fields], check_exact=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path)
    parser.add_argument('--snapshots', type=Path, help='Write the independently computed Boolean snapshot table')
    parser.add_argument('--reference', type=Path, help='Compare every snapshot with the published availability.csv')
    args = parser.parse_args()
    frame, summary = verify(args.traces)
    if args.snapshots:
        frame.to_csv(args.snapshots, index=False)
    if args.reference:
        compare(frame, args.reference)
        summary['verified_against_reference'] = len(frame)
    print(json.dumps(summary, indent=2))
