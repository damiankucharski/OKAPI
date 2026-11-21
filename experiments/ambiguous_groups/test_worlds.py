from itertools import combinations

import numpy as np

from experiments.ambiguous_groups.worlds import Design, make_design, perfect_matchings, rank_matchings, sample


def test_cycle_supports_exactly_two_matchings_and_provider_ids_are_exchangeable():
    candidates = perfect_matchings()
    assert len(candidates) == len(set(candidates)) == 105
    assert candidates == sorted(candidates)
    for matching in candidates:
        assert matching == tuple(sorted(matching))
        assert all(a < b for a, b in matching)
        assert sorted(i for pair in matching for i in pair) == list(range(8))
    seen = {"A": set(), "H": set()}
    for seed in range(64):
        design = make_design(seed)
        hidden = {design.matching(w) for w in ("A", "H")}
        cycle_edges = set().union(*hidden)
        assert len(cycle_edges) == 8
        assert {m for m in candidates if set(m) <= cycle_edges} == hidden
        for w in seen:
            seen[w].update(design.matching(w))
    # Every sensor pair sometimes belongs to either world after scrambling.
    assert seen["A"] == seen["H"] == set(combinations(range(8), 2))


def test_exchanging_worlds_preserves_every_observed_bit_and_exchanges_labels():
    design = make_design(19)
    x, labels = sample(design, 4, 2, 4096, .5)
    exchanged = Design(design.h_paths, design.a_paths)
    same_x, other = sample(exchanged, 4, 2, 4096, .5)
    assert x.shape == (4096, 8, 2)
    np.testing.assert_array_equal(x, same_x)
    np.testing.assert_array_equal(x.sum(axis=-1), 1)
    for before, after in (("A", "H"), ("H", "A")):
        np.testing.assert_array_equal(labels[before], other[after])
        # OR of two independent fair-sign ANDs has positive mass 7/16.
        assert abs(labels[before].mean() - 7 / 16) < .03
    assert abs(np.mean(labels["A"] * labels["H"]) - (7 / 16) ** 2) < .03
    assert not np.array_equal(x, sample(design, 4, 3, 4096, .5)[0])


def test_fit_only_ranking_recovers_both_alternatives_and_breaks_ties_canonically():
    design = make_design(9)
    x, _ = sample(design, 32, 0, 2048, .5)
    candidates = perfect_matchings()
    assert set(rank_matchings(x, candidates)[:2]) == {design.matching("A"), design.matching("H")}
    assert rank_matchings(np.full((8, 8, 2), .5), reversed(candidates)) == candidates
