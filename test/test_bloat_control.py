"""Bloat / depth control (Track C1).

Covers the depth/size primitives and the hard ``max_depth`` / ``max_nodes`` caps that
are enforced at generation time. The caps default to ``None`` (no limit), which must
reproduce the historical uncapped behaviour exactly; when set, no individual deeper /
larger than the cap may ever enter the population, and the search must still terminate
even under a punishing cap (bounded-retry crossover + parent-copy fallback).
"""

import time

import numpy as np
import pytest

from okapi.crossover import _legal_crossover_pair, crossover
from okapi.mutation import append_new_node_mutation, get_allowed_mutations
from okapi.node import MeanNode, ValueNode
from okapi.okapi import Okapi
from okapi.pareto import maximize
from okapi.tree import Tree


def _identity(x):
    return x


def _vnode(model_id="m"):
    return ValueNode(None, np.array([[0.6, 0.4], [0.3, 0.7]]), model_id)


def _fusion_tree(model_ids=("a", "b", "c")):
    """A single-operator fusion: root value -> op -> children values (depth 3)."""
    root = _vnode(model_ids[0])
    op = MeanNode(None)
    for mid in model_ids[1:]:
        op.add_child(_vnode(mid))
    root.add_child(op)
    return Tree.create_tree_from_root(root)


def _node_depth(node):
    return 1 if not node.children else 1 + max(_node_depth(c) for c in node.children)


def _deep_chain_tree(depth_levels=5):
    """A value/operator alternating chain with the requested (odd) depth in levels."""
    assert depth_levels % 2 == 1 and depth_levels >= 1
    node = _vnode("leaf")  # depth 1
    while _node_depth(node) < depth_levels:
        op = MeanNode(None)
        op.add_child(node)
        parent_value = _vnode("v")
        parent_value.add_child(op)
        node = parent_value  # adds two levels (value -> op -> ...)
    return Tree.create_tree_from_root(node)


# --------------------------------------------------------------------------- #
# Tree.depth (pure)
# --------------------------------------------------------------------------- #


def test_depth_single_node_is_one():
    assert Tree.create_tree_from_root(_vnode()).depth == 1


def test_depth_single_fusion_is_three():
    tree = _fusion_tree()
    assert tree.depth == 3
    assert tree.nodes_count == 4


def test_depth_matches_scanner_convention_on_deep_chain():
    tree = _deep_chain_tree(5)
    assert tree.depth == 5


# --------------------------------------------------------------------------- #
# get_allowed_mutations gate (append dropped only at the node cap)
# --------------------------------------------------------------------------- #


def test_append_allowed_when_under_node_cap():
    tree = _fusion_tree()  # 4 nodes
    assert append_new_node_mutation in get_allowed_mutations(tree, max_nodes=None)
    assert append_new_node_mutation in get_allowed_mutations(tree, max_nodes=5)


def test_append_dropped_at_or_above_node_cap():
    tree = _fusion_tree()  # 4 nodes
    assert append_new_node_mutation not in get_allowed_mutations(tree, max_nodes=4)
    assert append_new_node_mutation not in get_allowed_mutations(tree, max_nodes=3)


def test_depth_does_not_gate_append_so_width_growth_survives():
    # A depth-3 tree must still be allowed to append (grow wider) — depth is enforced
    # post-hoc, not by removing append, so wide-shallow trees can form at max_depth=3.
    tree = _fusion_tree()
    assert append_new_node_mutation in get_allowed_mutations(tree, max_nodes=None)


# --------------------------------------------------------------------------- #
# Engine integration
# --------------------------------------------------------------------------- #


def _acc(prediction, gt):
    p = np.asarray(prediction)
    y = np.asarray(gt).ravel().astype(int)
    yhat = (p[:, 0] > 0.5).astype(int) if p.shape[1] == 1 else p.argmax(axis=1)
    return float((yhat == y).mean())


def _make_engine(tmp_path, *, n_models=10, n=200, max_depth=None, max_nodes=None,
                 minimize_node_count=True, seed=0, pop=10):
    rng = np.random.default_rng(seed)
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir(parents=True)
    for k in range(n_models):
        np.save(fit_dir / f"m{k}.npy", rng.random((n, 1)).astype(np.float64))
    gt_path = tmp_path / "gt.npy"
    np.save(gt_path, (rng.random(n) > 0.5).astype(np.int64))
    return Okapi(
        preds_source=fit_dir,
        gt_path=gt_path,
        population_size=pop,
        population_multiplier=3,
        tournament_size=4,
        minimize_node_count=minimize_node_count,
        objective_functions=(_acc,),
        objectives=(maximize,),
        backend="numpy",
        seed=seed,
        postprocessing_function=_identity,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )


def test_engine_caps_default_to_none(tmp_path):
    ok = _make_engine(tmp_path)
    assert ok.max_depth is None and ok.max_nodes is None


def test_within_caps_rejects_oversized_trees(tmp_path):
    # Deterministic proof the cap mechanism is sound, independent of evolution dynamics.
    ok = _make_engine(tmp_path, max_depth=3, max_nodes=6)
    shallow = _fusion_tree()                 # depth 3, 4 nodes
    deep = _deep_chain_tree(5)               # depth 5
    assert ok._within_caps(shallow)
    assert not ok._within_caps(deep)         # exceeds max_depth

    ok_nodes = _make_engine(tmp_path / "b", max_nodes=3)
    assert not ok_nodes._within_caps(shallow)  # 4 nodes > 3


def test_max_depth_cap_never_exceeded(tmp_path):
    ok = _make_engine(tmp_path, max_depth=3, minimize_node_count=False, seed=1)
    ok.train(40)
    assert all(t.depth <= 3 for t in ok.population)
    assert all(t.depth <= 3 for t in ok.pareto_trees)


def test_max_nodes_cap_never_exceeded(tmp_path):
    ok = _make_engine(tmp_path, max_nodes=7, minimize_node_count=False, seed=2)
    ok.train(40)
    assert all(t.nodes_count <= 7 for t in ok.population)
    assert all(t.nodes_count <= 7 for t in ok.pareto_trees)


def test_uncapped_search_grows_beyond_depth_three(tmp_path):
    # Control: with no cap and no parsimony pressure the search builds trees deeper
    # than 3 levels, proving the caps above are actually binding (not vacuous).
    ok = _make_engine(tmp_path, max_depth=None, minimize_node_count=False, seed=3)
    ok.train(80)
    assert max(t.depth for t in ok.population) > 3


def test_tight_cap_terminates(tmp_path):
    # Punishing cap: bounded-retry crossover + parent-copy fallback must keep the
    # evolution loop terminating, with every individual within both caps.
    start = time.time()
    ok = _make_engine(tmp_path, max_depth=3, max_nodes=4, minimize_node_count=False, seed=5)
    ok.train(30)
    assert time.time() - start < 180
    assert ok.population, "population must remain non-empty under a tight cap"
    assert all(t.depth <= 3 and t.nodes_count <= 4 for t in ok.population)


# --------------------------------------------------------------------------- #
# C1b: deterministic legal-target selection (no retry, no revert)
# --------------------------------------------------------------------------- #


def test_legal_append_targets_depth_budget():
    tree = _fusion_tree()  # depth 3: root(V,1) -> op(2) -> [V,V](3); 4 nodes
    depth_of = tree._depth_map()
    root = tree.root
    op = tree.nodes["op_nodes"][0]
    leaves = [v for v in tree.nodes["value_nodes"] if depth_of[v] == 3]

    # max_depth=3: only shallow nodes can host an append without deepening past 3.
    t3 = tree.legal_append_targets(max_depth=3)
    assert root in t3 and op in t3
    assert all(leaf not in t3 for leaf in leaves)
    # max_depth=5: the depth-3 leaves become legal (value target adds two levels -> 5).
    t5 = tree.legal_append_targets(max_depth=5)
    assert all(leaf in t5 for leaf in leaves)
    # uncapped: every node is a legal target.
    assert len(tree.legal_append_targets()) == tree.nodes_count


def test_legal_append_targets_node_budget():
    tree = _fusion_tree()  # 4 nodes
    op = tree.nodes["op_nodes"][0]
    # Room for exactly one node: only an op target (adds 1) qualifies, not a value (adds 2).
    t5 = tree.legal_append_targets(max_nodes=5)
    assert op in t5 and tree.root not in t5
    # No room at all.
    assert tree.legal_append_targets(max_nodes=4) == []


def test_capped_append_grows_width_not_depth():
    # The capability the deterministic selector unlocks over the bugged shallow-only
    # search: at max_depth=3 append still fires, building WIDER depth-3 trees, never deeper.
    np.random.seed(0)
    models = [np.full((2, 1), 0.5) for _ in range(5)]
    ids = [f"m{i}" for i in range(5)]
    tree = _fusion_tree()
    for _ in range(6):
        tree = append_new_node_mutation(tree, models, ids, allowed_ops=(MeanNode,), max_depth=3)
    assert tree.depth == 3          # never deepened past the cap
    assert tree.nodes_count > 4     # but grew wider than the minimal single fusion


def test_capped_append_no_legal_target_is_noop():
    # A 4-node fusion at max_nodes=4 has no legal attachment point -> unchanged copy.
    np.random.seed(0)
    models = [np.full((2, 1), 0.5) for _ in range(3)]
    ids = ["m0", "m1", "m2"]
    tree = _fusion_tree()
    out = append_new_node_mutation(tree, models, ids, allowed_ops=(MeanNode,), max_nodes=4)
    assert out.nodes_count == tree.nodes_count


def test_crossover_offspring_respect_caps():
    np.random.seed(1)
    t1 = _fusion_tree(("a", "b", "c"))  # both within caps (depth 3)
    t2 = _fusion_tree(("d", "e", "f"))
    for _ in range(100):
        o1, o2 = crossover(t1, t2, max_depth=3, max_nodes=8)
        assert o1.depth <= 3 and o2.depth <= 3
        assert o1.nodes_count <= 8 and o2.nodes_count <= 8


def test_legal_crossover_value_pair_always_exists():
    # value root x root swaps whole within-caps trees -> always legal, even at the
    # tightest binding cap, so the engine path never needs a retry or parent-copy fallback.
    t1 = _fusion_tree(("a", "b", "c"))
    t2 = _fusion_tree(("d", "e", "f"))
    pair = _legal_crossover_pair(
        t1, t2, "value_nodes",
        max_depth=max(t1.depth, t2.depth),
        max_nodes=max(t1.nodes_count, t2.nodes_count),
    )
    assert pair is not None


def test_legal_crossover_pair_excludes_deep_into_shallow_swaps():
    # Grafting t2's whole height-3 tree onto a depth-3 leaf of t1 would make depth 5;
    # the filter must never offer that pair at max_depth=3 (deterministic exclusion).
    t1 = _fusion_tree(("a", "b", "c"))
    t2 = _fusion_tree(("d", "e", "f"))
    depth1 = t1._depth_map()
    deep_leaf = next(v for v in t1.nodes["value_nodes"] if depth1[v] == 3)
    root2 = t2.root
    for _ in range(100):
        n1, n2 = _legal_crossover_pair(t1, t2, "value_nodes", max_depth=3, max_nodes=None)
        assert not (n1 is deep_leaf and n2 is root2)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
