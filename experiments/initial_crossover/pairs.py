"""Uncapped native crossover probability on arbitrary-width binary MIN/MAX tables."""

from collections import Counter
from fractions import Fraction
from functools import cache
from itertools import combinations_with_replacement

from fusionbench.eval.donor_crossover import PreparedTree, Site
from fusionbench.eval.lattice_signature import lattice_signature
from okapi.node import MaxNode, MinNode, ValueNode


def prepare(tree, provider_masks):
    """Fill the source experiment's composable Site records using this full cube."""
    all_corners = (1 << (1 << len(provider_masks))) - 1
    arrivals = {}

    def evaluate(node, path, incoming, override, replacement, remember=False):
        if path == override:
            return replacement
        if remember:
            arrivals[path] = (node, incoming)
        if type(node) is ValueNode:
            value = provider_masks[node.id]
            for i, op in enumerate(node.children):
                value = evaluate(op, path + (i,), value, override, replacement, remember)
            return value
        if type(node) not in (MinNode, MaxNode):
            raise ValueError("Only native MIN/MAX and identity postprocessing are supported")
        value = incoming
        for i, operand in enumerate(node.children):
            child = evaluate(operand, path + (i,), 0, override, replacement, remember)
            value = value & child if type(node) is MinNode else value | child
        return value

    signature = evaluate(tree.root, (), 0, None, 0, True)
    assert signature == lattice_signature(tree, provider_masks)
    values, operators = [], []
    for path, (node, incoming) in arrivals.items():
        outer = tuple(evaluate(tree.root, (), 0, path, forced) for forced in (0, all_corners))
        if type(node) is ValueNode:
            fixed = evaluate(node, path, 0, None, 0)
            donor = (fixed, fixed)
            sites = values
        else:
            donor = tuple(evaluate(node, path, v, None, 0) for v in (0, all_corners))
            sites = operators
        site = Site(path, outer, incoming, donor)
        assert site.with_donor(site) == signature
        sites.append(site)
    return PreparedTree(tuple(provider_masks), signature, tuple(values), tuple(operators), frozenset(v.donor[0] for v in values))


@cache
def child_tables(left, right):
    groups = [(left.value_sites, right.value_sites)]
    if left.operator_sites and right.operator_sites:
        groups.append((left.operator_sites, right.operator_sites))
    outcomes = Counter()
    for aa, bb in groups:
        count = Counter((a.with_donor(b), b.with_donor(a)) for a in aa for b in bb)
        denominator = len(groups) * len(aa) * len(bb)
        for children, n in count.items():
            outcomes[children] += Fraction(n, denominator)
    assert sum(outcomes.values()) == 1
    return dict(outcomes)


def productive_by_target(parents):
    """For every target, the exact per-pair probability from target-free parents."""
    targets = {}
    for i, j in combinations_with_replacement(range(len(parents)), 2):
        left, right = parents[i], parents[j]
        forbidden = {left.signature, right.signature}
        probs = Counter()
        for children, mass in child_tables(left, right).items():
            for target in set(children) - forbidden:
                probs[target] += mass
        for target, probability in probs.items():
            targets.setdefault(target, []).append((i, j, probability))
    return targets
