"""Check the one-update native ceiling of a two-path conditional law on a binary cube."""

from fractions import Fraction
from itertools import combinations, product

import numpy as np

from okapi.globals import set_backend, set_postprocessing_function
from okapi.node import MaxNode, MeanNode, MinNode, ValueNode, WeightedMeanNode
from okapi.tree import Tree


def native_programs(x):
    def terminal(i, operations=()):
        return ValueNode(operations, x[:, i], i)

    # V0 is updated first by MIN(V0,V1), then by MAX(current,MIN(V2,V3)).
    right = terminal(2, [MinNode([terminal(3)])])
    tree = Tree(terminal(0, [MinNode([terminal(1)]), MaxNode([right])]))
    others = [terminal(i) for i in (1, 2, 3)]
    mean = Tree(terminal(0, [MeanNode(others)]))
    weighted = Tree(terminal(0, [WeightedMeanNode([terminal(i) for i in (1, 2, 3)], [.25] * 4)]))
    return tree, mean, weighted


def verify():
    set_backend("numpy")
    set_postprocessing_function(lambda x: x)
    bits = np.array(list(product((0, 1), repeat=6)))
    target = np.maximum(np.minimum(bits[:, 0], bits[:, 1]), np.minimum(bits[:, 2], bits[:, 3]))
    assert Fraction(int(target.sum()), len(bits)) == Fraction(7, 16)
    covariances = [
        Fraction(int((x * target).sum()), 64) - Fraction(1, 2) * Fraction(7, 16)
        for x in bits.T
    ]
    assert covariances == [Fraction(3, 32)] * 4 + [0, 0]

    # The two opposite positive corners and two opposite negative corners
    # have identical summed feature vectors. Affine scores must share their sum.
    positive = bits[[0b110000, 0b001100]].sum(axis=0)
    for negative in ([0b101000, 0b010100], [0b100100, 0b011000]):
        np.testing.assert_array_equal(positive, bits[negative].sum(axis=0))
    low, high = Fraction(1, 64), Fraction(63, 64)
    pooled = (high + 2 * low) / 3
    relaxed = (high - pooled) ** 2 + 2 * (low - pooled) ** 2
    assert relaxed / (2 * 16) == Fraction(961, 49152)
    score = bits[:, :4] @ [2, 2, 1, 1] - 3.5
    assert Fraction(int(np.sum((score >= 0) != target)), 64) == Fraction(1, 16)

    scores = {}
    for name, aggregate in (("min", np.min), ("max", np.max)):
        errors = []
        for size in range(1, 7):
            for group in combinations(range(6), size):
                p = aggregate(bits[:, group], axis=1)
                errors.append(Fraction(int(np.sum((p - target) ** 2)), len(bits)))
        scores[name] = min(errors)
    assert scores == {"min": Fraction(3, 16), "max": Fraction(5, 16)}

    for low, high in [(0, 1), (Fraction(1, 64), Fraction(63, 64))]:
        span = high - low
        x = float(low) + float(span) * bits
        conditional = float(low) + float(span) * target
        tree, mean, weighted = native_programs(x)
        assert len(tree.op_nodes) == 3
        np.testing.assert_array_equal(tree.predict(), conditional)
        # Permutation symmetry of the convex quadratic gives the lower bound
        # derived in the note. Equal weights on the four signals attain it.
        optimum = Fraction(1, 8) * span**2
        for model in (mean, weighted):
            assert len(model.op_nodes) == 1
            assert Fraction(float(np.mean((model.predict() - conditional) ** 2))) == optimum
        print(f"endpoints {low}, {high}: one update {optimum}, three updates 0")
    print(f"half-corner mixture lower bound for one update: {optimum / 2}")


if __name__ == "__main__":
    verify()
