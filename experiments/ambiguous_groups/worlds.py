"""Two label laws sharing every observed sensor output on an alternating cycle."""

from dataclasses import dataclass
from math import fsum

import numpy as np
from scipy.special import expit, logit

Pair = tuple[int, int]
Matching = tuple[Pair, ...]
Paths = tuple[tuple[Pair, Pair], tuple[Pair, Pair]]


@dataclass(frozen=True)
class Design:
    """Hidden oracle metadata. Each world is an OR of two paths of two ANDed groups."""

    a_paths: Paths
    h_paths: Paths

    def paths(self, world: str) -> Paths:
        return {"A": self.a_paths, "H": self.h_paths}[world]

    def matching(self, world: str) -> Matching:
        """Canonical matching for comparison with the label-free proposal ranking."""
        return tuple(sorted(pair for path in self.paths(world) for pair in path))


def make_design(seed: int) -> Design:
    """Randomize the eight observed column IDs and both worlds' path assignments."""
    rng = np.random.default_rng(seed)
    cycle = rng.permutation(8).tolist()

    def paths(offset):
        pairs = [tuple(sorted((cycle[i], cycle[(i + 1) % 8]))) for i in range(offset, 8, 2)]
        a, b, c, d = [pairs[i] for i in rng.permutation(4)]
        return ((a, b), (c, d))

    return Design(paths(0), paths(1))


def sample(design: Design, seed: int, split_id: int, n: int, sigma: float):
    """Return one shared (rows, 8, 2) probability tensor and two label vectors.

    Each edge gets an independent symmetric sign per item. Assigning random
    draws in canonical edge order makes X bit-identical even if the two worlds'
    names are exchanged. A sensor sums its two incident signs and independent
    Gaussian noise. Only the oracle uses the path assignments to form labels.
    """
    if n < 1 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("n must be positive and sigma must be finite and nonnegative")
    rng = np.random.default_rng([seed, split_id])
    edges = sorted((*design.matching("A"), *design.matching("H")))
    signs = 2 * rng.integers(2, size=(n, len(edges))) - 1
    signal = np.zeros((n, 8), dtype=float)
    positive = {}
    for column, pair in enumerate(edges):
        i, j = pair
        signal[:, i] += signs[:, column]
        signal[:, j] += signs[:, column]
        positive[pair] = signs[:, column] > 0
    p = expit(1.2 * (signal + rng.normal(0, sigma, size=signal.shape)))
    labels = {}
    for world in ("A", "H"):
        (a, b), (c, d) = design.paths(world)
        labels[world] = ((positive[a] & positive[b]) | (positive[c] & positive[d])).astype(int)
    return np.stack((1 - p, p), axis=-1), labels


def perfect_matchings() -> list[Matching]:
    """All 105 perfect matchings on the generic observed IDs 0 through 7."""
    def enumerate_pairs(vertices):
        if not vertices:
            yield ()
            return
        first, *rest = vertices
        for i, partner in enumerate(rest):
            for tail in enumerate_pairs(rest[:i] + rest[i + 1:]):
                yield ((first, partner), *tail)

    return list(enumerate_pairs(range(8)))


def rank_matchings(x: np.ndarray, matchings) -> list[Matching]:
    """Rank by summed fit-logit Pearson correlation; break ties by matching.

    A constant provider contributes zero correlation with other providers.
    Exact saturation uses the closest interior floating-point probability.
    """
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (8, 2) or len(x) < 2:
        raise ValueError("Expected at least two rows of eight binary provider probabilities")
    p = np.asarray(x[:, :, 1], dtype=float)
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Expected finite positive-class probabilities in [0, 1]")
    logits = logit(np.clip(p, np.nextafter(0., 1.), np.nextafter(1., 0.)))
    centered = logits - logits[:1]
    centered -= centered.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    np.divide(centered, norm, out=centered, where=norm > 0)
    correlation = centered.T @ centered
    return sorted(matchings, key=lambda matching: (-fsum(correlation[pair] for pair in matching), matching))
