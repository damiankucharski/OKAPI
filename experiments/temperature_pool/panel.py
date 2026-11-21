"""Independent Gaussian provider banks, with generating quantities reserved for evaluation."""

import numpy as np
from fusionbench.scenarios.calibrated import mc_strength
from fusionbench.scenarios.calibration import MulticlassCommittee
from fusionbench.scenarios.committee import Saturated
from scipy.special import softmax

from okapi.node import LogitMeanNode, ValueNode
from okapi.tree import Tree

FAMILIES = ('multiclass_committee', 'saturated', 'heterogeneous')
SIZES = {'fit': 512, 'validation': 256, 'iid': 4096, 'shift': 4096}


def bindings(p):
    return {f'm{i}.npy': p[:, i, :] for i in range(p.shape[1])}


def tree_at_temperature(p, temperature):
    values = [ValueNode(None, rows, name) for name, rows in bindings(p).items()]
    values[0].add_child(LogitMeanNode(values[1:], temperature=float(temperature)))
    return Tree(values[0])


def oracle_weights(family, shifted):
    if family == 'multiclass_committee':
        k, classes, accuracy, rho = 8, 5, .6, .55 if shifted else 0.
    elif family == 'saturated':
        k, classes, accuracy, rho = 6, 4, .95, .35 if shifted else .95
    elif family == 'heterogeneous':
        weights = np.full(8, .10)
        weights[[2, 3] if shifted else [0, 1]] = 2.25
        return weights
    else:
        raise ValueError(f'Unknown family {family!r}')
    return np.full(k, mc_strength(accuracy, classes) / (1 + (k - 1) * rho))


def _generate(family, shifted, size, rng):
    weights = oracle_weights(family, shifted)
    if family == 'multiclass_committee':
        generator = Saturated(k=8, n_classes=5, accuracy=.60, corr=.55) if shifted else MulticlassCommittee()
    elif family == 'saturated':
        generator = Saturated(corr=.35) if shifted else Saturated()
    else:
        classes = 5
        y = rng.integers(0, classes, size)
        raw = rng.standard_normal((size, len(weights), classes))
        raw[np.arange(size), :, y] += weights
        p = softmax(raw, axis=2)
        return p, y, weights
    p, y, _ = generator._generate(size, rng)
    return p, y, weights


def data(family, bank):
    """Return [rows, providers, classes], observed labels and a held-out posterior per split."""
    index = FAMILIES.index(family)
    n_providers = len(oracle_weights(family, False))
    permutation = np.random.default_rng([bank, index, 1701]).permutation(n_providers)
    splits = {}
    for i, (split, size) in enumerate(SIZES.items()):
        rng = np.random.default_rng([bank, index, i, 1702])
        p, y, weights = _generate(family, split == 'shift', size, rng)
        p = np.ascontiguousarray(p[:, permutation])
        q = softmax(np.einsum('nmc,m->nc', np.log(p), weights[permutation]), axis=1)
        p.flags.writeable = y.flags.writeable = q.flags.writeable = False
        splits[split] = (p, y, q)
    return splits
