"""Compile binary min/max trees over affine sigmoid experts into raw ReLU circuits."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from okapi.node import MaxNode, MinNode


class BinaryLatticeCircuit(nn.Module):
    """One affine input layer and a directed acyclic sequence of residual ReLUs."""

    def __init__(self, weights, biases, gates, original_gates):
        super().__init__()
        self.leaves = nn.Linear(weights.shape[1], len(weights), dtype=torch.float64)
        with torch.no_grad():
            self.leaves.weight.copy_(torch.as_tensor(weights, dtype=torch.float64))
            self.leaves.bias.copy_(torch.as_tensor(biases, dtype=torch.float64))
        self.gates = tuple(gates)
        self.original_gates = original_gates

    def forward(self, x):
        """Positive-class logit. Apply sigmoid for a probability."""
        signals = list(self.leaves(x).unbind(dim=-1))
        for op, left, right in self.gates:
            a, b = signals[left], signals[right]
            difference = torch.relu(a - b)
            signals.append(a - difference if op == "min" else b + difference)
        return signals[-1]


def compile_binary_lattice(tree, affines: Mapping, *, signature_limit=12):
    """Compile a scalar, identity-postprocessed tree whose leaf is sigmoid(w·x+b).

    Each value-node id maps to a pair (weight_vector, bias), expressed in common
    raw feature coordinates. PF must be identity throughout the source tree.
    Other operators and multiclass probability normalization are unsupported.

    Up to signature_limit distinct affines, Boolean threshold signatures give
    exact semantic sharing: min/max commute with every threshold, so equality
    on all Boolean corners implies equality on a totally ordered score space.
    Above the limit only exact input and syntactic identities are shared.
    """
    if not isinstance(signature_limit, int) or not 0 <= signature_limit <= 16:
        raise ValueError("signature_limit must be an integer from 0 through 16")
    leaf_index, weights, biases, seen = {}, [], [], {}
    for leaf in tree.value_nodes:
        if leaf.id in leaf_index:
            continue
        w, b = affines[leaf.id]
        w = np.asarray(w, dtype=np.float64)
        if (
            w.ndim != 1
            or len(w) == 0
            or (weights and len(w) != len(weights[0]))
            or not np.isfinite(w).all()
            or not np.isscalar(b)
            or not np.isfinite(b)
        ):
            raise ValueError("expected finite affine weights on a common raw feature space")
        # Normalize signed zeros before hashing equal floating point vectors.
        key = ((w + 0.0).tobytes(), float(b))
        if key not in seen:
            seen[key] = len(weights)
            weights.append(w)
            biases.append(float(b))
        leaf_index[leaf.id] = seen[key]
    n = len(weights)
    nodes = [("leaf", i, i) for i in range(n)]
    semantic, by_signature = [], {}
    if n <= signature_limit:
        mask = (1 << (1 << n)) - 1
        for i in range(n):
            # Repeat 2**i zero bits and as many one bits across the table.
            run = 1 << i
            block = ((1 << run) - 1) << run
            truth = block * (mask // ((1 << (2 * run)) - 1))
            semantic.append(truth)
            by_signature[truth] = i
    by_inputs = {}
    original_gates = 0

    def combine(op, a, b):
        nonlocal original_gates
        original_gates += 1
        if a == b:
            return a
        ordered = (op, min(a, b), max(a, b))
        if ordered in by_inputs:
            return by_inputs[ordered]
        if semantic:
            key = semantic[a] & semantic[b] if op == "min" else semantic[a] | semantic[b]
            if key in by_signature:
                return by_signature[key]
            by_signature[key] = len(nodes)
            semantic.append(key)
        by_inputs[ordered] = len(nodes)
        nodes.append((op, a, b))
        return len(nodes) - 1

    def visit(value):
        signal = leaf_index[value.id]
        # OKAPI applies an ordered chain of operators to a ValueNode's current
        # prediction. Each child of the operator starts its own value program.
        for operator in value.children or ():
            if type(operator) not in (MinNode, MaxNode):
                raise ValueError(f"cannot compile {type(operator).__name__} as a min/max gate")
            op = "min" if type(operator) is MinNode else "max"
            for child in operator.children or ():
                signal = combine(op, signal, visit(child))
        return signal

    output = visit(tree.root)
    live, pending = set(), [output]
    while pending:
        current = pending.pop()
        if current in live:
            continue
        live.add(current)
        op, a, b = nodes[current]
        if op != "leaf":
            pending.extend((a, b))
    ordered = sorted(live)
    remap = {old: new for new, old in enumerate(ordered)}
    leaves = [i for i in ordered if i < n]
    gates = [(nodes[i][0], remap[nodes[i][1]], remap[nodes[i][2]]) for i in ordered if i >= n]
    return BinaryLatticeCircuit(np.stack([weights[i] for i in leaves]), np.array([biases[i] for i in leaves]), gates, original_gates)
