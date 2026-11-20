"""Fit a four-leaf OKAPI topology on expert scores, then train its logits from task labels.

Raw evidence is rotated and mixed with four irrelevant coordinates. The trainable
leaves are dense affine projections; the network receives no sensor groups, base
weights, concept labels, or expert predictions. A fixed split is regenerated from
its seed. The generator extends FusionBench TwoPathways with raw inputs.
"""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch
from fusionbench.scenarios import calibrated as cal
from fusionbench.scenarios.gates import TwoPathways
from torch import nn
from torch.nn import functional as F

from okapi.globals import set_backend
from okapi.node import MaxNode, MeanNode, MinNode, ValueNode
from okapi.tree import Tree

OPS = {"min": MinNode, "max": MaxNode, "mean": MeanNode}
STRENGTH = cal.acc_to_strength(.90)


def draw(n, rng, rotation):
    bits = rng.integers(0, 2, size=(n, 4))
    evidence = (2 * bits - 1) * STRENGTH + rng.normal(size=bits.shape)
    y = (bits[:, 0] & bits[:, 1]) | (bits[:, 2] & bits[:, 3])
    nuisance = rng.normal(size=bits.shape)
    raw = np.column_stack((evidence, nuisance)) @ rotation
    scores = cal.sigmoid(2 * STRENGTH * evidence)
    return torch.tensor(raw, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scores


def tree_for(pairing, ops, scores):
    a, b, c, d = pairing
    inner_left, inner_right, outer = [OPS[name] for name in ops]
    left = ValueNode([inner_left([ValueNode(None, scores[:, b], b)])], scores[:, a], a)
    right = ValueNode([inner_right([ValueNode(None, scores[:, d], d)])], scores[:, c], c)
    left.add_child(outer([right]))
    return Tree(left)


def fit_topology(scores, y):
    candidates = []
    for pairing in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
        for ops in itertools.product(OPS, repeat=3):
            prediction = tree_for(pairing, ops, scores).predict()
            candidates.append((np.mean((prediction - y) ** 2), ops, pairing))
    return min(candidates)[1:]


class Circuit(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.leaves = nn.Linear(8, 4)
        self.ops = ops

    @staticmethod
    def combine(a, b, op):
        if op == "min":
            return torch.minimum(a, b)
        if op == "max":
            return torch.maximum(a, b)
        # Average probabilities, expressed as a log-odds difference for stability.
        return torch.logaddexp(F.logsigmoid(a), F.logsigmoid(b)) - torch.logaddexp(F.logsigmoid(-a), F.logsigmoid(-b))

    def forward(self, x):
        a, b, c, d = self.leaves(x).unbind(dim=1)
        left = self.combine(a, b, self.ops[0])
        right = self.combine(c, d, self.ops[1])
        return self.combine(left, right, self.ops[2])


def train(model, x, y, schedule):
    opt = torch.optim.Adam(model.parameters(), lr=.025)
    for idx in schedule:
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x[idx]).reshape(-1), y[idx])
        loss.backward()
        opt.step()


def circuit_alignment(model, rotation):
    # Rotation rows are the true concept directions in raw coordinates. They
    # enter this post-training diagnostic only. Assignment accounts for global
    # leaf permutations, with a signed cosine rewarding the correct polarity.
    weights = model.leaves.weight.detach().numpy()
    normalized = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-10)
    cosine = normalized @ rotation[:4].T
    return float(max(np.mean(cosine[range(4), perm]) for perm in itertools.permutations(range(4))))


def run_seed(seed, sizes, steps):
    rng = np.random.default_rng(10_000 + seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    fit_x, fit_y, fit_p = draw(2048, rng, rotation)
    ops, pairing = fit_topology(fit_p, fit_y.numpy())
    x, y, _ = draw(max(sizes), rng, rotation)
    test, labels, p = draw(8192, rng, rotation)
    y_test = labels.numpy()
    teacher = tree_for(pairing, ops, p).predict()
    oracle = TwoPathways(rep=1).bayes_oracle(np.stack([1-p, p], axis=-1), {})[:, 1]
    dual = tuple({"max": "min", "min": "max", "mean": "mean"}[name] for name in ops)
    rows = []
    for size in sizes:
        generator = torch.Generator().manual_seed(seed + 3000)
        schedule = torch.randint(size, (steps, 128), generator=generator)
        choices = {
            "recovered": lambda: Circuit(ops),
            "dual": lambda: Circuit(dual),
            "mlp4": lambda: nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 1)),
            "mlp32": lambda: nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1)),
            "mlp32_pooled": lambda: nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1)),
            "relu32_pooled": lambda: nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1)),
        }
        predicted = {"expert_tree": (teacher, 0, None), "bayes": (oracle, 0, None)}
        for name, builder in choices.items():
            torch.manual_seed(seed + 223)
            model = builder()
            if name.endswith("_pooled"):
                # Give the baseline the raw data and task labels used for
                # topology selection too. The optimization budget stays fixed.
                train_x = torch.cat((x[:size], fit_x))
                train_y = torch.cat((y[:size], fit_y))
                batch_rng = torch.Generator().manual_seed(seed + 3000)
                pooled_schedule = torch.randint(len(train_y), (steps, 128), generator=batch_rng)
                train(model, train_x, train_y, pooled_schedule)
            else:
                train(model, x[:size], y[:size], schedule)
            with torch.no_grad():
                prob = torch.sigmoid(model(test)).reshape(-1).numpy()
            alignment = circuit_alignment(model, rotation) if isinstance(model, Circuit) else None
            predicted[name] = (prob, sum(t.numel() for t in model.parameters()), alignment)
        for name, (prob, params, alignment) in predicted.items():
            correct = (prob >= .5) == y_test
            rows.append(dict(seed=seed, n_fit=size, n_test=len(y_test), model=name,
                             params=params, steps=steps, topology_n_fit=len(fit_y), alignment=alignment, ops=",".join(ops), pairing=",".join(map(str, pairing)),
                             accuracy=float(correct.mean()), brier=float(np.mean((prob-y_test)**2)),
                             teacher_agreement=float(((prob >= .5) == (teacher >= .5)).mean())))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 2048])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seeds, *args.sizes, args.steps) < 1:
        parser.error("counts must be positive")
    torch.set_num_threads(1)
    set_backend("numpy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        for seed in range(args.start, args.start + args.seeds):
            rows = run_seed(seed, args.sizes, args.steps)
            if seed == args.start:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
            stream.flush()
