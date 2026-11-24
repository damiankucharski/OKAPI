"""Select complete native GP programs as additional log-evidence features."""

import numpy as np


def _coordinates(p):
    bounded = np.clip(p, 1e-6, 1.0 - 1e-6)
    if p.shape[1] == 1:
        return np.log(bounded) - np.log1p(-bounded)
    logs = np.log(bounded)
    return logs - logs.mean(axis=1, keepdims=True)


def _prediction(coordinates):
    if coordinates.shape[1] == 1:
        return 1.0 / (1.0 + np.exp(-coordinates))
    mass = np.exp(coordinates - coordinates.max(axis=1, keepdims=True))
    return mass / mass.sum(axis=1, keepdims=True)


def select_pattern_features(archive, provider_predictions, flat_prediction, labels, *, objective):
    """Greedily screen up to three complete GP trees against a fitted prediction.

    Supply ordered (tree, prediction) pairs from the discovery sample, raw
    provider predictions on those rows, and a fitted flat model's predictions.
    Predictions are NumPy arrays shaped (N, 1) for binary positive probabilities
    or (N, C) for multiclass probabilities. The higher-is-better scalar callback
    receives this same prediction format and the original labels.

    Singletons and programs equal to a raw provider or earlier program within
    1e-12 on every discovery probability are skipped. For each remaining
    program try adding its log-odds (or centered log probabilities) with each
    fixed signed weight. Accept the largest strict score increase, preferring
    the first archive entry and then the first trial weight on exact ties.
    Returns selected trees in addition order plus search counts and scores.
    """
    flat = np.asarray(flat_prediction, dtype=float)
    if flat.ndim != 2 or len(flat) != len(labels) or min(flat.shape) < 1:
        raise ValueError("Supply row-aligned native probability matrices and labels")

    def checked(p):
        p = np.asarray(p, dtype=float)
        if p.shape != flat.shape or not np.all((p >= 0.0) & (p <= 1.0)):
            raise ValueError("All predictions must have the same shape with finite probabilities in [0, 1]")
        if p.shape[1] > 1 and not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-6):
            raise ValueError("Multiclass probabilities must sum to one on each row")
        return p

    archive = tuple(archive)
    seen = [checked(p) for p in provider_predictions]
    if not seen:
        raise ValueError("Supply at least one raw provider")
    probes = np.linspace(0, flat.size - 1, num=min(16, flat.size), dtype=int)
    signatures = np.empty((len(seen) + len(archive), len(probes)))
    for i, p in enumerate(seen):
        signatures[i] = p.flat[probes]
    candidates = {}
    non_singletons = 0
    for i, (tree, output) in enumerate(archive):
        if tree.nodes_count == 1:
            continue
        non_singletons += 1
        p = checked(output)
        signature = p.flat[probes]
        close = np.all(np.abs(signatures[:len(seen)] - signature) <= 1e-12, axis=1)
        if any(np.all(np.abs(seen[j] - p) <= 1e-12) for j in np.flatnonzero(close)):
            continue
        signatures[len(seen)] = signature
        seen.append(p)
        candidates[i] = _coordinates(p)

    calls = 0

    def score(p):
        nonlocal calls
        calls += 1
        value = np.asarray(objective(p, labels))
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError("objective must return a finite scalar")
        return float(value)

    current = _coordinates(checked(flat))
    initial_score = best_score = score(flat)
    eligible = len(candidates)
    selected = []
    for _ in range(min(3, eligible)):
        winner = None
        for i, feature in candidates.items():
            for weight in (0.25, -0.25, 1.0, -1.0):
                trial = current + weight * feature
                value = score(_prediction(trial))
                if value > best_score:
                    winner = i, weight, trial
                    best_score = value
        if winner is None:
            break
        i, weight, current = winner
        del candidates[i]
        selected.append({"archive_index": i, "weight": weight, "score": best_score})
    return [archive[item["archive_index"]][0] for item in selected], {
        "archive_programs": len(archive),
        "non_singletons": non_singletons,
        "unique_novel_features": eligible,
        "initial_score": initial_score,
        "selected": selected,
        "evaluations": calls,
    }
