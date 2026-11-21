"""Research controls for a categorically selected, genuine native operator retype."""

from dataclasses import dataclass
from hashlib import sha256
from struct import pack

import numpy as np

from okapi.interventions import copy_for_variation, operator_paths
from okapi.mutation import replace_operator_mutation, tree_with_reinitialized_operator
from okapi.random_state import current_rng, use_rng


@dataclass(frozen=True)
class PointEvent:
    path: tuple[int, ...]
    original_family: str
    proposed_family: str
    returned_family: str
    changed: bool


def _refresh_stream():
    """Fork the post-proposal state without reusing the next native draws."""
    algorithm, keys, position, has_gaussian, cached_gaussian = current_rng().get_state()
    digest = sha256(b"okapi/point-ablation/refresh/v1\x00")
    digest.update(algorithm.encode("ascii"))
    digest.update(keys.astype("<u4", copy=False).tobytes())
    digest.update(pack("<IId", position, has_gaussian, cached_gaussian))
    return np.random.RandomState(np.frombuffer(digest.digest(), dtype="<u4"))


class MatchedPointControls:
    """Mix in before Okapi, set point_policy and enable native point mutation.

    All matched policies execute the same genuine proposal first. Identity
    returns a private copy. Refresh redraws the original family's parameters
    with either a direct copy of the post-proposal state (refresh) or a
    domain-separated hash of that state (refresh_independent).
    An optional point_event_callback(event) observes completed point events.
    """

    point_policy = "retype"
    point_event_callback = None

    def _apply_structural_mutation(self, mutation, tree):
        if mutation is not replace_operator_mutation:
            return super()._apply_structural_mutation(mutation, tree)
        if self.point_policy not in {"retype", "identity", "refresh", "refresh_independent"}:
            raise ValueError("point_policy must be retype, identity, refresh or refresh_independent")
        proposal = super()._apply_structural_mutation(mutation, tree)
        # Production retyping retains native node-list order and changes exactly one family.
        differences = [
            (i, orig, alt) for i, (orig, alt) in enumerate(zip(tree.op_nodes, proposal.op_nodes, strict=True))
            if type(orig) is not type(alt)
        ]
        if not differences:
            return proposal
        (index, old, new), = differences
        path = next(path for path, op in operator_paths(tree) if op is old)
        if self.point_policy == "retype":
            result = proposal
        elif self.point_policy == "identity":
            result = copy_for_variation(tree)
        else:
            if self.point_policy == "refresh_independent":
                shadow = _refresh_stream()
            else:
                shadow = np.random.RandomState(0)
                shadow.set_state(current_rng().get_state())
            with use_rng(shadow):
                result = tree_with_reinitialized_operator(tree, path)
        if self.point_event_callback is not None:
            self.point_event_callback(PointEvent(
                path, type(old).__name__, type(new).__name__, type(result.op_nodes[index]).__name__,
                tree.structural_key() != result.structural_key(),
            ))
        return result
