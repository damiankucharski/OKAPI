"""Research controls for a categorically selected, genuine native operator retype."""

from dataclasses import dataclass

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


class MatchedPointControls:
    """Mix in before Okapi, set point_policy and enable native point mutation.

    The three policies always execute the same genuine proposal first. The
    identity policy returns a private copy; refresh initializes the original
    operator's family from a copy of the native post-proposal RNG state.
    An optional point_event_callback(event) observes completed point events.
    """

    point_policy = "retype"
    point_event_callback = None

    def _apply_structural_mutation(self, mutation, tree):
        if mutation is not replace_operator_mutation:
            return super()._apply_structural_mutation(mutation, tree)
        if self.point_policy not in {"retype", "identity", "refresh"}:
            raise ValueError("point_policy must be retype, identity or refresh")
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
