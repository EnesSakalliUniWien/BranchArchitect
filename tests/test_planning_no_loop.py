import pytest

from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.subtree_paths.planning.state import (
    InterpolationState,
)


def P(indices, enc):
    return Partition(tuple(indices), enc)


def test_interpolation_state_progress_no_loop():
    """
    Ensure InterpolationState loop makes progress and terminates.

    Sets up a synthetic scenario with two subtrees and a mix of
    shared and unique collapse/expand splits, then runs the
    selection + mark-as-processed loop. Asserts termination within
    a reasonable iteration bound (no infinite loop).
    """
    enc = {"A": 0, "B": 1, "C": 2, "D": 3}

    # Subtrees (keys)
    S1 = P((0, 1), enc)  # AB
    S2 = P((2, 3), enc)  # CD

    # Splits
    X = P((0, 2), enc)  # shared collapse
    Y = P((1, 3), enc)  # shared expand
    U1 = P((0, 3), enc)  # unique collapse for S1
    U2 = P((1, 2), enc)  # unique expand for S2

    collapse_by_subtree = {
        S1: PartitionSet({X, U1}, encoding=enc),
        S2: PartitionSet({X}, encoding=enc),
    }
    expand_by_subtree = {
        S1: PartitionSet({Y}, encoding=enc),
        S2: PartitionSet({Y, U2}, encoding=enc),
    }

    all_collapse = PartitionSet({X, U1}, encoding=enc)
    all_expand = PartitionSet({Y, U2}, encoding=enc)

    active_edge = P((0, 1, 2, 3), enc)

    state = InterpolationState(
        all_collapse_splits=all_collapse,
        all_expand_splits=all_expand,
        collapse_splits_by_subtree=collapse_by_subtree,
        expand_splits_by_subtree=expand_by_subtree,
        active_changing_edge=active_edge,
    )

    # Iterate selection + processing ensuring termination
    max_iters = 20
    iters = 0
    while state.has_remaining_work():
        iters += 1
        assert iters <= max_iters, "Planner loop exceeded iteration bound (possible infinite loop)"

        subtree = state.get_next_subtree()
        assert subtree is not None, "No subtree selected despite remaining work"

        # Build simple paths from available/unique sets
        sc = state.get_available_shared_collapse_splits(subtree)
        uc = state.get_unique_collapse_splits(subtree)
        sl = state.get_expand_splits_for_last_user(subtree)
        ue = state.get_unique_expand_splits(subtree)

        # Compatible expand splits with current collapsed set
        comp = state.find_compatible_expand_splits_for_subtree(subtree, sc | uc)

        expand_path = PartitionSet(encoding=enc).union(sl, ue, comp)
        incompatible = state.find_all_incompatible_splits_for_expand(
            expand_path, state.all_collapsible_splits
        )
        collapse_path = PartitionSet(encoding=enc).union(sc, uc, incompatible)

        # On last subtree, only add extras that actually belong to this subtree
        if state.is_last_subtree(subtree):
            extras = state.available_compatible_splits - state.used_compatible_splits
            expand_path = expand_path | (
                extras
                & state.expand_splits_by_subtree.get(
                    subtree, PartitionSet(encoding=state.encoding)
                )
            )

        # Apply
        state.mark_splits_as_processed(subtree, collapse_path, expand_path)

    # Finished
    assert not state.has_remaining_work()

