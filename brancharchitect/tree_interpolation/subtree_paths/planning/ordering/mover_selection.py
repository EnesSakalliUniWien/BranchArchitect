"""Select the next mover subtree for a pivot transition."""

from __future__ import annotations

from typing import Mapping, Optional, Set, Tuple

from brancharchitect.elements.partition import Partition

from .path_group_manager import PathGroupManager
from .subtree_ordering import order_key_for_subtree
from ..claims import SplitClaimTracker


def select_next_mover_subtree(
    collapse_tracker: SplitClaimTracker,
    expand_tracker: SplitClaimTracker,
    path_group_manager: Optional[PathGroupManager],
    processed_subtrees: Set[Partition],
    subtree_order_key: Mapping[Partition, Tuple[int, ...]],
) -> Optional[Partition]:
    """Select the next unprocessed mover with transition work remaining."""
    unprocessed = remaining_mover_subtrees(
        collapse_tracker, expand_tracker, processed_subtrees
    )
    if not unprocessed:
        return None

    if any(collapse_tracker.get_shared_resources(subtree) for subtree in unprocessed):
        return _select_by_shared_collapse_priority(
            unprocessed, collapse_tracker, expand_tracker, subtree_order_key
        )

    if path_group_manager and path_group_manager.enabled:
        next_subtree = path_group_manager.get_next_subtree(processed_subtrees)
        if next_subtree is not None and next_subtree in unprocessed:
            return next_subtree

    return _select_by_smallest_expand_path(
        unprocessed, expand_tracker, subtree_order_key
    )


def remaining_mover_subtrees(
    collapse_tracker: SplitClaimTracker,
    expand_tracker: SplitClaimTracker,
    processed_subtrees: Set[Partition],
) -> set[Partition]:
    """Return subtrees that still own collapse or expand transition work."""
    remaining: Set[Partition] = set()
    remaining.update(collapse_tracker.get_all_owners())
    remaining.update(expand_tracker.get_all_owners())
    return remaining - processed_subtrees


def _select_by_shared_collapse_priority(
    unprocessed: Set[Partition],
    collapse_tracker: SplitClaimTracker,
    expand_tracker: SplitClaimTracker,
    subtree_order_key: Mapping[Partition, Tuple[int, ...]],
) -> Partition:
    """Prioritize movers that can remove shared source-only structure."""
    candidates = []
    for subtree in unprocessed:
        shared_collapse = collapse_tracker.get_shared_resources(subtree)
        tie_breaker = order_key_for_subtree(subtree, subtree_order_key)

        if shared_collapse:
            priority = (0, -len(shared_collapse), tie_breaker)
        elif expand_tracker.get_shared_resources(subtree):
            shared_expand = expand_tracker.get_shared_resources(subtree)
            priority = (2, -len(shared_expand), tie_breaker)
        else:
            priority = (1, 0, tie_breaker)

        candidates.append((priority, subtree))

    return min(candidates)[1]


def _select_by_smallest_expand_path(
    unprocessed: Set[Partition],
    expand_tracker: SplitClaimTracker,
    subtree_order_key: Mapping[Partition, Tuple[int, ...]],
) -> Partition:
    """Choose smaller expand paths first so shared expands are applied last."""
    candidates = []
    for subtree in unprocessed:
        shared_expand_count = len(expand_tracker.get_shared_resources(subtree))
        total_expand_count = len(expand_tracker.get_resources(subtree))
        tie_breaker = order_key_for_subtree(subtree, subtree_order_key)
        candidates.append(
            (shared_expand_count, total_expand_count, tie_breaker, subtree)
        )

    return min(candidates)[3]
