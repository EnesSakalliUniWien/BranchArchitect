"""
State management for tree interpolation process.

This module provides the InterpolationState class which manages the bookkeeping
of shared splits, unique splits, and which subtrees can process which splits
during the tree interpolation sequence.

Sections:
    1. Initialization and State Variables
    2. Shared/Unique Split Queries
    3. Split Deletion and Processing
    4. Subtree Selection and Prioritization
    5. Compatibility/Incompatibility Logic
    6. Remaining Work Queries
"""

import logging
from typing import Dict, Optional
from collections import Counter
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from ..analysis.split_analysis import find_incompatible_splits


# ============================================================================
# 1. Initialization and State Variables
# ============================================================================


class InterpolationState:
    """
    Manages state during tree interpolation to ensure shared splits are only used once.

    Tracks:
        - Which splits have been processed (and by which subtree)
        - Available splits for each subtree at any given time
        - Proper cleanup when subtrees complete their processing
    """

    def __init__(
        self,
        all_collapse_splits: PartitionSet[Partition],
        all_expand_splits: PartitionSet[Partition],
        collapse_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
        expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
        active_changing_edge: Partition,
    ):
        """
        Initialize the interpolation state.
        Args:
            all_collapse_splits: All splits that can be collapsed
            all_expand_splits: All splits that can be expanded
            collapse_splits_by_subtree: Initial collapse splits for each subtree
            expand_splits_by_subtree: Initial expand splits for each subtree
            active_changing_edge: The edge being processed
        """
        self.encoding = active_changing_edge.encoding
        self.all_collapsible_splits: PartitionSet[Partition] = all_collapse_splits

        all_unique_expand_splits: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )
        for subtree_expand_splits in expand_splits_by_subtree.values():
            all_unique_expand_splits |= subtree_expand_splits

        self.available_compatible_splits = all_expand_splits - all_unique_expand_splits
        self.collapse_splits_by_subtree = collapse_splits_by_subtree
        self.expand_splits_by_subtree = expand_splits_by_subtree
        self.processed_subtrees: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )
        self.used_compatible_splits: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )

    # ============================================================================
    # 2. Shared/Unique Split Queries
    # ============================================================================

    def _get_shared_splits(
        self, paths_by_subtree: Dict[Partition, PartitionSet[Partition]]
    ) -> Counter[Partition]:
        """Dynamically counts occurrences of each split across all subtrees."""
        return Counter(split for path in paths_by_subtree.values() for split in path)

    def get_available_shared_collapse_splits(
        self, subtree: Partition
    ) -> PartitionSet[Partition]:
        """Get shared collapse splits still available for this subtree."""
        shared_counts = self._get_shared_splits(self.collapse_splits_by_subtree)
        subtree_splits = self.collapse_splits_by_subtree.get(
            subtree, PartitionSet(encoding=self.encoding)
        )
        return PartitionSet(
            {s for s in subtree_splits if shared_counts.get(s, 0) > 1},
            encoding=self.encoding,
        )

    def get_available_shared_expand_splits(
        self, subtree: Partition
    ) -> PartitionSet[Partition]:
        """Get shared expand splits still available for this subtree."""
        shared_counts = self._get_shared_splits(self.expand_splits_by_subtree)
        subtree_splits = self.expand_splits_by_subtree.get(
            subtree, PartitionSet(encoding=self.encoding)
        )
        return PartitionSet(
            {s for s in subtree_splits if shared_counts.get(s, 0) > 1},
            encoding=self.encoding,
        )

    def get_expand_splits_for_last_user(
        self, subtree: Partition
    ) -> PartitionSet[Partition]:
        """
        Get expand splits that should be processed by this subtree because it's the last one that needs them.
        Implements the 'expand-last' strategy.
        """
        shared_counts = self._get_shared_splits(self.expand_splits_by_subtree)
        subtree_splits = self.expand_splits_by_subtree.get(
            subtree, PartitionSet(encoding=self.encoding)
        )
        return PartitionSet(
            {s for s in subtree_splits if shared_counts.get(s, 0) == 1},
            encoding=self.encoding,
        )

    def get_unique_collapse_splits(self, subtree: Partition) -> PartitionSet[Partition]:
        """Get unique collapse splits for this subtree (not shared with others)."""
        return self.collapse_splits_by_subtree.get(
            subtree, PartitionSet(encoding=self.encoding)
        ) - self.get_available_shared_collapse_splits(subtree)

    def get_unique_expand_splits(self, subtree: Partition) -> PartitionSet[Partition]:
        """Get unique expand splits for this subtree (not shared with others)."""
        return self.expand_splits_by_subtree.get(
            subtree, PartitionSet(encoding=self.encoding)
        ) - self.get_available_shared_expand_splits(subtree)

    # ============================================================================
    # 3. Split Deletion and Processing
    # ============================================================================

    def _delete_collapse_split(self, split: Partition) -> None:
        """Delete a collapse split from all subtrees."""
        for subtree_paths in self.collapse_splits_by_subtree.values():
            subtree_paths.discard(split)

    def _delete_expand_split(self, split: Partition) -> None:
        """Delete an expand split from all subtrees."""
        for subtree_paths in self.expand_splits_by_subtree.values():
            subtree_paths.discard(split)

    def delete_global_collapse_splits(self, splits: PartitionSet[Partition]) -> None:
        """Public helper: delete these collapse splits globally."""
        logger = logging.getLogger(__name__)
        for split in splits:
            logger.info(
                "[state] global collapse(delete incompatible): %s", list(split.indices)
            )
            self._delete_collapse_split(split)

    def _cleanup_empty_subtree_entries(self) -> None:
        """Remove subtrees that have no remaining splits from the dictionaries."""
        # Remove empty entries from collapse_splits_by_subtree
        empty_collapse_keys = [
            subtree
            for subtree, splits in self.collapse_splits_by_subtree.items()
            if not splits
        ]
        for subtree in empty_collapse_keys:
            del self.collapse_splits_by_subtree[subtree]

        # Remove empty entries from expand_splits_by_subtree
        empty_expand_keys = [
            subtree
            for subtree, splits in self.expand_splits_by_subtree.items()
            if not splits
        ]
        for subtree in empty_expand_keys:
            del self.expand_splits_by_subtree[subtree]

    def mark_splits_as_processed(
        self,
        subtree: Partition,
        processed_collapse_splits: PartitionSet[Partition],
        processed_expand_splits: PartitionSet[Partition],
    ) -> None:
        """Mark splits as processed by removing them from all data structures."""
        for split in processed_collapse_splits:
            self._delete_collapse_split(split)

        for split in processed_expand_splits:
            self._delete_expand_split(split)

        # FIX: Clean up empty subtree entries to prevent endless loops
        self._cleanup_empty_subtree_entries()

    # ============================================================================
    # 4. Subtree Selection and Prioritization
    # ============================================================================

    def get_next_subtree(self) -> Optional[Partition]:
        """
        Selects the next subtree to process based on a clear priority system.

        Priority Order (lower is better):
        1. (Priority 0): Subtrees with shared COLLAPSE splits. Tie-breaker: fewer shared splits.
        2. (Priority 1): Subtrees with only unique splits (no shared dependencies).
        3. (Priority 2): Subtrees with shared EXPAND splits. Tie-breaker: fewer shared splits.
        """
        unprocessed_subtrees = (
            set(self.collapse_splits_by_subtree.keys())
            | set(self.expand_splits_by_subtree.keys())
        ) - self.processed_subtrees

        if not unprocessed_subtrees:
            return None

        candidates: list[tuple[int, int, str, Partition]] = []
        for subtree in unprocessed_subtrees:
            # FIX: Add deterministic tie-breaker using string representation of partition indices
            tie_breaker = str(sorted(list(subtree.indices)))

            shared_collapse = self.get_available_shared_collapse_splits(subtree)
            if len(shared_collapse) > 0:
                priority_score = (0, len(shared_collapse), tie_breaker, subtree)
                candidates.append(priority_score)
                continue

            shared_expand = self.get_available_shared_expand_splits(subtree)
            if len(shared_expand) > 0:
                priority_score = (2, len(shared_expand), tie_breaker, subtree)
                candidates.append(priority_score)
                continue

            priority_score = (1, 0, tie_breaker, subtree)
            candidates.append(priority_score)

        if not candidates:
            return None

        _, _, _, best_subtree = min(candidates)
        return best_subtree

    # ============================================================================
    # 5. Compatibility/Incompatibility Logic
    # ============================================================================

    def find_all_incompatible_splits_for_expand(
        self,
        expand_partitions: PartitionSet[Partition],
        all_available_collapse_splits: PartitionSet[Partition],
    ) -> PartitionSet[Partition]:
        """
        Find ALL incompatible splits from the entire tree that conflict with the given expand partitions.
        """
        return find_incompatible_splits(
            reference_splits=expand_partitions,
            candidate_splits=all_available_collapse_splits,
        )

    def consume_compatible_expand_splits_for_subtree(
        self,
        subtree: Partition,
        collapsed_splits: PartitionSet[Partition],
    ) -> PartitionSet[Partition]:
        """
        Finds compatible expand splits for this subtree AND marks them as used.
        This is an atomic operation to prevent reuse.
        """
        compatible_splits: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )

        if not collapsed_splits:
            return compatible_splits

        unique_collapsed_splits: PartitionSet[Partition] = (
            self.get_unique_collapse_splits(subtree)
        )

        if not unique_collapsed_splits:
            return compatible_splits

        biggest_unique_collapsed: Partition = max(
            unique_collapsed_splits, key=lambda split: len(split.indices)
        )

        # Find splits that are subsets of the collapsed area and not already used
        for expand_split in self.available_compatible_splits:
            is_subset = set(expand_split.indices).issubset(
                set(biggest_unique_collapsed.indices)
            )
            if is_subset and expand_split not in self.used_compatible_splits:
                compatible_splits.add(expand_split)

        # FIX: Immediately mark the found splits as used to prevent reuse.
        self.used_compatible_splits |= compatible_splits

        return compatible_splits

    # ============================================================================
    # 6. Remaining Work Queries
    # ============================================================================

    def get_remaining_subtrees(self) -> set[Partition]:
        """Get the set of subtrees that still have work to do."""
        all_subtrees: set[Partition] = set(
            self.collapse_splits_by_subtree.keys()
        ) | set(self.expand_splits_by_subtree.keys())
        return all_subtrees - self.processed_subtrees

    def is_last_subtree(self, subtree: Partition) -> bool:
        """Check if the given subtree is the last one with work to do."""
        remaining: set[Partition] = self.get_remaining_subtrees()
        return len(remaining) == 1 and subtree in remaining

    def has_remaining_work(self) -> bool:
        """Check if there are still subtrees with work to do."""
        # More efficient than building a full list of remaining subtrees
        all_subtrees = set(self.collapse_splits_by_subtree.keys()) | set(
            self.expand_splits_by_subtree.keys()
        )
        return any(sub not in self.processed_subtrees for sub in all_subtrees)

    def get_all_remaining_expand_splits(self) -> PartitionSet[Partition]:
        """
        Gathers all expand splits that have not yet been processed.
        """
        remaining_splits: PartitionSet[Partition] = PartitionSet(encoding=self.encoding)

        for subtree_splits in self.expand_splits_by_subtree.values():
            remaining_splits |= subtree_splits

        remaining_splits |= (
            self.available_compatible_splits - self.used_compatible_splits
        )

        return remaining_splits
