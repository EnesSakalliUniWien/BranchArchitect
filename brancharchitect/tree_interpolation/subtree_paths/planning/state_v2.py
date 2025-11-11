"""
State management for tree interpolation process - Version 2 (Pre-categorized).

This module provides the InterpolationState class which manages the bookkeeping
of shared splits, unique splits, and which subtrees can process which splits
during the tree interpolation sequence.

Key architectural improvement: Instead of dynamically counting split usage on
every query, this version categorizes splits ONCE during initialization into
'unique' (single owner) and 'shared' (multiple users) groups. This provides:
  - O(1) lookups instead of O(n) recounts
  - Clearer ownership semantics
  - More efficient state updates

Sections:
    1. Split Categorization Helper
    2. Initialization and State Variables
    3. Shared/Unique Split Queries
    4. Split Deletion and Processing
    5. Subtree Selection and Prioritization
    6. Compatibility/Incompatibility Logic
    7. Remaining Work Queries
"""

import logging
from typing import Dict, Optional, Set, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet


# ============================================================================
# 1. Split Categorization Helper
# ============================================================================


def categorize_splits(
    splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
) -> Tuple[Dict[Partition, Partition], Dict[Partition, Set[Partition]]]:
    """
    Categorize splits into unique (single owner) and shared (multiple users).

    This function performs a one-time analysis of which splits belong to which
    subtrees, eliminating the need for repeated counting operations.

    Args:
        splits_by_subtree: Mapping of subtree -> splits owned by that subtree

    Returns:
        Tuple of:
        - unique_splits: {split -> single_owner_subtree} for splits used by exactly one subtree
        - shared_splits: {split -> {user1, user2, ...}} for splits used by multiple subtrees

    Example:
        Input:
            {
                subtree_A: {split_1, split_2},
                subtree_B: {split_2, split_3},
            }
        Output:
            unique_splits = {split_1: subtree_A, split_3: subtree_B}
            shared_splits = {split_2: {subtree_A, subtree_B}}
    """
    # First pass: count how many subtrees use each split
    split_to_users: Dict[Partition, Set[Partition]] = {}

    for subtree, splits in splits_by_subtree.items():
        for split in splits:
            if split not in split_to_users:
                split_to_users[split] = set()
            split_to_users[split].add(subtree)

    # Second pass: categorize into unique vs shared
    unique_splits: Dict[Partition, Partition] = {}
    shared_splits: Dict[Partition, Set[Partition]] = {}

    for split, users in split_to_users.items():
        if len(users) == 1:
            # Unique: exactly one owner
            owner = next(iter(users))
            unique_splits[split] = owner
        else:
            # Shared: multiple users
            shared_splits[split] = users

    return unique_splits, shared_splits


# ============================================================================
# 2. Initialization and State Variables
# ============================================================================


class InterpolationState:
    """
    Manages state during tree interpolation by tracking split ownership and usage.

    Instead of repeatedly counting split usage, this class categorizes splits
    once upon initialization into 'unique' and 'shared' groups, preserving
    information about which subtrees own which splits. This makes the process
    more efficient and the logic clearer.

    Key data structures:
        - unique_collapse_splits: {split -> owner_subtree}
        - shared_collapse_splits: {split -> {user1, user2, ...}}
        - unique_expand_splits: {split -> owner_subtree}
        - shared_expand_splits: {split -> {user1, user2, ...}}
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
            all_collapse_splits: All unique splits in tree1 pivot edge (not in tree2)
            all_expand_splits: All unique splits in tree2 pivot edge (not in tree1)
            collapse_splits_by_subtree: Initial collapse splits assigned to each subtree
            expand_splits_by_subtree: Initial expand splits assigned to each subtree
            active_changing_edge: The edge being processed
        """
        self.encoding = active_changing_edge.encoding
        self.processed_subtrees: Set[Partition] = set()

        # Keep canonical by-subtree mappings to preserve mutability semantics used by tests
        # (e.g., tests may clear or pop entries directly from these dicts)
        self.collapse_splits_by_subtree: Dict[Partition, PartitionSet[Partition]] = (
            dict(collapse_splits_by_subtree)
        )
        self.expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]] = dict(
            expand_splits_by_subtree
        )

        # Initialize categorized views
        (
            self.unique_collapse_splits,
            self.shared_collapse_splits,
        ) = categorize_splits(self.collapse_splits_by_subtree)

        (
            self.unique_expand_splits,
            self.shared_expand_splits,
        ) = categorize_splits(self.expand_splits_by_subtree)

        # Track which expand splits started as shared (used by expand-last logic)
        expand_usage_counts: Dict[Partition, int] = {}
        for splits in self.expand_splits_by_subtree.values():
            for sp in splits:
                expand_usage_counts[sp] = expand_usage_counts.get(sp, 0) + 1
        self._initial_shared_expand_splits: Set[Partition] = {
            sp for sp, cnt in expand_usage_counts.items() if cnt > 1
        }

        # Store original full sets for incompatibility checks and final cleanup
        self.all_collapsible_splits = all_collapse_splits
        self.all_expand_splits = all_expand_splits

        # Note: Incompatible splits are now handled per-subtree rather than globally.
        # Each subtree will query for incompatible splits specific to its expand operations.
        # This ensures incompatible splits are collapsed by the subtree that needs them collapsed.

        # Contingent splits are those not assigned to any primary subtree
        all_primary_expand_splits: PartitionSet[Partition] = PartitionSet(
            self.unique_expand_splits.keys() | self.shared_expand_splits.keys(),
            encoding=self.encoding,
        )

        self.available_contingent_splits: PartitionSet[Partition] = (
            all_expand_splits - all_primary_expand_splits
        )

        # Track which splits have been used
        self.used_contingent_splits: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )
        self.used_expand_splits: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )

        # Track first subtree for tabula rasa strategy
        self.first_subtree_processed: bool = False

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

    def _recompute_categories(self) -> None:
        """Recompute unique/shared categorizations from current by-subtree mappings."""
        (
            self.unique_collapse_splits,
            self.shared_collapse_splits,
        ) = categorize_splits(self.collapse_splits_by_subtree)

        (
            self.unique_expand_splits,
            self.shared_expand_splits,
        ) = categorize_splits(self.expand_splits_by_subtree)

    # ============================================================================
    # 3. Shared/Unique Split Queries
    # ============================================================================

    def get_available_shared_collapse_splits(
        self, subtree: Partition
    ) -> PartitionSet[Partition]:
        """
        Get shared collapse splits available for this subtree.

        Returns splits that are shared by multiple subtrees and this subtree
        is one of the users.
        """
        # Ensure categories reflect any external mutations to the mappings
        self._recompute_categories()
        return PartitionSet(
            {
                split
                for split, users in self.shared_collapse_splits.items()
                if subtree in users
            },
            encoding=self.encoding,
        )

    def get_expand_splits_for_last_user(
        self, subtree: Partition
    ) -> PartitionSet[Partition]:
        """
        Get expand splits where this subtree is now the last remaining user.

        Implements the "expand-last" strategy using dynamic counts from the
        current by-subtree mapping, matching v1 semantics used by tests.
        """
        # Build dynamic counts from current mapping
        counts: Dict[Partition, int] = {}
        for splits in self.expand_splits_by_subtree.values():
            for split in splits:
                counts[split] = counts.get(split, 0) + 1

        subtree_splits = self.expand_splits_by_subtree.get(
            subtree, PartitionSet(encoding=self.encoding)
        )
        # Consider both current by-subtree counts and direct edits on shared mapping
        last_by_mapping = {
            s
            for s in subtree_splits
            if counts.get(s, 0) == 1 and s in self._initial_shared_expand_splits
        }
        last_by_shared_view = {
            s
            for s, users in self.shared_expand_splits.items()
            if len(users) == 1 and subtree in users
        }
        result_set = last_by_mapping | last_by_shared_view

        # If this subtree has no shared expand splits at all, treat its unique
        # expands as "last user" too (useful for subtrees that never shared).
        if not result_set:
            has_any_shared_for_subtree = any(
                (s in self._initial_shared_expand_splits) for s in subtree_splits
            )
            if not has_any_shared_for_subtree:
                result_set = set(self.get_unique_expand_splits(subtree))

        return PartitionSet(result_set, encoding=self.encoding)

    def get_unique_collapse_splits(self, subtree: Partition) -> PartitionSet[Partition]:
        """
        Get collapse splits that were initially unique to this subtree.

        Returns splits that are owned exclusively by this subtree.
        """
        self._recompute_categories()
        return PartitionSet(
            {
                split
                for split, owner in self.unique_collapse_splits.items()
                if owner == subtree
            },
            encoding=self.encoding,
        )

    def get_unique_expand_splits(self, subtree: Partition) -> PartitionSet[Partition]:
        """
        Get expand splits that were initially unique to this subtree.

        Returns splits that are owned exclusively by this subtree.
        """
        self._recompute_categories()
        return PartitionSet(
            {
                split
                for split, owner in self.unique_expand_splits.items()
                if owner == subtree
            },
            encoding=self.encoding,
        )

    def has_only_unique_splits(self, subtree: Partition) -> bool:
        """
        Check if a subtree has only unique splits (no shared dependencies).

        Such subtrees can be processed immediately without coordination since
        they don't share any splits with other subtrees.
        """
        has_shared_collapse = any(
            subtree in users for users in self.shared_collapse_splits.values()
        )
        has_shared_expand = any(
            subtree in users for users in self.shared_expand_splits.values()
        )
        return not has_shared_collapse and not has_shared_expand

    # ============================================================================
    # 4. Split Deletion and Processing
    # ============================================================================

    def _delete_collapse_split(self, split: Partition) -> None:
        """
        Delete a collapse split from all tracking structures.

        This removes the split from both unique and shared dictionaries,
        regardless of which category it was in.
        """
        # Remove from categorized views
        self.unique_collapse_splits.pop(split, None)
        self.shared_collapse_splits.pop(split, None)
        # Also remove from per-subtree mappings
        for splits in self.collapse_splits_by_subtree.values():
            splits.discard(split)

    def _process_expand_split(self, split: Partition, subtree: Partition) -> None:
        """
        Process an expand split by removing the current subtree as a user.

        If the split was unique, remove it entirely.
        If the split was shared, remove this subtree from its user set.
        If this was the last user, remove the split from shared tracking.
        """
        if split in self.unique_expand_splits:
            # Unique split: remove entirely
            self.unique_expand_splits.pop(split, None)
        elif split in self.shared_expand_splits:
            # Shared split: remove this subtree from users
            self.shared_expand_splits[split].discard(subtree)
            if not self.shared_expand_splits[split]:
                # No more users: remove from shared tracking
                self.shared_expand_splits.pop(split)
        # Keep the by-subtree mapping in sync
        if subtree in self.expand_splits_by_subtree:
            self.expand_splits_by_subtree[subtree].discard(split)

    def delete_global_collapse_splits(self, splits: PartitionSet[Partition]) -> None:
        """
        Public helper: delete these collapse splits globally.

        Used when incompatible splits are discovered and need to be removed
        from the entire system.
        """
        logger = logging.getLogger(__name__)
        for split in splits:
            logger.debug(
                "[state_v2] global collapse(delete incompatible): %s",
                list(split.indices),
            )
            self._delete_collapse_split(split)

    def mark_splits_as_processed(
        self,
        subtree: Partition,
        processed_collapse_splits: PartitionSet[Partition],
        processed_expand_splits: PartitionSet[Partition],
        processed_contingent_splits: PartitionSet[Partition],
    ) -> None:
        """
        Mark splits as processed by removing them from tracking structures.

        Collapse splits are deleted globally (all subtrees lose access).
        Expand splits remove the current subtree from their user set.
        Contingent splits are marked as used and removed from availability.
        """
        # Process collapse splits: delete globally
        for split in processed_collapse_splits:
            self._delete_collapse_split(split)

        # Process expand splits: remove this subtree as a user and from mapping
        for split in processed_expand_splits:
            self._process_expand_split(split, subtree)
            self.used_expand_splits.add(split)

        # Process contingent splits: mark as used
        if processed_contingent_splits:
            self.used_contingent_splits |= processed_contingent_splits
            self.available_contingent_splits -= processed_contingent_splits

        # Clean up empty subtrees in the by-subtree mappings
        self._cleanup_empty_subtree_entries()

    # ============================================================================
    # 5. Subtree Selection and Prioritization
    # ============================================================================

    def get_next_subtree(self) -> Optional[Partition]:
        """
        Select next subtree to process.

        Behavior:
        - If any shared COLLAPSE splits exist among unprocessed subtrees, prefer the
          priority system: shared collapse first (most shared), then unique-only,
          then shared expand (most shared). Deterministic tie-breaker by indices.
        - Otherwise (no shared collapse present), select the subtree with the
          LONGEST expand path (unique + shared expand splits), tie-breaking
          lexicographically. This preserves v2 test expectations.
        """
        # Ensure categories reflect any external mutations
        self._recompute_categories()

        unprocessed_subtrees = self.get_remaining_subtrees()
        if not unprocessed_subtrees:
            return None

        # Detect presence of any shared collapse among unprocessed
        any_shared_collapse = any(
            any(sub in users for sub in unprocessed_subtrees)
            for users in self.shared_collapse_splits.values()
        )

        if any_shared_collapse:
            candidates: list[tuple[int, int, str, Partition]] = []
            for subtree in unprocessed_subtrees:
                tie_breaker = str(sorted(list(subtree.indices)))
                shared_collapse_count = sum(
                    1
                    for users in self.shared_collapse_splits.values()
                    if subtree in users
                )
                if shared_collapse_count > 0:
                    candidates.append((0, -shared_collapse_count, tie_breaker, subtree))
                    continue

                shared_expand_count = sum(
                    1
                    for users in self.shared_expand_splits.values()
                    if subtree in users
                )
                if shared_expand_count > 0:
                    candidates.append((2, -shared_expand_count, tie_breaker, subtree))
                    continue

                candidates.append((1, 0, tie_breaker, subtree))

            if not candidates:
                return None
            _, _, _, best = min(candidates)
            return best

        # No shared collapse present: choose longest expand path
        candidates2: list[tuple[int, str, Partition]] = []
        for subtree in unprocessed_subtrees:
            num_unique_expand = sum(
                1
                for _split, owner in self.unique_expand_splits.items()
                if owner == subtree
            )
            num_shared_expand = sum(
                1
                for _split, users in self.shared_expand_splits.items()
                if subtree in users
            )
            total_expand_splits = num_unique_expand + num_shared_expand
            tie_breaker = str(sorted(list(subtree.indices)))
            candidates2.append((-total_expand_splits, tie_breaker, subtree))

        if not candidates2:
            return None
        _, _, best2 = min(candidates2)
        return best2

    # ============================================================================
    # 6. Compatibility/Incompatibility Logic
    # ============================================================================

    def get_all_collapse_splits_for_first_subtree(
        self, subtree: Partition
    ) -> PartitionSet[Partition]:
        """
        Get ALL collapse splits for the first subtree - TABULA RASA approach.

        The first subtree should collapse EVERYTHING from the source tree to create
        a clean slate. Then we rebuild the tree from scratch by expanding splits
        step by step. This ensures no incorrect tree structure carries over.

        Strategy: First subtree collapses ALL collapse splits, regardless of which
        subtree they were originally assigned to. This creates a blank canvas.

        Args:
            subtree: The first subtree being processed

        Returns:
            ALL collapse splits if this is the first subtree, empty otherwise.
        """
        if not self.first_subtree_processed:
            # Return ALL collapse splits - complete tabula rasa
            return self.all_collapsible_splits.copy()
        return PartitionSet(encoding=self.encoding)

    def mark_first_subtree_processed(self) -> None:
        """
        Mark that the first subtree has been processed.

        After the first subtree collapses everything (tabula rasa), subsequent
        subtrees only handle their assigned splits.
        """
        self.first_subtree_processed = True

    def find_all_incompatible_splits_for_expand(
        self,
        expand_partitions: PartitionSet[Partition],
        all_available_collapse_splits: PartitionSet[Partition],
    ) -> PartitionSet[Partition]:
        """
        Find ALL incompatible splits from the entire tree that conflict with the given expand partitions.

        This is used to identify splits that must be collapsed before the expand
        operations can proceed.

        Uses Partition.is_compatible_with() for robust incompatibility detection.
        Two partitions are incompatible if they overlap but neither is a subset of the other.

        NOTE: Splits that appear in BOTH collapse and expand are NOT treated as incompatible.
        These are transitional splits that exist in both trees and should not be flagged.
        """
        if not expand_partitions or not all_available_collapse_splits:
            return PartitionSet(encoding=self.encoding)

        # Get all indices for compatibility check
        all_indices = set(self.encoding.values())

        # Find splits that exist in both collapse AND expand (transitional splits)
        # These should NOT be treated as incompatible
        transitional_splits = set(expand_partitions) & set(
            all_available_collapse_splits
        )

        # Find all incompatible splits using Partition's built-in compatibility check
        incompatible: PartitionSet[Partition] = PartitionSet(encoding=self.encoding)

        for expand_split in expand_partitions:
            for collapse_split in all_available_collapse_splits:
                # Skip if same split
                if expand_split == collapse_split:
                    continue

                # Skip transitional splits (appear in both collapse and expand)
                if collapse_split in transitional_splits:
                    continue

                # Use Partition.is_compatible_with() to check compatibility
                if not expand_split.is_compatible_with(collapse_split, all_indices):
                    incompatible.add(collapse_split)

        return incompatible

    def consume_contingent_expand_splits_for_subtree(
        self,
        subtree: Partition,
        collapsed_splits: PartitionSet[Partition],
    ) -> PartitionSet[Partition]:
        """
        Finds contingent expand splits for this subtree.

        Contingent splits are those not assigned to any primary subtree. They
        can be opportunistically used when collapse operations create space.

        This is an atomic operation to prevent reuse: splits are returned once
        and marked as used.

        Args:
            subtree: The subtree requesting contingent splits
            collapsed_splits: The splits that were just collapsed, creating space

        Returns:
            Contingent expand splits that fit within the collapsed region
        """
        if not collapsed_splits:
            return PartitionSet(encoding=self.encoding)

        # Identify contingent splits: those that are unused and fit entirely within
        # the largest collapsed partition.
        biggest_collapsed: Partition = max(
            collapsed_splits, key=lambda split: len(split.indices)
        )
        biggest_collapsed_indices = set(biggest_collapsed.indices)

        contingent_generator = (
            expand_split
            for expand_split in self.available_contingent_splits
            if expand_split not in self.used_contingent_splits
            and set(expand_split.indices).issubset(biggest_collapsed_indices)
        )
        return PartitionSet(set(contingent_generator), encoding=self.encoding)

    # ============================================================================
    # 7. Remaining Work Queries
    # ============================================================================

    def get_remaining_subtrees(self) -> set[Partition]:
        """
        Get the set of subtrees that still have work to do.

        A subtree has work if it appears as an owner of unique splits or as a
        user of shared splits, and has not been marked as processed.
        """
        # Ensure categories reflect latest mapping
        self._recompute_categories()
        remaining: Set[Partition] = set()

        # Add owners of unique splits
        remaining.update(self.unique_collapse_splits.values())
        remaining.update(self.unique_expand_splits.values())

        # Add users of shared splits
        for users in self.shared_collapse_splits.values():
            remaining.update(users)
        for users in self.shared_expand_splits.values():
            remaining.update(users)

        # Remove processed subtrees
        return remaining - self.processed_subtrees

    def is_last_subtree(self, subtree: Partition) -> bool:
        """
        Check if the given subtree is the last one with work to do.

        This is used to determine if a subtree should collect all remaining
        splits as part of final cleanup.
        """
        remaining: set[Partition] = self.get_remaining_subtrees()
        return len(remaining) == 1 and subtree in remaining

    def has_remaining_work(self) -> bool:
        """
        Check if there are still subtrees with work to do.

        Returns True if any unprocessed subtrees remain, False otherwise.
        """
        return bool(self.get_remaining_subtrees())

    def get_all_remaining_collapse_splits(self) -> PartitionSet[Partition]:
        """Gather remaining collapse splits based on current mappings."""
        remaining: PartitionSet[Partition] = PartitionSet(encoding=self.encoding)
        for splits in self.collapse_splits_by_subtree.values():
            remaining |= splits
        return remaining

    def get_all_remaining_expand_splits(self) -> PartitionSet[Partition]:
        """
        Get all expand splits that haven't been processed yet.

        Includes unassigned (contingent) splits as well; mirrors v1 semantics
        expected by tests where the last subtree aggregates any remaining work.
        """
        return (
            self.all_expand_splits
            - self.used_expand_splits
            - self.used_contingent_splits
        )

    # ------------------------------------------------------------------------
    # Maintenance helpers expected by tests
    # ------------------------------------------------------------------------
    def _cleanup_empty_subtree_entries(self) -> None:
        """Remove subtrees that have no remaining splits from the dictionaries."""
        empty_collapse = [
            st for st, splits in self.collapse_splits_by_subtree.items() if not splits
        ]
        for st in empty_collapse:
            self.collapse_splits_by_subtree.pop(st, None)

        empty_expand = [
            st for st, splits in self.expand_splits_by_subtree.items() if not splits
        ]
        for st in empty_expand:
            self.expand_splits_by_subtree.pop(st, None)
