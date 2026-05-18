from typing import AbstractSet, Mapping, Optional, Set, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from ..claims import (
    claim_containing_expand_splits,
    claim_contingent_expand_splits,
    claim_valid_transition_splits,
    SplitClaimTracker,
)
from ..ordering import PathGroupManager
from ..ordering.mover_selection import (
    remaining_mover_subtrees,
    select_next_mover_subtree,
)


class PivotTransitionState:
    """
    Tracks split ownership while building one pivot-edge transition plan.

    Collapse splits are resources to remove from the source topology. Expand
    splits are resources to create in the destination topology. Mover subtrees
    claim those resources until each mover step is planned and marked complete.
    """

    def __init__(
        self,
        all_collapse_splits: PartitionSet[Partition],
        all_expand_splits: PartitionSet[Partition],
        collapse_splits_by_subtree: Mapping[Partition, AbstractSet[Partition]],
        expand_splits_by_subtree: Mapping[Partition, AbstractSet[Partition]],
        pivot_edge: Partition,
        use_path_grouping: bool = True,
        subtree_order_key: Optional[Mapping[Partition, Tuple[int, ...]]] = None,
    ):
        """
        Initialize the interpolation state.

        Args:
            all_collapse_splits: All unique splits in tree1 pivot edge (not in tree2)
            all_expand_splits: All unique splits in tree2 pivot edge (not in tree1)
            collapse_splits_by_subtree: Initial collapse splits assigned to each subtree
            expand_splits_by_subtree: Initial expand splits assigned to each subtree
            pivot_edge: The edge being processed
            use_path_grouping: Whether to use path-based grouping for subtree ordering
            subtree_order_key: Optional visual order key for equal-priority subtrees
        """
        self.encoding = pivot_edge.encoding
        self.processed_subtrees: Set[Partition] = set()
        self._subtree_order_key: Mapping[Partition, Tuple[int, ...]] = (
            subtree_order_key or {}
        )

        self.collapse_tracker = SplitClaimTracker(self.encoding)
        self.expand_tracker = SplitClaimTracker(self.encoding)

        claim_valid_transition_splits(
            self.collapse_tracker,
            collapse_splits_by_subtree,
            all_collapse_splits,
        )
        claim_valid_transition_splits(
            self.expand_tracker,
            expand_splits_by_subtree,
            all_expand_splits,
        )
        claim_containing_expand_splits(
            self.expand_tracker,
            expand_splits_by_subtree,
            all_expand_splits,
        )

        # Store original full sets for incompatibility checks and final cleanup
        self.all_collapsible_splits = all_collapse_splits
        self.all_expand_splits = all_expand_splits

        # Track which splits have been used (for last subtree cleanup)
        self.used_expand_splits: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )

        # Track first subtree for stepwise bookkeeping.
        self.first_subtree_processed: bool = False

        # Initialize path group manager for topological ordering
        self._path_group_manager: Optional[PathGroupManager] = None
        if use_path_grouping and expand_splits_by_subtree:
            # Use COMPREHENSIVE split ownership from tracker (including parent/related splits)
            # This ensures that overlapping claims (like shared parent splits) are
            # correctly identified as dependencies for grouping.
            full_expand_splits = {
                subtree: self.expand_tracker.get_resources(subtree)
                for subtree in expand_splits_by_subtree
            }

            self._path_group_manager = PathGroupManager(
                expand_splits_by_subtree=full_expand_splits,
                encoding=self.encoding,
                enabled=True,
                subtree_order_key=self._subtree_order_key,
            )

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
        return self.collapse_tracker.get_shared_resources(subtree)

    def get_expand_splits_for_last_user(
        self, subtree: Partition
    ) -> PartitionSet[Partition]:
        """
        Get expand splits where this subtree is now the last remaining user.

        Implements the "expand-last" strategy: returns splits where this subtree
        is the sole remaining owner. Works for both unique splits (always last owner)
        and shared splits (becomes last owner after others are processed).

        IMPORTANT: For shared splits, we check if this subtree is the last owner
        among UNPROCESSED subtrees. This ensures that when earlier subtrees are
        processed and release their claims, the remaining subtrees can pick up
        the shared splits.
        """
        # Get all resources owned by this subtree
        subtree_resources = self.expand_tracker.get_resources(subtree)

        # Filter to only those where this subtree is the last owner
        # Note: is_last_owner() correctly handles both unique and shared splits
        last_user_splits = {
            split
            for split in subtree_resources
            if self.expand_tracker.is_last_owner(split, subtree)
        }

        return PartitionSet(last_user_splits, encoding=self.encoding)

    def get_unique_collapse_splits(self, subtree: Partition) -> PartitionSet[Partition]:
        """
        Get collapse splits that are unique to this subtree.

        Returns splits that are owned exclusively by this subtree.
        """
        return self.collapse_tracker.get_unique_resources(subtree)

    def get_unique_expand_splits(self, subtree: Partition) -> PartitionSet[Partition]:
        """
        Get expand splits that are unique to this subtree.

        Returns splits that are owned exclusively by this subtree.
        """
        return self.expand_tracker.get_unique_resources(subtree)

    # ============================================================================
    # 4. Split Processing
    # ============================================================================

    def mark_splits_as_processed(
        self,
        subtree: Partition,
        processed_collapse_splits: PartitionSet[Partition],
        processed_expand_splits: PartitionSet[Partition],
    ) -> None:
        """
        Mark splits as processed by removing them from tracking structures.

        Collapse splits are deleted globally (all subtrees lose access).
        Expand splits (including contingent) are tracked as used for bookkeeping.

        Note: After processing, this subtree releases ALL expand claims (not just
        the processed ones) to enable the "expand-last" strategy where remaining
        subtrees become the last users of shared splits.
        """
        # Mark subtree as processed FIRST (truth-based: once marked, won't be reprocessed)
        self.processed_subtrees.add(subtree)

        # Process collapse splits: delete globally
        for split in processed_collapse_splits:
            self.collapse_tracker.release_all(split)
            # Remove from global snapshot to keep incompatibility checks accurate
            self.all_collapsible_splits.discard(split)

        # Track which expand splits were used (for bookkeeping/queries)
        # This includes contingent splits that were consumed and added to processed_expand_splits
        self.used_expand_splits |= processed_expand_splits

        # Release ALL expand claims for this subtree (expand-last strategy)
        # This allows remaining subtrees to become the last users of shared splits
        # Note: Contingent splits consumed by this subtree are automatically released here
        self.expand_tracker.release_owner_from_all_resources(subtree)

    # ============================================================================
    # 5. Subtree Selection and Prioritization
    # ============================================================================

    def get_next_subtree(self) -> Optional[Partition]:
        """
        Select next subtree to process.

        Priority system:
        1. If any subtree has shared collapse work: prioritize by shared collapse count,
           then unique-only, then shared expand count
        2. Otherwise: delegate to PathGroupManager for topological ordering based on
           expand path relationships
        3. Fallback: select subtree with smallest expand path

        Tie-breaker: optional visual order key, then bitmask
        """
        return select_next_mover_subtree(
            collapse_tracker=self.collapse_tracker,
            expand_tracker=self.expand_tracker,
            path_group_manager=self._path_group_manager,
            processed_subtrees=self.processed_subtrees,
            subtree_order_key=self._subtree_order_key,
        )

    # ============================================================================
    # 6. Compatibility/Incompatibility Logic
    # ============================================================================

    def mark_first_subtree_processed(self) -> None:
        """
        Mark that the first subtree has been processed.

        The stepwise planner uses this as bookkeeping so the first processed
        subtree is tracked even when it has no collapse work.
        """
        self.first_subtree_processed = True

    def consume_contingent_expand_splits_for_subtree(
        self,
        subtree: Partition,
        collapsed_splits: PartitionSet[Partition],
    ) -> PartitionSet[Partition]:
        """
        Finds contingent expand splits for this subtree.

        Contingent splits are those not assigned to any primary subtree. They
        can be opportunistically used when collapse operations create space.

        This is an atomic operation to prevent reuse: splits are claimed in
        the expand_tracker.

        Args:
            subtree: The subtree requesting contingent splits
            collapsed_splits: The splits that were just collapsed, creating space

        Returns:
            Contingent expand splits that fit within the collapsed region
        """
        return claim_contingent_expand_splits(
            expand_tracker=self.expand_tracker,
            all_expand_splits=self.all_expand_splits,
            subtree=subtree,
            collapsed_splits=collapsed_splits,
        )

    # ============================================================================
    # 7. Remaining Work Queries
    # ============================================================================

    def get_remaining_subtrees(self) -> set[Partition]:
        """
        Get the set of subtrees that still have work to do.

        A subtree has work if it appears as an owner of unique or shared splits,
        and has not been marked as processed.
        """
        return remaining_mover_subtrees(
            self.collapse_tracker,
            self.expand_tracker,
            self.processed_subtrees,
        )

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
        """Gather remaining collapse splits from tracker."""
        return PartitionSet(
            set(self.collapse_tracker.get_all_resources()), encoding=self.encoding
        )

    def get_all_remaining_expand_splits(self) -> PartitionSet[Partition]:
        """
        Get all expand splits that haven't been processed yet.

        Includes unassigned (contingent) splits. Uses expand_tracker as truth:
        remaining = all - used.
        """
        return self.all_expand_splits - self.used_expand_splits
