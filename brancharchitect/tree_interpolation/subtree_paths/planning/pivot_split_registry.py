from typing import Dict, Optional, Set
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from .ownership_tracker import OwnershipTracker


# ============================================================================
# Split Registry Class
# ============================================================================


class PivotSplitRegistry:
    """
    Registry that tracks split ownership and usage during tree interpolation.

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

        # Initialize ownership trackers for collapse and expand splits
        self.collapse_tracker = OwnershipTracker(self.encoding)
        self.expand_tracker = OwnershipTracker(self.encoding)

        # Populate trackers from initial assignments
        for subtree, splits in collapse_splits_by_subtree.items():
            self.collapse_tracker.claim_batch(splits, subtree)

        for subtree, splits in expand_splits_by_subtree.items():
            self.expand_tracker.claim_batch(splits, subtree)

        # Store original full sets for incompatibility checks and final cleanup
        self.all_collapsible_splits = all_collapse_splits
        self.all_expand_splits = all_expand_splits

        # Track which splits have been used (for last subtree cleanup)
        self.used_expand_splits: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )

        # Track first subtree for tabula rasa strategy
        self.first_subtree_processed: bool = False

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

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
        - If any subtree has shared collapse work: prioritize by shared collapse count,
          then unique-only, then shared expand count
        - Otherwise: select subtree with longest expand path
        - Tie-breaker: lexicographic ordering of indices
        """
        unprocessed = self.get_remaining_subtrees()
        if not unprocessed:
            return None

        # Check if any unprocessed subtree has shared collapse work
        has_shared_collapse = any(
            self.collapse_tracker.get_shared_resources(sub) for sub in unprocessed
        )

        if has_shared_collapse:
            # Priority system: shared collapse > unique-only > shared expand
            candidates = []
            for subtree in unprocessed:
                shared_collapse = self.collapse_tracker.get_shared_resources(subtree)
                tie_breaker = str(sorted(list(subtree.indices)))

                if shared_collapse:
                    priority = (0, -len(shared_collapse), tie_breaker)
                elif self.expand_tracker.get_shared_resources(subtree):
                    shared_expand = self.expand_tracker.get_shared_resources(subtree)
                    priority = (2, -len(shared_expand), tie_breaker)
                else:
                    priority = (1, 0, tie_breaker)

                candidates.append((priority, subtree))

            return min(candidates)[1]

        # No shared collapse: choose longest expand path
        candidates = [
            (
                -len(self.expand_tracker.get_resources(sub)),
                str(sorted(list(sub.indices))),
                sub,
            )
            for sub in unprocessed
        ]
        return min(candidates)[2]

    # ============================================================================
    # 6. Compatibility/Incompatibility Logic
    # ============================================================================

    def get_tabula_rasa_collapse_splits(self) -> PartitionSet[Partition]:
        """
        Get ALL collapse splits for tabula rasa (clean slate) strategy.

        The first subtree collapses EVERYTHING from the source tree to create
        a blank canvas. Then we rebuild the tree from scratch by expanding splits
        step by step. This ensures no incorrect tree structure carries over.

        Strategy: First subtree gets ALL collapse splits, regardless of which
        subtree they were originally assigned to. Subsequent subtrees get none
        (they only handle their assigned splits).

        Returns:
            ALL collapse splits if first subtree not yet processed, empty otherwise.
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
        tracked_resources = self.expand_tracker.get_all_resources()

        if not collapsed_splits:
            # If nothing was collapsed, allow claiming any unassigned expands.
            contingent_set = PartitionSet(
                {s for s in self.all_expand_splits if s not in tracked_resources},
                encoding=self.encoding,
            )
            for split in contingent_set:
                self.expand_tracker.claim(split, subtree)
            return contingent_set

        # Identify contingent splits: those NOT in expand_tracker (unclaimed)
        # and fit spatially within collapsed regions
        collapsed_regions = [set(split.indices) for split in collapsed_splits]

        contingent_set: PartitionSet[Partition] = PartitionSet(encoding=self.encoding)
        for expand_split in self.all_expand_splits:
            # Skip if already claimed by ANY subtree
            if expand_split in tracked_resources:
                continue
            # Check spatial containment
            expand_indices = set(expand_split.indices)
            if any(expand_indices.issubset(region) for region in collapsed_regions):
                contingent_set.add(expand_split)

        # Atomic consumption: claim in expand_tracker (single source of truth)
        if contingent_set:
            for split in contingent_set:
                self.expand_tracker.claim(split, subtree)

        return contingent_set

    # ============================================================================
    # 7. Remaining Work Queries
    # ============================================================================

    def get_remaining_subtrees(self) -> set[Partition]:
        """
        Get the set of subtrees that still have work to do.

        A subtree has work if it appears as an owner of unique or shared splits,
        and has not been marked as processed.
        """
        remaining: Set[Partition] = set()

        # Add all owners from both trackers
        remaining.update(self.collapse_tracker.get_all_owners())
        remaining.update(self.expand_tracker.get_all_owners())

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
