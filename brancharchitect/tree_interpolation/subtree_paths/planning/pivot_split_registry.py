from typing import Dict, Optional, Set, Tuple, Any
from collections import OrderedDict
import logging
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from .split_claim_tracker import SplitClaimTracker
from .path_group_manager import PathGroupManager
from ..analysis.split_analysis import (
    find_incompatible_splits,
    get_unique_splits_for_current_pivot_edge_subtree,
)
from brancharchitect.tree import Node

logger = logging.getLogger(__name__)


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
        use_path_grouping: bool = True,
    ):
        """
        Initialize the interpolation state.

        Args:
            all_collapse_splits: All unique splits in tree1 pivot edge (not in tree2)
            all_expand_splits: All unique splits in tree2 pivot edge (not in tree1)
            collapse_splits_by_subtree: Initial collapse splits assigned to each subtree
            expand_splits_by_subtree: Initial expand splits assigned to each subtree
            active_changing_edge: The edge being processed
            use_path_grouping: Whether to use path-based grouping for subtree ordering
        """
        self.encoding = active_changing_edge.encoding
        self.processed_subtrees: Set[Partition] = set()

        # Initialize ownership trackers for collapse and expand splits
        self.collapse_tracker = SplitClaimTracker(self.encoding)
        self.expand_tracker = SplitClaimTracker(self.encoding)

        # Populate trackers from initial assignments
        # Use intersection (&) to ensure we only claim splits that are globally valid
        # This filters out "shared splits" that exist in both trees but appear in local paths
        for subtree, splits in collapse_splits_by_subtree.items():
            valid_splits = splits & all_collapse_splits
            self.collapse_tracker.claim_batch(valid_splits, subtree)

        for subtree, splits in expand_splits_by_subtree.items():
            valid_splits = splits & all_expand_splits
            self.expand_tracker.claim_batch(valid_splits, subtree)

        # CRITICAL: Claim any expand split that CONTAINS a subtree's taxa (Parent),
        # AND any split that is a SIBLING (child of a Parent, disjoint from subtree).
        self._claim_related_expand_splits(expand_splits_by_subtree, all_expand_splits)

        # Store original full sets for incompatibility checks and final cleanup
        self.all_collapsible_splits = all_collapse_splits
        self.all_expand_splits = all_expand_splits

        # Track which splits have been used (for last subtree cleanup)
        self.used_expand_splits: PartitionSet[Partition] = PartitionSet(
            encoding=self.encoding
        )

        # Track first subtree for tabula rasa strategy
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
            )

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

    def _claim_related_expand_splits(
        self,
        initial_assignments: Dict[Partition, PartitionSet[Partition]],
        all_expand_splits: PartitionSet[Partition],
    ) -> None:
        """
        Force subtrees to claim ownership of related structural splits.

        1. Containing Splits (Parents): If a split contains the subtree, the subtree
           owns it (it's inside).

        All subtrees inside a parent split claim it. The "expand-last" strategy
        handles shared ownership by having the last remaining owner expand the split.
        """
        for split in all_expand_splits:
            split_taxa = split.taxa
            for subtree in initial_assignments:
                # If subtree is a subset of the split, it's a "Parent" split
                if subtree.taxa.issubset(split_taxa):
                    self.expand_tracker.claim(split, subtree)

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
    # 3.5 Path Computation (encapsulates all path logic)
    # ============================================================================

    def compute_paths_for_subtree(
        self, subtree: Partition
    ) -> Tuple[PartitionSet[Partition], PartitionSet[Partition]]:
        """
        Compute collapse and expand paths for a subtree.

        This method encapsulates all path computation logic:
        1. Gathers split categories (shared/unique collapse, last-user/unique expand)
        2. Applies tabula rasa strategy for first subtree
        3. Computes incompatible splits that must be collapsed
        4. Consumes contingent expand splits that fit in collapsed regions
        5. Handles last subtree cleanup (remaining splits)

        Args:
            subtree: The subtree to compute paths for

        Returns:
            Tuple of (collapse_path, expand_path) as PartitionSets
        """
        # Debug: Log subtree being processed
        logger.info(f"=== Computing paths for subtree: {subtree.taxa} ===")

        # 1. Gather split categories for this subtree
        shared_collapse = self.get_available_shared_collapse_splits(subtree)
        unique_collapse = self.get_unique_collapse_splits(subtree)
        last_user_expand = self.get_expand_splits_for_last_user(subtree)
        unique_expand = self.get_unique_expand_splits(subtree)

        # Debug: Log expand splits ownership
        all_subtree_expand = self.expand_tracker.get_resources(subtree)
        logger.info(
            f"  Subtree expand splits ({len(all_subtree_expand)}): {[s.taxa for s in all_subtree_expand]}"
        )
        logger.info(
            f"  Unique expand ({len(unique_expand)}): {[s.taxa for s in unique_expand]}"
        )
        logger.info(
            f"  Last-user expand ({len(last_user_expand)}): {[s.taxa for s in last_user_expand]}"
        )

        # Debug: For each expand split, log all owners
        for split in all_subtree_expand:
            owners = self.expand_tracker.get_owners(split)
            is_last = self.expand_tracker.is_last_owner(split, subtree)
            logger.info(
                f"    Split {split.taxa}: owners={[o.taxa for o in owners]}, is_last_owner={is_last}"
            )

        # 2. Compute collapse path
        collapse_path = self._compute_collapse_path(
            shared_collapse, unique_collapse, last_user_expand, unique_expand
        )

        # 3. Consume contingent expand splits that fit within collapsed regions
        contingent_expand = self.consume_contingent_expand_splits_for_subtree(
            subtree=subtree, collapsed_splits=collapse_path
        )

        # 4. Build expand path
        # Include last-user expands immediately to match build_edge_plan semantics.
        expand_path = build_expand_path(
            last_user_expand, unique_expand, contingent_expand
        )

        # Debug: Log final expand path
        logger.info(
            f"  Final expand path ({len(expand_path)}): {[s.taxa for s in expand_path]}"
        )

        # 5. Handle last subtree cleanup
        if self.is_last_subtree(subtree):
            expand_path |= self.get_all_remaining_expand_splits()
            collapse_path |= self.get_all_remaining_collapse_splits()

        return collapse_path, expand_path

    def _compute_collapse_path(
        self,
        shared_collapse: PartitionSet[Partition],
        unique_collapse: PartitionSet[Partition],
        last_user_expand: PartitionSet[Partition],
        unique_expand: PartitionSet[Partition],
    ) -> PartitionSet[Partition]:
        """
        Compute the collapse path for a subtree using STEPWISE strategy.

        Stepwise Strategy:
        - Each subtree collapses ONLY its assigned splits + incompatible splits.
        - This produces smoother animations (incremental changes) compared to
          Tabula Rasa (which collapsed everything at once for the first subtree).

        The collapse path is the union of:
        1. shared_collapse: Splits shared with other subtrees (first to process wins)
        2. unique_collapse: Splits owned exclusively by this subtree
        3. incompatible: Splits that conflict with planned expansions

        Mathematical correctness is guaranteed because:
        - Incompatible splits are computed against the GLOBAL remaining set
        - The global set is updated after each subtree processes
        - Any split blocking an expansion will be collapsed (regardless of owner)

        Args:
            shared_collapse: Shared collapse splits for this subtree
            unique_collapse: Unique collapse splits for this subtree
            last_user_expand: Expand splits where this subtree is last user
            unique_expand: Unique expand splits for this subtree

        Returns:
            The collapse path for this subtree
        """
        # Mark first subtree as processed for bookkeeping (even without Tabula Rasa)
        if not self.first_subtree_processed:
            self.first_subtree_processed = True

        # Compute incompatibilities for this subtree's planned expands
        # NOTE: Don't include contingent_expand here - they haven't been consumed yet!
        prospective_expand = last_user_expand | unique_expand
        incompatible = find_incompatible_splits(
            prospective_expand, self.all_collapsible_splits
        )

        return shared_collapse | unique_collapse | incompatible

    def mark_subtree_complete(
        self,
        subtree: Partition,
        collapse_path: PartitionSet[Partition],
        expand_path: PartitionSet[Partition],
    ) -> None:
        """
        Mark a subtree as complete and update internal state.

        This is a convenience method that wraps mark_splits_as_processed.

        Args:
            subtree: The subtree that was processed
            collapse_path: The collapse path that was executed
            expand_path: The expand path that was executed
        """
        self.mark_splits_as_processed(
            subtree=subtree,
            processed_collapse_splits=collapse_path,
            processed_expand_splits=expand_path,
        )

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

        Tie-breaker: lexicographic ordering of indices
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
            return self._select_by_shared_collapse_priority(unprocessed)

        # Delegate to path group manager for topological ordering
        if self._path_group_manager and self._path_group_manager.enabled:
            next_subtree = self._path_group_manager.get_next_subtree(
                self.processed_subtrees
            )
            if next_subtree is not None and next_subtree in unprocessed:
                return next_subtree

        # Fallback: choose smallest expand path (fewest expand splits first)
        # This allows shared splits to be deferred to smaller subtrees
        return self._select_by_smallest_expand_path(unprocessed)

    def _select_by_shared_collapse_priority(
        self, unprocessed: Set[Partition]
    ) -> Partition:
        """
        Select subtree using shared collapse priority system.

        Priority: shared collapse > unique-only > shared expand
        """
        candidates = []
        for subtree in unprocessed:
            shared_collapse = self.collapse_tracker.get_shared_resources(subtree)
            # Use bitmask for deterministic tie-breaking
            tie_breaker = subtree.bitmask

            if shared_collapse:
                priority = (0, -len(shared_collapse), tie_breaker)
            elif self.expand_tracker.get_shared_resources(subtree):
                shared_expand = self.expand_tracker.get_shared_resources(subtree)
                priority = (2, -len(shared_expand), tie_breaker)
            else:
                priority = (1, 0, tie_breaker)

            candidates.append((priority, subtree))

        return min(candidates)[1]

    def _select_by_smallest_expand_path(self, unprocessed: Set[Partition]) -> Partition:
        """
        Select subtree with SMALLEST expand splits first.

        This ensures the expand-last strategy works correctly for shared splits:
        - Subtrees with FEWER/SMALLER expand splits are processed FIRST
        - They see the split is shared (because larger subtrees also own it)
        - They defer/release their claims
        - Subtrees with MORE/LARGER expand splits are processed LAST
        - They become the "last owner" and apply the shared splits

        Tie-breaker: total expand path size, then lexicographic ordering.
        """
        candidates = []
        for sub in unprocessed:
            # We want SMALLEST first.
            shared_expand_count = len(self.expand_tracker.get_shared_resources(sub))
            total_expand_count = len(self.expand_tracker.get_resources(sub))
            # Use bitmask for deterministic tie-breaking
            tie_breaker = sub.bitmask

            # Primary: shared expand count (smallest first)
            # Secondary: total expand count (smallest first)
            # Tertiary: bitmask tie-breaker
            candidates.append(
                (shared_expand_count, total_expand_count, tie_breaker, sub)
            )

        return min(candidates)[3]

    # ============================================================================
    # 6. Compatibility/Incompatibility Logic
    # ============================================================================

    def get_tabula_rasa_collapse_splits(self) -> PartitionSet[Partition]:
        """
        DEPRECATED: Get ALL collapse splits for tabula rasa (clean slate) strategy.

        NOTE: This method is no longer used by _compute_collapse_path().
        The stepwise strategy replaces Tabula Rasa for smoother animations.
        This method is kept for backwards compatibility and the backup file
        pivot_split_registry_tabula_rasa.py.

        The first subtree collapses EVERYTHING from the source tree to create
        a blank canvas. Then we rebuild the tree from scratch by expanding splits
        step by step. This ensures no incorrect tree structure carries over.

        Strategy: First subtree gets ALL collapse splits, regardless of which
        subtree they were originally assigned to. Subsequent subtrees get none
        (they only handle their assigned splits).

        Note: This method is idempotent - calling it multiple times after the
        first call returns empty. The first call that returns splits also marks
        the first subtree as processed.

        Returns:
            ALL collapse splits if first subtree not yet processed, empty otherwise.
        """
        if not self.first_subtree_processed:
            self.first_subtree_processed = True
            # Return ALL collapse splits - complete tabula rasa
            return self.all_collapsible_splits.copy()
        return PartitionSet(encoding=self.encoding)

    def mark_first_subtree_processed(self) -> None:
        """
        Mark that the first subtree has been processed.

        After the first subtree collapses everything (tabula rasa), subsequent
        subtrees only handle their assigned splits.

        Note: This is now called automatically by get_tabula_rasa_collapse_splits(),
        but kept for explicit marking when tabula rasa returns empty (no collapses).
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


# ============================================================================
# Helper Functions for Path Building
# ============================================================================


def build_collapse_path(
    shared_splits: PartitionSet[Partition],
    unique_splits: PartitionSet[Partition],
    incompatible_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """Build the collapse path from component splits using set operations."""
    return shared_splits | unique_splits | incompatible_splits


def build_expand_path(
    shared_splits: PartitionSet[Partition],
    unique_splits: PartitionSet[Partition],
    contingent_splits: PartitionSet[Partition],
) -> PartitionSet[Partition]:
    """Build the expand path from component splits using set operations."""
    return shared_splits | unique_splits | contingent_splits


def _gather_subtree_splits(
    state: PivotSplitRegistry, subtree: Partition
) -> Dict[str, PartitionSet[Partition]]:
    """Gathers all necessary split sets for a subtree from the state."""
    shared_collapse: PartitionSet[Partition] = (
        state.get_available_shared_collapse_splits(subtree)
    )
    unique_collapse: PartitionSet[Partition] = state.get_unique_collapse_splits(subtree)
    last_user_expand: PartitionSet[Partition] = state.get_expand_splits_for_last_user(
        subtree
    )
    unique_expand: PartitionSet[Partition] = state.get_unique_expand_splits(subtree)

    return {
        "shared_collapse": shared_collapse,
        "unique_collapse": unique_collapse,
        "last_user_expand": last_user_expand,
        "unique_expand": unique_expand,
        "contingent_expand": PartitionSet(encoding=state.encoding),
    }


def _finalize_and_store_plan(
    plans: OrderedDict[Partition, Dict[str, Any]],
    state: PivotSplitRegistry,
    subtree: Partition,
    collapse_path: PartitionSet[Partition],
    expand_path: PartitionSet[Partition],
) -> tuple[PartitionSet[Partition], PartitionSet[Partition]]:
    """Handles last subtree logic, sorts paths, stores the plan, and returns final paths."""
    if state.is_last_subtree(subtree):
        expand_path |= state.get_all_remaining_expand_splits()
        collapse_path |= state.get_all_remaining_collapse_splits()

    # Deterministic path ordering (larger partitions first, tie-break by bitmask)
    # Sort collapse paths Smallest First (Leaves Inward)
    collapse_path_list = sorted(
        collapse_path, key=lambda p: (len(p.indices), p.bitmask)
    )
    # Sort expand paths Largest First (Root Outward)
    expand_path_list = sorted(expand_path, key=lambda p: (-len(p.indices), p.bitmask))

    # Store the full paths in the plan
    plans[subtree] = {
        "collapse": {"path_segment": collapse_path_list},
        "expand": {"path_segment": expand_path_list},
    }
    return collapse_path, expand_path


def _update_state(
    state: PivotSplitRegistry,
    subtree: Partition,
    collapse_path: PartitionSet[Partition],
    expand_path: PartitionSet[Partition],
) -> None:
    """Marks splits and the subtree as processed in the state.

    Args:
        state: The interpolation state to update
        subtree: The subtree being processed
        collapse_path: The ACTUAL collapse path that will be executed (may be ALL splits for TABULA RASA)
        expand_path: The ACTUAL expand path that will be executed (includes final cleanup splits)
    """
    # Use the actual collapse_path that will be executed (covers TABULA RASA first subtree)
    processed_collapse = collapse_path
    processed_expand = expand_path

    # Mark splits as processed in the state - this will remove shared splits from all subtrees
    # Note: This also marks the subtree as processed to prevent reprocessing
    state.mark_splits_as_processed(
        subtree=subtree,
        processed_collapse_splits=processed_collapse,
        processed_expand_splits=processed_expand,
    )


def build_edge_plan(
    expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    collapse_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
    collapse_tree: Node,
    expand_tree: Node,
    current_pivot_edge: Partition,
) -> OrderedDict[Partition, Dict[str, Any]]:
    """Build execution plan for a pivot edge by assigning splits to subtrees.

    Note: expand_splits_by_subtree contains PATH-based assignments (splits on the path
    between subtree and pivot). This may not cover ALL splits in the pivot edge subtree,
    so we ensure completeness below.
    """
    plans: OrderedDict[Partition, Dict[str, Any]] = OrderedDict()

    # Get splits within the active changing edge scope only
    all_collapse_splits, all_expand_splits = (
        get_unique_splits_for_current_pivot_edge_subtree(
            collapse_tree,
            expand_tree,
            current_pivot_edge,
        )
    )

    # COMPLETENESS GUARANTEE: Path-based assignments may miss splits not on any path
    # (e.g., contingent splits from jumping taxa, cross-branch splits). Assign any
    # unassigned expands to the LAST subtree as fallback. This aligns better with
    # the "Expand Last" strategy, preventing premature creation by the first subtree.
    claimed_expands = PartitionSet(
        set().union(*expand_splits_by_subtree.values())
        if expand_splits_by_subtree
        else set(),
        encoding=all_expand_splits.encoding,
    )

    unassigned_expands = all_expand_splits - claimed_expands

    if unassigned_expands:
        # Assign to the LAST subtree instead of the first (deterministic ordering).
        target_subtree = (
            max(expand_splits_by_subtree.keys(), key=lambda p: p.bitmask)
            if expand_splits_by_subtree
            else current_pivot_edge
        )
        if target_subtree not in expand_splits_by_subtree:
            expand_splits_by_subtree[target_subtree] = PartitionSet(
                encoding=all_expand_splits.encoding
            )
        # Reassign with a new PartitionSet to avoid in-place quirks
        expand_splits_by_subtree[target_subtree] = (
            expand_splits_by_subtree[target_subtree] | unassigned_expands
        )
        logger.debug(
            "[builder] pivot=%s assigning %d unclaimed expands to target_subtree=%s",
            current_pivot_edge.bipartition(),
            len(unassigned_expands),
            target_subtree.bipartition(),
        )

    # Initialize state management for proper shared splits handling
    state = PivotSplitRegistry(
        all_collapse_splits,
        all_expand_splits,
        collapse_splits_by_subtree,
        expand_splits_by_subtree,
        current_pivot_edge,
    )

    logger.debug(
        "[builder] pivot=%s all_expand_splits=%s expand_paths_by_subtree=%s",
        current_pivot_edge.bipartition(),
        [list(p.indices) for p in all_expand_splits],
        {
            st.bipartition(): [list(p.indices) for p in splits]
            for st, splits in expand_splits_by_subtree.items()
        },
    )

    while state.has_remaining_work():
        # Get next subtree using priority algorithm
        subtree: Partition | None = state.get_next_subtree()

        if subtree is None:
            break

        # 1. Gather all component splits for the current subtree
        splits = _gather_subtree_splits(state, subtree)

        # ========================================================================
        # 2. TABULA RASA STRATEGY
        # ========================================================================
        # First subtree may collapse everything (tabula rasa) only if there are
        # actual collapses at this pivot edge. Otherwise, respect per-subtree
        # assignments even for the first subtree.
        is_first_subtree = not state.first_subtree_processed

        collapse_path: PartitionSet[Partition]
        incompatible: PartitionSet[Partition] = PartitionSet(encoding=state.encoding)
        if is_first_subtree:
            all_collapse_splits = state.get_tabula_rasa_collapse_splits()
            if len(all_collapse_splits) > 0:
                collapse_path = all_collapse_splits
            else:
                # Compute incompatibilities for this subtree's planned expands
                # NOTE: Don't include contingent_expand here - they haven't been consumed yet!
                prospective_expand = (
                    splits["last_user_expand"] | splits["unique_expand"]
                )
                incompatible = find_incompatible_splits(
                    prospective_expand, state.all_collapsible_splits
                )

                collapse_path = build_collapse_path(
                    splits["shared_collapse"],
                    splits["unique_collapse"],
                    incompatible,
                )
        else:
            # Subsequent subtrees: only their assigned splits
            # Compute incompatibilities for this subtree's planned expands
            # NOTE: Don't include contingent_expand here - they haven't been consumed yet!
            prospective_expand = splits["last_user_expand"] | splits["unique_expand"]
            incompatible = find_incompatible_splits(
                prospective_expand, state.all_collapsible_splits
            )

            collapse_path = build_collapse_path(
                splits["shared_collapse"],
                splits["unique_collapse"],
                incompatible,
            )

        # After determining the actual collapse path (including tabula rasa or
        # incompatibility collapses), consume contingent splits that fit within
        # ANY collapsed region.
        extra_contingent = state.consume_contingent_expand_splits_for_subtree(
            subtree=subtree, collapsed_splits=collapse_path
        )

        splits["contingent_expand"] |= extra_contingent

        # Build expand path (all subtrees get their expand work)
        expand_path: PartitionSet[Partition] = build_expand_path(
            splits["last_user_expand"],
            splits["unique_expand"],
            splits["contingent_expand"],
        )

        # Mark first subtree as processed
        if is_first_subtree:
            state.mark_first_subtree_processed()

        # Finalize the plan for this subtree
        collapse_path, expand_path = _finalize_and_store_plan(
            plans, state, subtree, collapse_path, expand_path
        )

        # Update the global state - pass the actual collapse_path for TABULA RASA handling
        _update_state(
            state,
            subtree,
            collapse_path,
            expand_path,
        )

    return plans
