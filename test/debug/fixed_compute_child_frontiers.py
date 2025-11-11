"""
Fixed version of compute_child_frontiers from build_pivot_lattices.py

MATHEMATICAL FRAMEWORK:
This module solves a "complete cover problem" where frontier splits (shared clades)
must be covered by bottom splits (unique topology structures). Some shared clades
aren't nested within any unique structure, so they become "self-covering" bottoms.

PHYLOGENETIC CONTEXT:
- Frontier splits = maximal shared clades between two tree topologies
- Bottom splits = unique clades from one tree's topology (children_to_process)
- Coverage = shared clade is subset of (nested within) a unique clade
- Self-covering = shared clade that needs its own bottom entry for reconciliation

CHANGE SUMMARY:
After processing explicit bottoms from topology differences, we find frontier splits
not covered by any bottom and add them as their own bottom entries. This ensures they
appear in the conflict matrix for the lattice-based jumping taxa algorithm.
"""

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.types import TopToBottom
from brancharchitect.jumping_taxa.debug import jt_logger


# ============================================================================
# Helper Functions: Coverage Analysis for Frontier/Bottom Relationships
# ============================================================================


def _is_frontier_covered_by_bottoms(
    frontier_split: Partition, bottom_splits: PartitionSet[Partition]
) -> bool:
    """
    Check if a frontier split (shared clade) is covered by any bottom split.

    MATHEMATICAL: Tests if ∃b ∈ bottoms: frontier ⊆ b
    PHYLOGENETIC: Checks if shared clade is nested within any unique clade structure

    Args:
        frontier_split: A maximal shared split between tree topologies
        bottom_splits: Unique splits from one tree (from children_to_process)

    Returns:
        True if frontier is a subset of at least one bottom (covered)
        False if frontier needs its own bottom entry (self-covering)
    """
    return any(
        (frontier_split.bitmask & bottom.bitmask) == frontier_split.bitmask
        for bottom in bottom_splits
    )


def _find_uncovered_frontiers(
    frontier_splits: PartitionSet[Partition], bottom_splits: PartitionSet[Partition]
) -> PartitionSet[Partition]:
    """
    Find frontier splits (shared clades) not covered by any bottom split.

    MATHEMATICAL: Returns {f ∈ frontiers | ¬∃b ∈ bottoms: f ⊆ b}
    PHYLOGENETIC: Identifies shared clades not nested in unique topology structures

    These uncovered frontiers need their own bottom entries to appear in the
    reconciliation lattice and conflict matrix.

    Args:
        frontier_splits: Maximal shared splits between trees
        bottom_splits: Unique splits from one tree (children_to_process)

    Returns:
        Subset of frontier_splits that are self-covering (need own bottom entry)

    Example:
        In basic_1_taxon_partial test:
        - frontiers = {(X), (A1), (A2), (A3,A4)} on LEFT
        - bottoms = {(A1,X)} from children_to_process
        - (A2) and (A3,A4) are uncovered → need own bottom entries
    """
    uncovered: PartitionSet[Partition] = PartitionSet(encoding=frontier_splits.encoding)

    for frontier in frontier_splits:
        if not _is_frontier_covered_by_bottoms(frontier, bottom_splits):
            uncovered.add(frontier)

    return uncovered


def _add_bottom_to_frontiers_entry(
    child_frontiers: dict[Partition, TopToBottom],
    unique_maximal_split: Partition,
    all_frontier_splits: PartitionSet[Partition],
    bottom_split: Partition,
    bottom_node: Node,
) -> None:
    """
    Add a bottom→frontiers mapping entry for lattice construction.

    MATHEMATICAL: Creates mapping b → {f ∈ frontiers | f ⊆ b}
    PHYLOGENETIC: Links unique clade to all shared clades it contains

    This mapping is used to build the conflict matrix for finding minimal
    reconciliation solutions (jumping taxa).

    Args:
        child_frontiers: Result dictionary being built
        unique_maximal_split: Key (maximal unique split from this child)
        all_frontier_splits: All maximal shared splits (for initialization)
        bottom_split: The bottom split to map from (unique or self-covering)
        bottom_node: Tree node representing this split
    """
    # Ensure entry exists for this child's unique split
    child_frontiers.setdefault(
        unique_maximal_split,
        TopToBottom(
            shared_top_splits=all_frontier_splits,
            bottom_to_frontiers={},
        ),
    )

    # Find which frontiers this bottom covers (shared clades nested within)
    covered_frontiers = all_frontier_splits.bottoms_under(
        bottom_split
    ).maximal_elements()

    # Add the bottom→frontiers mapping
    child_frontiers[unique_maximal_split].bottom_to_frontiers[
        bottom_node.split_indices
    ] = covered_frontiers


def compute_child_frontiers(
    parent: Node,
    children_to_process: PartitionSet[Partition],
    shared_splits: PartitionSet[Partition],
) -> dict[Partition, TopToBottom]:
    """
    Compute per-child frontiers (maximal shared elements) under the given parent node.

    For each child identified by a split in children_to_process, computes:
        For each child: maximal(child.all_splits ∩ shared_splits) via maximal_elements().

    where maximal_elements() extracts the frontier (no element is subset of another).

    Any maximal shared splits not covered by children are appended as a separate
    cover element (uncovered shared splits).

    Args:
        parent: Parent node under which children are found
        children_to_process: Splits identifying which children to analyze
                            (typically from minimum_cover of unique splits)
        shared_splits: Common splits between two trees (restricted universe)

    Returns:
        Dictionary mapping each maximal child split to its TopToBottom structure,
        containing shared top splits and their corresponding bottom splits.
        Sorted deterministically by minimum bitmask for reproducibility.

    Raises:
        ValueError: If a child split cannot be found under parent
    """
    # Handle empty case
    if not children_to_process:
        return {}

    # Dictionary to store TopToBottom structures for each maximal child split
    child_frontiers: dict[Partition, TopToBottom] = {}

    for unique_maximal_child_splits in children_to_process.maximal_elements():
        top_child: Node | None = parent.find_node_by_split(unique_maximal_child_splits)

        if top_child is None:
            raise ValueError(
                f"Child split {unique_maximal_child_splits} not found under parent subtree "
                f"{tuple(sorted(parent.get_current_order()))}"
            )

        # Get shared splits in this child's subtree
        child_all_splits: PartitionSet[Partition] = top_child.to_splits(
            with_leaves=True
        )

        # Across-trees shared splits within this child's subtree
        child_splits_across_trees: PartitionSet[Partition] = (
            child_all_splits & shared_splits
        )
        # Per-child frontier (maximal shared elements)
        child_frontier_splits: PartitionSet[Partition] = (
            child_splits_across_trees.maximal_elements()
        )

        not_shared_applied_splits = child_frontier_splits

        jt_logger.info("---------------------------------------")
        jt_logger.info(f"Processing child split {unique_maximal_child_splits}")
        jt_logger.info(f"Child Splits Across Trees: {child_splits_across_trees}")
        jt_logger.info(f"Child Frontier Splits: {not_shared_applied_splits}")

        # Get candidate bottoms from tree topology differences (min_size=2)
        candidate_bottoms: PartitionSet[Partition] = children_to_process.bottoms_under(
            unique_maximal_child_splits, min_size=2
        )
        candidate_bottoms = candidate_bottoms.maximal_elements()

        # ====================================================================
        # PHASE A: Process explicit bottoms from topology differences
        # ====================================================================
        # These are unique clades from one tree that contain shared clades

        for bottom in candidate_bottoms:
            bottom_node: Node | None = parent.find_node_by_split(bottom)

            if not bottom_node:
                raise ValueError(
                    f"Bottom split {bottom} not found under parent subtree "
                    f"{tuple(sorted(parent.get_current_order()))}"
                )

            # Add this bottom and the frontiers it covers
            _add_bottom_to_frontiers_entry(
                child_frontiers=child_frontiers,
                unique_maximal_split=unique_maximal_child_splits,
                all_frontier_splits=child_frontier_splits,
                bottom_split=bottom,
                bottom_node=bottom_node,
            )

        # ====================================================================
        # PHASE B: Add self-covering frontiers (shared clades not in unique structure)
        # ====================================================================
        # Find frontier splits not covered by any candidate bottom.
        # These are shared clades that need their own bottom entry because
        # they aren't nested within any unique topology structure.

        uncovered_frontiers = _find_uncovered_frontiers(
            child_frontier_splits, candidate_bottoms
        )

        for frontier_split in uncovered_frontiers:
            frontier_node = parent.find_node_by_split(frontier_split)

            if frontier_node:
                jt_logger.info(
                    f"  Self-covering frontier {frontier_split}: "
                    f"shared clade not nested in unique topology structure"
                )

                # Add this frontier as its own bottom (self-covering)
                _add_bottom_to_frontiers_entry(
                    child_frontiers=child_frontiers,
                    unique_maximal_split=unique_maximal_child_splits,
                    all_frontier_splits=child_frontier_splits,
                    bottom_split=frontier_split,  # bottom = frontier (self-covering)
                    bottom_node=frontier_node,
                )

    return child_frontiers


# ============================================================================
# Implementation Summary
# ============================================================================
#
# REFACTORED STRUCTURE:
#
# 1. Helper Functions (lines 27-120):
#    - _is_frontier_covered_by_bottoms(): Check if shared clade is nested in unique clade
#    - _find_uncovered_frontiers(): Find shared clades needing own bottom entries
#    - _add_bottom_to_frontiers_entry(): Build bottom→frontiers mappings
#
# 2. Main Algorithm (lines 153-253):
#    PHASE A: Process topology differences
#      - Iterate candidate_bottoms from children_to_process
#      - Link each unique clade to shared clades it contains
#
#    PHASE B: Add self-covering frontiers
#      - Find frontiers not covered by any candidate bottom
#      - Add them as their own bottom entries
#
# MATHEMATICAL FRAMEWORK:
# - Solves complete cover problem: every frontier must be covered by some bottom
# - Complete bottoms = candidate_bottoms ∪ uncovered_frontiers
# - Ensures: ∀f ∈ frontiers, ∃b ∈ complete_bottoms: f ⊆ b
#
# PHYLOGENETIC INTERPRETATION:
# - Frontiers = maximal shared clades between tree topologies
# - Candidate bottoms = unique clades from topology differences (min_size ≥ 2)
# - Uncovered frontiers = shared clades not nested in any unique structure
# - Self-covering = frontier that serves as its own bottom
#
# TEST CASE EXAMPLE (basic_1_taxon_partial):
# Trees: T1 = ((((X,A1),A2),(A3,A4)),((B1,B2),(B3,B4)))
#        T2 = (X,(((A1,A2),(A3,A4)),((B1,B2),(B3,B4))))
#
# LEFT side (T1):
#   - frontiers = {(X), (A1), (A2), (A3,A4)}
#   - candidate_bottoms = {(A1,X)} (from children_to_process)
#   - uncovered = {(A2), (A3,A4)} → added as own bottoms
#
# RIGHT side (T2):
#   - frontiers = {(A1), (A2), (A3,A4), (B1,B2,B3,B4)}
#   - candidate_bottoms = {(A1,A2)} (from children_to_process)
#   - uncovered = {(A3,A4), (B1,B2,B3,B4)} → added as own bottoms
#
# RESULT: All frontiers appear in bottom_to_frontiers, ensuring they're included
#         in the conflict matrix for finding minimal jumping taxa solutions.
