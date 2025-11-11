"""
Original compute_child_frontiers implementation (backup).

This is the original implementation from build_pivot_lattices.py before
the refactoring that adds self-covering frontiers. Kept for reference
and comparison purposes.

NOTE: This version has a known issue where frontier splits not covered
by any bottom are omitted from the bottom_to_frontiers mapping, which
can cause incorrect jumping taxa solutions in some cases.

For the fixed version, see child_frontiers_refactored.py
"""

from __future__ import annotations
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.types import TopToBottom
from brancharchitect.jumping_taxa.debug import jt_logger


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

    Known Issue:
        Frontier splits that are not covered by any bottom (with min_size=2)
        will not appear in bottom_to_frontiers, causing them to be missing
        from the conflict matrix. This can lead to incorrect jumping taxa
        solutions. See child_frontiers_refactored.py for the fixed version.
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

        bottoms: PartitionSet[Partition] = children_to_process.bottoms_under(
            unique_maximal_child_splits, min_size=2
        )

        bottoms = bottoms.maximal_elements()

        for bottom in bottoms:
            bottom_node: Node | None = parent.find_node_by_split(bottom)
            shared_bottom_splits: PartitionSet[Partition] = (
                child_frontier_splits.bottoms_under(bottom)
            ).maximal_elements()

            if not bottom_node:
                raise ValueError(
                    f"Bottom split {bottom} not found under parent subtree "
                    f"{tuple(sorted(parent.get_current_order()))}"
                )

            # Create entry with setdefault if it doesn't exist
            child_frontiers.setdefault(
                unique_maximal_child_splits,
                TopToBottom(
                    shared_top_splits=child_frontier_splits,
                    bottom_to_frontiers={},
                ),
            )

            # Add bottom mapping
            child_frontiers[unique_maximal_child_splits].bottom_to_frontiers[
                bottom_node.split_indices
            ] = shared_bottom_splits

    return child_frontiers
