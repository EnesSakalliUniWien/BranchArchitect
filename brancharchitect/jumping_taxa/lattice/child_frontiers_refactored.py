"""
Refactored compute_child_frontiers with improved modularity.

MATHEMATICAL FRAMEWORK:
This module computes frontier splits (shared clades) and their relationships with
bottom splits (unique topology structures). Shared direct children of the pivot that
are not covered by unique splits become self-covering bottoms.

PHYLOGENETIC CONTEXT:
- Frontier splits = maximal shared clades between two tree topologies
- Bottom splits = unique clades from one tree's topology (children_to_process)
- Coverage = shared clade is subset of (nested within) a unique clade
- Self-covering = shared direct child that needs its own bottom entry

IMPLEMENTATION:
The algorithm handles two types of children under the pivot:
1. Shared direct children not covered by unique splits → self-covering entries
2. Unique children with their topology differences → standard bottom-to-frontier mappings

This is a refactored version of compute_child_frontiers from build_pivot_lattices.py
with improved modularity and mathematical clarity.
"""

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.types import TopToBottom
from brancharchitect.jumping_taxa.debug import jt_logger


# ============================================================================
# Helper Functions: Building Bottom-to-Frontiers Mappings
# ============================================================================


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

    This refactored version handles two scenarios:
    1. Shared direct children of pivot not covered by unique splits
    2. Unique children with topology differences and their frontier relationships

    For each unique child identified by a split in children_to_process, computes:
        maximal(child.all_splits ∩ shared_splits) via maximal_elements()

    where maximal_elements() extracts the frontier (no element is subset of another).

    Args:
        parent: Parent node under which children are found
        children_to_process: Splits identifying which children to analyze
                            (typically from minimum_cover of unique splits)
        shared_splits: Common splits between two trees (restricted universe)

    Returns:
        Dictionary mapping each maximal child split to its TopToBottom structure,
        containing shared top splits and their corresponding bottom splits.

    Raises:
        ValueError: If a child split cannot be found under parent
    """
    # Handle empty case: Create a TopToBottom entry for each shared split
    if not children_to_process:
        child_frontiers: dict[Partition, TopToBottom] = {}

        for partition in shared_splits:
            # Create singleton PartitionSet for this partition
            singleton_set: PartitionSet[Partition] = PartitionSet(
                splits={partition},
                encoding=shared_splits.encoding,
                name=f"{shared_splits.name}_singleton",
            )
            # Each partition gets its own TopToBottom entry
            child_frontiers[partition] = TopToBottom(
                shared_top_splits=singleton_set,
                bottom_to_frontiers={partition: singleton_set},
            )

        return child_frontiers

    # Dictionary to store TopToBottom structures for each maximal child split
    child_frontiers: dict[Partition, TopToBottom] = {}

    # ========================================================================
    # Add shared direct children of pivot as separate TopToBottom entries
    # ========================================================================
    # These are children of the parent node whose splits are in shared_splits
    # but are NOT covered by any split in children_to_process (unique splits).
    # They need their own TopToBottom entry with self as frontier and bottom.

    for child in parent.children:
        if child.split_indices in shared_splits:
            # Check if this shared child is covered by any unique child
            # MATHEMATICAL: Test if ∃u ∈ children_to_process: child.split ⊆ u
            if not children_to_process.covers(child.split_indices):
                # This shared direct child is not nested under any unique child
                # Add it as its own TopToBottom entry (self-covering)
                frontier_set: PartitionSet[Partition] = PartitionSet(
                    encoding=shared_splits.encoding
                )
                frontier_set.add(child.split_indices)

                child_frontiers[child.split_indices] = TopToBottom(
                    shared_top_splits=frontier_set,
                    bottom_to_frontiers={child.split_indices: frontier_set},
                )

                jt_logger.info("---------------------------------------")
                jt_logger.info(
                    f"Added shared direct pivot child: {child.split_indices}"
                )
                jt_logger.info(
                    f"  Self-covering: frontier = bottom = {child.split_indices}"
                )

    # ========================================================================
    # Process unique children (existing algorithm)
    # ========================================================================

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

        jt_logger.info("---------------------------------------")
        jt_logger.info(f"Processing child split {unique_maximal_child_splits}")
        jt_logger.info(f"Child Splits Across Trees: {child_splits_across_trees}")
        jt_logger.info(f"Child Frontier Splits: {child_frontier_splits}")

        # Get candidate bottoms from tree topology differences (min_size=2)
        candidate_bottoms: PartitionSet[Partition] = children_to_process.bottoms_under(
            unique_maximal_child_splits, min_size=2
        )

        # Process explicit bottoms from topology differences
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

    return child_frontiers
