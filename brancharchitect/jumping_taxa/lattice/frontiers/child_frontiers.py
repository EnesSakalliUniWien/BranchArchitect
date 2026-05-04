"""
Compute child frontier structures for lattice construction.

Mathematical context:
- Frontier splits = maximal shared clades between two tree topologies
- Bottom splits = unique clades from one tree's topology
- Coverage = shared clade ⊆ unique clade (nested within)
- Self-covering = shared direct child not covered by unique splits
"""

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.types.child_frontiers import ChildFrontiers


# ============================================================================
# Helper Functions: Building Bottom-to-Frontiers Mappings
# ============================================================================


def _add_bottom_to_frontiers_entry(
    child_frontiers: dict[Partition, ChildFrontiers],
    unique_maximal_split: Partition,
    all_frontier_splits: PartitionSet[Partition],
    bottom_split: Partition,
    bottom_node: Node,
) -> None:
    """
    Add a bottom→frontiers mapping: b → {f ∈ frontiers | f ⊆ b}.

    Links unique clade to all shared clades it contains for conflict matrix construction.
    """
    child_frontiers.setdefault(
        unique_maximal_split,
        ChildFrontiers(
            shared_top_splits=all_frontier_splits,
            bottom_partition_map={},
        ),
    )

    # Find which frontiers this bottom covers (shared clades nested within)
    covered_frontiers = all_frontier_splits.maximals_under(bottom_split)

    # Add the bottom→frontiers mapping
    child_frontiers[unique_maximal_split].bottom_partition_map[
        bottom_node.split_indices
    ] = covered_frontiers


def compute_child_frontiers(
    parent: Node,
    children_to_process: PartitionSet[Partition],
    shared_splits: PartitionSet[Partition],
) -> dict[Partition, ChildFrontiers]:
    """
    Compute per-child frontiers (maximal shared elements) under the given parent.

    Handles two scenarios:
    1. Shared direct children not covered by unique splits → self-covering
    2. Unique children with topology differences → bottom-to-frontier mappings

    For each unique child: maximal(child.splits ∩ shared_splits)

    Args:
        parent: Parent node under which children are found
        children_to_process: Splits identifying children to analyze
        shared_splits: Common splits between two trees

    Returns:
        Dict mapping each maximal child split to its ChildFrontiers structure.

    Raises:
        ValueError: If a child split cannot be found under parent
    """
    # Handle empty case: Create a ChildFrontiers entry for each shared split
    if not children_to_process:
        child_frontiers: dict[Partition, ChildFrontiers] = {}

        for partition in shared_splits:
            # Create singleton PartitionSet for this partition
            singleton_set: PartitionSet[Partition] = PartitionSet(
                splits={partition},
                encoding=shared_splits.encoding,
                name=f"{shared_splits.name}_singleton",
            )
            # Each partition gets its own ChildFrontiers entry
            child_frontiers[partition] = ChildFrontiers(
                shared_top_splits=singleton_set,
                bottom_partition_map={partition: singleton_set},
            )

        return child_frontiers

    # Dictionary to store ChildFrontiers structures for each maximal child split
    child_frontiers: dict[Partition, ChildFrontiers] = {}

    # ========================================================================
    # Add shared direct children of pivot as separate ChildFrontiers entries
    # ========================================================================
    # These are children of the parent node whose splits are in shared_splits
    # but are NOT covered by any split in children_to_process (unique splits).
    # They need their own ChildFrontiers entry with self as frontier and bottom.

    # Enforce deterministic iteration order for children
    # Sort children by their split bitmask to ensure consistent frontier headers
    sorted_children = sorted(
        parent.children, key=lambda c: c.split_indices.bitmask if c.split_indices else 0
    )

    for child in sorted_children:
        if child.split_indices in shared_splits:
            # Check if this shared child is covered by any unique child
            # MATHEMATICAL: Test if ∃u ∈ children_to_process: child.split ⊆ u
            if not children_to_process.covers(child.split_indices):
                # This shared direct child is not nested under any unique child
                # Add it as its own ChildFrontiers entry (self-covering)
                frontier_set: PartitionSet[Partition] = PartitionSet(
                    encoding=shared_splits.encoding
                )
                frontier_set.add(child.split_indices)

                child_frontiers[child.split_indices] = ChildFrontiers(
                    shared_top_splits=frontier_set,
                    bottom_partition_map={child.split_indices: frontier_set},
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

        # Per-child frontier (maximal shared elements nested under this unique child)
        # effectively: maximals(shared_splits ∩ subtree(child))
        child_frontier_splits: PartitionSet[Partition] = shared_splits.maximals_under(
            unique_maximal_child_splits
        )

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
