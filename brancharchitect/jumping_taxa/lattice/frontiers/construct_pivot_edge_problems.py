from __future__ import annotations
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
    get_child_splits,
)
from brancharchitect.jumping_taxa.exceptions import LatticeConstructionError
from typing import (
    List,
    Tuple,
)
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.types.child_frontiers import ChildFrontiers
from brancharchitect.jumping_taxa.lattice.frontiers.child_frontiers import (
    compute_child_frontiers,
)
from brancharchitect.jumping_taxa.lattice.logging_helpers import (
    log_lattice_construction_start,
    log_pivot_processing,
    log_lattice_edge_details,
)
from brancharchitect.logger import jt_logger


"""
Pivot Lattice Construction: builds the lattice of reticulation events between
two phylogenetic trees via split/frontier/partition analysis.

Naming conventions:
  - pivot_split:         focal shared split being analyzed
  - across_trees:        intersection between T₁ and T₂
  - under_pivot:         restricted to clade defined by pivot_split
  - child_subtree:       computed from child nodes' subtrees
  - unique_subtree:      splits unique to one tree
  - child_frontiers:     per-child frontier structures
  - frontier:            maximal elements (antichain) under subset order
"""


# ============================================================================
# Lattice Construction API (Entry Point)
# ============================================================================
def construct_pivot_edge_problems(t1: Node, t2: Node) -> List[PivotEdgeSubproblem]:
    """Compute detailed split information for two trees (per pivot/frontiers)."""
    # Ensure both trees have their indices built

    # Validate encoding consistency between trees
    if t1.taxa_encoding != t2.taxa_encoding:
        raise ValueError("Tree encodings must be identical for lattice construction")

    t1_splits: PartitionSet[Partition] = t1.to_splits()  # fresh splits
    t2_splits: PartitionSet[Partition] = t2.to_splits()  # fresh splits

    # Get common splits and verify they exist in both trees
    intersection: PartitionSet[Partition] = t1_splits.intersection(t2_splits)

    if not jt_logger.disabled:
        log_lattice_construction_start(t1, t2, intersection)
    pivot_edge_subproblems: List[PivotEdgeSubproblem] = []

    # Sort splits deterministically by size (approx. topological) then bitmask
    sorted_common_splits = sorted(intersection, key=lambda s: (len(s), s.bitmask))

    for pivot_split in sorted_common_splits:
        subproblem = construct_pivot_edge_problem(pivot_split, t1, t2)
        if subproblem is not None:
            pivot_edge_subproblems.append(subproblem)

    return pivot_edge_subproblems


def construct_pivot_edge_problem(
    pivot_split: Partition,
    t1: Node,
    t2: Node,
) -> PivotEdgeSubproblem | None:
    """Process a single pivot split to create a Lattice Map edge subproblem."""

    # Skip atomic (singleton) splits at this stage
    if pivot_split.is_singleton:
        return None

    t1_node: Node | None = t1.find_node_by_split(pivot_split)
    t2_node: Node | None = t2.find_node_by_split(pivot_split)

    # Validate that both trees contain the pivot split
    _validate_nodes_exist(pivot_split, t1_node, t2_node)

    # Type narrowing: after validation, nodes are guaranteed to be non-None
    assert t1_node is not None
    assert t2_node is not None

    is_pivot, child_subtree_splits_across_trees = is_pivot_edge(t1_node, t2_node)

    # Process further if there are child splits unique to either tree.
    if is_pivot:
        # Across-trees shared set under pivot (include leaves)
        across_trees_splits_under_pivot_with_leaves: PartitionSet[Partition] = (
            t1_node.to_splits(with_leaves=True) & t2_node.to_splits(with_leaves=True)
        )

        across_trees_splits_under_pivot_with_leaves.discard(pivot_split)

        if not jt_logger.disabled:
            log_pivot_processing(pivot_split)

        # Compute unique splits for each tree (topological differences)
        # Reuse cached splits from nodes (to_splits() caches results)
        t1_node_splits = t1_node.to_splits()
        t2_node_splits = t2_node.to_splits()
        t1_unique_subtree_splits = t1_node_splits - t2_node_splits
        t2_unique_subtree_splits = t2_node_splits - t1_node_splits

        # Compute per-child frontiers (maximal shared elements) for relevant children
        t1_child_frontiers: dict[Partition, ChildFrontiers] = compute_child_frontiers(
            t1_node,
            t1_unique_subtree_splits,
            across_trees_splits_under_pivot_with_leaves,
        )

        t2_child_frontiers: dict[Partition, ChildFrontiers] = compute_child_frontiers(
            t2_node,
            t2_unique_subtree_splits,
            across_trees_splits_under_pivot_with_leaves,
        )

        edge = PivotEdgeSubproblem(
            pivot_split=pivot_split,
            tree1_child_frontiers=t1_child_frontiers,
            tree2_child_frontiers=t2_child_frontiers,
            child_subtree_splits_across_trees=child_subtree_splits_across_trees,
            tree1_node=t1_node,
            tree2_node=t2_node,
            encoding=t1.taxa_encoding,
        )

        if not jt_logger.disabled:
            log_lattice_edge_details(edge)

        return edge

    return None


def is_pivot_edge(t1_node: Node, t2_node: Node) -> Tuple[bool, PartitionSet[Partition]]:
    """
    Analyze if the split defined by t1_node/t2_node is a valid pivot edge.

    A pivot edge is valid if there are topological differences (unique child splits)
    in the subtrees below it.

    Returns:
        tuple containing:
        - bool: True if this is a pivot edge (has unique child splits)
        - PartitionSet: The shared child subtree splits (intersection)
    """
    t1_child_subtree_splits: PartitionSet[Partition] = get_child_splits(t1_node)
    t2_child_subtree_splits: PartitionSet[Partition] = get_child_splits(t2_node)

    child_subtree_splits_across_trees: PartitionSet[Partition] = (
        t1_child_subtree_splits & t2_child_subtree_splits
    )

    has_unique_child_splits: bool = (
        t1_child_subtree_splits != child_subtree_splits_across_trees
        or child_subtree_splits_across_trees != t2_child_subtree_splits
    )

    return has_unique_child_splits, child_subtree_splits_across_trees


###############################################################################
# Utility Functions
###############################################################################
# Note: compute_child_frontiers has been moved to child_frontiers.py
# The original implementation is preserved in child_frontiers_original.py


def _validate_nodes_exist(
    pivot_split: Partition, t1_node: Node | None, t2_node: Node | None
) -> None:
    """
    Validate both trees contain nodes for a shared pivot split.

    For S ∈ (T₁ ∩ T₂), nodes n₁ ∈ T₁ and n₂ ∈ T₂ must exist where split(n) = S.
    Missing nodes indicate corrupted tree structure or indexing errors.

    Raises:
        LatticeConstructionError: If either node is None
    """
    if t1_node is None or t2_node is None:
        LatticeConstructionError.raise_missing_node(
            split=pivot_split,
            is_missing_in_tree1=t1_node is None,
            is_missing_in_tree2=t2_node is None,
        )
