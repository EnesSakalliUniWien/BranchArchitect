from __future__ import annotations
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
    get_child_splits,
)
from brancharchitect.jumping_taxa.lattice.types import PMatrix
from brancharchitect.jumping_taxa.lattice.meet_product_solvers import (
    solution_size,
    matrix_row_size,
)
from brancharchitect.jumping_taxa.debug import (
    jt_logger,
)
from brancharchitect.jumping_taxa.exceptions import LatticeConstructionError
from typing import (
    Dict,
    List,
)
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.types import TopToBottom
from brancharchitect.jumping_taxa.lattice.child_frontiers_refactored import (
    compute_child_frontiers,
)
from brancharchitect.jumping_taxa.lattice.cover_relations import (
    collect_all_conflicts,
)


"""
Pivot Lattice Construction Utilities
------------------------------
This module provides the main logic and helpers for constructing the lattice of
reticulation events between two phylogenetic trees, including split/frontier/
partition analysis and edge depth propagation.

Naming conventions used in this module:
  - pivot_split:                        the focal shared split being analyzed
  - across_trees (suffix):              intersection between T1 and T2 (e.g., child_subtree_splits_across_trees)
  - under_pivot (suffix):               restricted to the clade defined by pivot_split (e.g., across_trees_splits_under_pivot_with_leaves)
  - child_subtree (prefix):             computed from child nodes' subtrees (e.g., t1_child_subtree_splits)
  - unique_subtree (suffix):            splits unique to one tree (e.g., t1_unique_subtree_splits)
  - child_frontiers (suffix):           per-child frontier structures (e.g., t1_child_frontiers, t2_child_frontiers)
  - frontier:                           maximal elements (antichain) of a set of splits under subset order
"""


# ============================================================================
# Lattice Construction API (Entry Point)
# ============================================================================
def construct_sublattices(t1: Node, t2: Node) -> List[PivotEdgeSubproblem]:
    """Compute detailed split information for two trees (per pivot/frontiers)."""
    # Ensure both trees have their indices built

    # Validate encoding consistency between trees
    if t1.taxa_encoding != t2.taxa_encoding:
        raise ValueError("Tree encodings must be identical for lattice construction")

    t1_splits: PartitionSet[Partition] = t1.to_splits()  # fresh splits
    t2_splits: PartitionSet[Partition] = t2.to_splits()  # fresh splits

    # Get common splits and verify they exist in both trees
    intersection: PartitionSet[Partition] = t1_splits.intersection(t2_splits)

    jt_logger.info(f"Found {len(intersection)} common splits to process")
    lattice_edges: List[PivotEdgeSubproblem] = []

    jt_logger.compare_tree_splits(tree1=t1, tree2=t2)

    # Sort splits deterministically by bitmask for reproducible processing
    sorted_intersection = sorted(intersection, key=lambda s: s.bitmask)
    for pivot_split in sorted_intersection:
        # Skip atomic (singleton) splits at this stage
        if pivot_split.is_singleton:
            continue

        t1_node: Node | None = t1.find_node_by_split(pivot_split)
        t2_node: Node | None = t2.find_node_by_split(pivot_split)

        # Validate that both trees contain the pivot split
        _validate_nodes_exist(pivot_split, t1_node, t2_node)

        # Type narrowing: after validation, nodes are guaranteed to be non-None
        assert t1_node is not None
        assert t2_node is not None

        # Across-trees shared set under pivot (include leaves)
        across_trees_splits_under_pivot_with_leaves: PartitionSet[Partition] = (
            t1_node.to_splits(with_leaves=True) & t2_node.to_splits(with_leaves=True)
        )

        t1_child_subtree_splits: PartitionSet[Partition] = get_child_splits(t1_node)
        t2_child_subtree_splits: PartitionSet[Partition] = get_child_splits(t2_node)

        child_subtree_splits_across_trees: PartitionSet[Partition] = (
            t1_child_subtree_splits & t2_child_subtree_splits
        )

        has_unique_child_splits: bool = (
            t1_child_subtree_splits != child_subtree_splits_across_trees
            or child_subtree_splits_across_trees != t2_child_subtree_splits
        )

        # Process further if there are child splits unique to either tree.
        if has_unique_child_splits:
            across_trees_splits_under_pivot_with_leaves.discard(pivot_split)

            jt_logger.info(
                f"Processing pivot_split {pivot_split.bipartition()} in both trees"
            )

            # Compute unique splits for each tree (topological differences)
            t1_unique_subtree_splits, t2_unique_subtree_splits = _compute_unique_splits(
                t1_node, t2_node
            )

            # Compute per-child frontiers (maximal shared elements) for relevant children
            t1_child_frontiers: dict[Partition, TopToBottom] = compute_child_frontiers(
                t1_node,
                t1_unique_subtree_splits,
                across_trees_splits_under_pivot_with_leaves,
            )

            t2_child_frontiers: dict[Partition, TopToBottom] = compute_child_frontiers(
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

            lattice_edges.append(edge)

            # Render helpful HTML tables for this edge (common covers, unique min covers, and atoms)
            # jt_logger is a combined Logger (includes TreeLogger methods)
            jt_logger.log_lattice_edge_tables(
                edge,
                show_common_covers=True,
                show_unique_min_covers=True,
                show_atoms=True,
                tablefmt="html",
            )

    return lattice_edges


###############################################################################
# Utility Functions
###############################################################################
# Note: compute_child_frontiers has been moved to child_frontiers_refactored.py
# The original implementation is preserved in child_frontiers_original.py


def _compute_unique_splits(
    tree1_node: Node, tree2_node: Node
) -> tuple[PartitionSet[Partition], PartitionSet[Partition]]:
    r"""
    Compute the splits unique to each tree under a shared pivot split.

    MATHEMATICAL DEFINITION:
        For two trees T₁ and T₂ with a shared pivot split S:

        Let S₁ = splits(T₁[S]) be all splits in T₁'s subtree rooted at S
        Let S₂ = splits(T₂[S]) be all splits in T₂'s subtree rooted at S

        Unique splits are computed via set difference:
        - unique₁ = S₁ \ S₂  (splits in T₁ but not in T₂)
        - unique₂ = S₂ \ S₁  (splits in T₂ but not in T₁)

    WHAT "UNIQUE" MEANS:
        A split is "unique to T₁" if it exists in T₁'s topology under the pivot
        but does NOT exist in T₂'s topology. These represent topological differences
        between the two trees within the same clade.

    PHYLOGENETIC INTERPRETATION:
        Unique splits identify where the two phylogenetic trees disagree in their
        branching structure. They represent:
        - Conflicting hypotheses about taxon groupings
        - Potential reticulation events or horizontal gene transfer

    EXAMPLE:
        Pivot split: {A, B, C, D, E}

        T₁ subtree splits: {(A,B), (C,D), (A,B,C,D), (E)}
        T₂ subtree splits: {(A,C), (B,D), (A,B,C,D), (E)}

        unique₁ = {(A,B), (C,D)}     ← groupings only in T₁
        unique₂ = {(A,C), (B,D)}     ← groupings only in T₂

        Shared: {(A,B,C,D), (E)}     ← splits both trees agree on

    ALGORITHMIC PURPOSE:
        These unique splits are used to:
        1. Identify conflicting topologies that require resolution
        2. Compute child frontiers (maximal shared elements per child)
        3. Build the conflict matrix for lattice construction
        4. Determine minimal jumping taxa solutions

    Args:
        tree1_node: Node from tree 1 rooted at the shared pivot split
        tree2_node: Node from tree 2 rooted at the shared pivot split

    Returns:
        A tuple (tree1_unique_splits, tree2_unique_splits) where:
        - tree1_unique_splits: Splits present in tree1 but absent in tree2
        - tree2_unique_splits: Splits present in tree2 but absent in tree1

    Mathematical Properties:
        - unique₁ ∩ unique₂ = ∅  (disjoint sets)
        - unique₁ ∪ unique₂ = (S₁ ∪ S₂) \ (S₁ ∩ S₂)  (symmetric difference)
        - If unique₁ = ∅ and unique₂ = ∅, then S₁ = S₂ (identical topologies)
    """
    tree1_unique_splits: PartitionSet[Partition] = (
        tree1_node.to_splits() - tree2_node.to_splits()
    )
    tree2_unique_splits: PartitionSet[Partition] = (
        tree2_node.to_splits() - tree1_node.to_splits()
    )

    return tree1_unique_splits, tree2_unique_splits


def _validate_nodes_exist(
    pivot_split: Partition, t1_node: Node | None, t2_node: Node | None
) -> None:
    """
    Validate that both trees contain nodes corresponding to a shared pivot split.

    MATHEMATICAL FOUNDATION:
        For a split S ∈ (T₁ ∩ T₂), there must exist corresponding nodes:
        - n₁ ∈ T₁ where split(n₁) = S
        - n₂ ∈ T₂ where split(n₂) = S

        If either node is missing, the trees are inconsistent with their
        reported common splits, indicating a data integrity error.

    PHYLOGENETIC INTERPRETATION:
        Shared splits represent clades that appear in both phylogenetic trees.
        Each split must correspond to an internal node (or leaf) in both trees.
        Missing nodes indicate corrupted tree structure or indexing errors.

    Args:
        pivot_split: The shared split being processed
        t1_node: Node from tree 1 (None if not found)
        t2_node: Node from tree 2 (None if not found)

    Raises:
        LatticeConstructionError: If either node is None, with details about
                                 which tree(s) are missing the split
    """
    if t1_node is None or t2_node is None:
        LatticeConstructionError.raise_missing_node(
            split=pivot_split,
            is_missing_in_tree1=t1_node is None,
            is_missing_in_tree2=t2_node is None,
        )


def _compute_conflict_taxa_size(conflicting_cover_pairs: PMatrix) -> int:
    """
    Compute the total taxa count from conflicting cover pairs.

    This helper function correctly counts taxa (not partitions) in the minimal
    cover of each conflict intersection. Used to compare conflict matrix size
    with nesting solution size during decision logic.

    Args:
        conflicting_cover_pairs: Matrix of conflicting PartitionSet pairs

    Returns:
        Total number of taxa across all minimal covers of conflicts
    """
    total = 0
    for left, right in conflicting_cover_pairs:
        minimal = (left & right).minimum_cover()
        total += sum(len(p.taxa) for p in minimal)  # Count TAXA, not partitions
    return total


def build_conflict_matrix(
    lattice_edge: PivotEdgeSubproblem,
) -> PMatrix:
    """
    Computes conflicting pairs of covers between two trees and returns them as a matrix.

    Each row in the returned matrix contains a conflicting pair [t1_cover, t2_cover].

    Decision logic: If nesting solutions are found, compares them with the conflict matrix
    and returns the approach that yields the smallest solution (most parsimonious).

    Args:
        lattice_edge: A PivotEdgeSubproblem object containing frontier information from both trees

    Returns:
        A matrix (list of lists) of conflicting PartitionSet pairs, or a 1×1 matrix
        containing the smallest nesting solution if that's more parsimonious.
        If no conflicts are found, returns an empty list.
    """
    left_covers: Dict[Partition, TopToBottom] = lattice_edge.tree1_child_frontiers
    right_covers: Dict[Partition, TopToBottom] = lattice_edge.tree2_child_frontiers

    # Collect all conflict types from cover pairs
    conflicting_cover_pairs, nesting_solutions, bottom_matrix = collect_all_conflicts(
        left_covers, right_covers
    )

    # Log matrices for debugging
    if bottom_matrix:
        jt_logger.matrix(
            bottom_matrix, title="Bottoms Conflict Matrix (Nesting Relationships)"
        )
    if conflicting_cover_pairs:
        jt_logger.matrix(conflicting_cover_pairs, title="Covering Conflict Matrix")

    # DECISION LOGIC: Choose between nesting solutions and conflict matrix
    # Compare the size of solutions to select the most parsimonious approach
    if nesting_solutions and conflicting_cover_pairs:
        # Both nesting and proper overlap conflicts exist - choose smaller solution

        # Find the row with smallest size
        min_row_size = min(matrix_row_size(row) for row in bottom_matrix)

        # Filter solutions from rows with smallest size
        candidates_with_smallest_rows = [
            (sol, idx)
            for idx, sol in enumerate(nesting_solutions)
            if matrix_row_size(bottom_matrix[idx]) == min_row_size
        ]

        # Among those, select the one with smallest solution size
        smallest_nesting_solution = min(
            candidates_with_smallest_rows, key=lambda x: solution_size(x[0])
        )[0]

        nesting_size = solution_size(smallest_nesting_solution)

        # ============================================================
        # BUGFIX #1: Use consistent metric for size comparison
        # Old code counted partitions, now counts taxa consistently
        # ============================================================
        # For conflict matrix, we would compute intersections which typically
        # yield smaller solutions than the original covers
        # Use consistent metric: count taxa (not partitions) for fair comparison
        conflict_size_estimate = _compute_conflict_taxa_size(conflicting_cover_pairs)

        jt_logger.info(
            f"Comparison: Nesting solution size={nesting_size} (from row size {min_row_size}), "
            f"Conflict matrix size estimate={conflict_size_estimate}"
        )

        # If conflict matrix would yield smaller solution, use it
        # In case of tie (equal sizes), prefer nesting solution (simpler logic)
        if conflict_size_estimate < nesting_size:
            jt_logger.info(
                f"Selecting conflict matrix (smaller: {conflict_size_estimate} < {nesting_size})"
            )
            return conflicting_cover_pairs
        else:
            jt_logger.info(
                f"Selecting nesting solution (smaller or equal: {nesting_size} <= {conflict_size_estimate})"
            )
            return [[smallest_nesting_solution]]

    # Only nesting solutions exist - return solution from smallest row
    elif nesting_solutions:
        # Find the row with smallest size
        min_row_size = min(matrix_row_size(row) for row in bottom_matrix)

        # Filter solutions from rows with smallest size
        candidates_with_smallest_rows = [
            (sol, idx)
            for idx, sol in enumerate(nesting_solutions)
            if matrix_row_size(bottom_matrix[idx]) == min_row_size
        ]

        # Among those, select the one with smallest solution size
        smallest_nesting = min(
            candidates_with_smallest_rows, key=lambda x: solution_size(x[0])
        )[0]

        jt_logger.info(
            f"Found {len(nesting_solutions)} nesting solution(s), "
            f"selected from smallest row (size {min_row_size}) "
            f"with solution size {solution_size(smallest_nesting)}: {smallest_nesting}"
        )

        # ============================================================
        # BUGFIX #2: Remove unreachable dead code
        # This entire if-block can never execute because if
        # conflicting_cover_pairs existed, we'd be in the first branch
        # ============================================================
        # Dead code removed (was lines 394-401 in original)

        # Return as 1×1 matrix: [[solution]]
        # When generalized_meet_product processes a 1×1 square matrix,
        # it returns the matrix element directly without computing intersection
        return [[smallest_nesting]]

    # Only conflict matrix exists - return it
    jt_logger.matrix(conflicting_cover_pairs, title="Partition Conflict Matrix")
    return conflicting_cover_pairs
