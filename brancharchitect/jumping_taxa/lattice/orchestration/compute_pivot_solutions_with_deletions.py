"""Iterative lattice algorithm driver that deletes jumping taxa across visits."""

from brancharchitect.tree import Node
from typing import List, Tuple, Dict, Set
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.debug import jt_logger
import logging
from brancharchitect.jumping_taxa.lattice.solvers.pivot_edge_solver import (
    lattice_algorithm,
)
from brancharchitect.jumping_taxa.lattice.orchestration.delete_taxa import (
    identify_and_delete_jumping_taxa,
)

logger = logging.getLogger(__name__)


def compute_pivot_solutions_with_deletions(
    input_tree1: Node, input_tree2: Node, leaf_order: List[str] = []
) -> Tuple[Dict[Partition, List[Partition]], List[Set[int]]]:
    """
    Iteratively apply the lattice algorithm to find jumping taxa solutions.
    Returns a tuple of:
      - Dict[Partition, List[Partition]] mapping each pivot edge to a flat list of
        solution partitions (jumping taxa groups) selected by parsimony.
      - List[Set[int]] of taxa indices actually deleted in each iteration.

    Note: Only returns splits mapped to the original input trees to ensure
    usability in interpolation.
    """
    jt_logger.section("Iterative Lattice Algorithm")

    # Initialize iteration variables
    jumping_subtree_solutions_dict: Dict[Partition, List[Partition]] = {}
    deleted_taxa_per_iteration: List[Set[int]] = []
    current_t1: Node = input_tree1.deep_copy()
    current_t2: Node = input_tree2.deep_copy()
    iteration_count = 0

    while True:
        iteration_count += 1
        jt_logger.subsection(f"Iteration {iteration_count}")

        # Check if trees are now identical (using Node.__eq__ which compares full topology)
        if current_t1 == current_t2:
            jt_logger.info("Trees are now identical - terminating iterations")
            break

        # Run lattice algorithm - it now returns a dictionary with mapped pivot edges
        solutions_dict_this_iter = lattice_algorithm(
            current_t1, current_t2, input_tree1, input_tree2
        )

        # Accumulate solutions (flat partitions) from this iteration into the global dictionary
        for split, partitions in solutions_dict_this_iter.items():
            jt_logger.info(f"[LATTICE]   Mapped split: {split.bipartition()}")
            jumping_subtree_solutions_dict.setdefault(split, []).extend(partitions)

        # For deletion, we need a flat list of partitions for this iteration
        partitions_this_iter = [
            part for parts in solutions_dict_this_iter.values() for part in parts
        ]

        # Identify and delete jumping taxa using helper function
        should_break_loop, deleted_indices = identify_and_delete_jumping_taxa(
            partitions_this_iter, current_t1, current_t2, iteration_count
        )

        # Track which taxa were actually deleted in this iteration
        if deleted_indices:
            deleted_taxa_per_iteration.append(deleted_indices)

        if should_break_loop:
            break

        # Summary for this iteration
        total_solutions = len(partitions_this_iter)
        jt_logger.info(
            f"  Total: {total_solutions} jumping subtree solution(s) in this iteration"
        )

    # Splits have already been mapped to original common splits during accumulation
    return jumping_subtree_solutions_dict, deleted_taxa_per_iteration


def adapter_compute_pivot_solutions_with_deletions(
    input_tree1: Node, input_tree2: Node, leaf_order: List[str] = []
) -> List[Tuple[int, ...]]:
    """
    Adapter for the lattice algorithm.

    Converts the dictionary format back to the old tuple format
    expected by callers needing the deleted-jumping-taxa list.

    Args:
        input_tree1: First input tree
        input_tree2: Second input tree
        leaf_order: Order of leaf nodes

    Returns:
        List of tuples representing jumping taxa indices in the old format.
        ONLY includes taxa that were actually deleted during the iterative process.
    """
    # Get results in new jumping subtree solutions format
    jumping_subtree_solutions, deleted_taxa_per_iteration = (
        compute_pivot_solutions_with_deletions(input_tree1, input_tree2, leaf_order)
    )

    # Convert to old tuple format for backward compatibility
    # Only include partitions that contain indices that were actually deleted
    jumping_taxa: List[Tuple[int, ...]] = []

    # Collect all deleted indices across all iterations
    all_deleted_indices: Set[int] = set()
    for deleted_set in deleted_taxa_per_iteration:
        all_deleted_indices.update(deleted_set)

    # Now extract the minimal jumping taxa from the solutions
    # that correspond to the actually deleted indices
    seen_partitions: Set[Tuple[int, ...]] = set()

    for _, partitions in jumping_subtree_solutions.items():
        for partition in partitions:
            indices: Tuple[int, ...] = tuple(sorted(partition.resolve_to_indices()))
            if indices and any(idx in all_deleted_indices for idx in indices):
                if indices not in seen_partitions:
                    seen_partitions.add(indices)
                    jumping_taxa.append(indices)

    return jumping_taxa
