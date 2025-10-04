"""
Updated lattice algorithm functions with proper Pydantic validation.
"""

from brancharchitect.tree import Node
from typing import List, Tuple, Dict
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.elements.partition_set import PartitionSet

# Map s-edges back to original common splits before accumulating
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_s_edges_by_jaccard_similarity,
)
from brancharchitect.jumping_taxa.lattice.lattice_solver import (
    lattice_algorithm,
)
from brancharchitect.jumping_taxa.lattice.delete_taxa import (
    identify_and_delete_jumping_taxa,
)


def _check_loop_termination_conditions(
    current_t1: Node,
    current_t2: Node,
) -> bool:
    """
    Check if the main iteration loop should terminate.

    Args:
        current_t1: Current state of first tree
        current_t2: Current state of second tree
        iteration_count: Current iteration number
        max_iterations: Maximum allowed iterations

    Returns:
        True if loop should terminate, False otherwise
    """
    # Check for unique splits between trees
    unique_splits: PartitionSet[Partition] = (
        current_t1.to_splits() ^ current_t2.to_splits()
    )
    if not unique_splits:
        return True
    return False


def _initialize_iteration_variables(
    input_tree1: Node, input_tree2: Node
) -> Tuple[Dict[Partition, List[List[Partition]]], Node, Node, int]:
    """
    Initialize variables for the iterate_lattice_algorithm function.

    Args:
        input_tree1: First input tree
        input_tree2: Second input tree

    Returns:
        Tuple containing:
        - jumping_subtree_solutions_dict: Empty dictionary for storing jumping subtree solutions
        - current_t1: Deep copy of first tree
        - current_t2: Deep copy of second tree
        - iteration_count: Starting iteration count (0)
    """
    jumping_subtree_solutions_dict: Dict[Partition, List[List[Partition]]] = {}

    current_t1: Node = input_tree1.deep_copy()
    current_t2: Node = input_tree2.deep_copy()
    iteration_count = 0

    return (
        jumping_subtree_solutions_dict,
        current_t1,
        current_t2,
        iteration_count,
    )


def iterate_lattice_algorithm(
    input_tree1: Node, input_tree2: Node, leaf_order: List[str] = []
) -> Dict[Partition, List[List[Partition]]]:
    """
    Iteratively apply the lattice algorithm to find jumping taxa solutions.
    Returns:
        Dictionary mapping active-changing splits to their corresponding jumping subtree solution sets.
        Format: {split: [solution_set_1, solution_set_2, ...]}

    Each solution set contains partitions representing taxa that need to "jump" between
    phylogenetic positions, enabling subtree rearrangements during interpolation.

    Note: Only returns splits that exist in the original input trees to ensure
    they can be used for interpolation.
    """
    jt_logger.section("Iterative Lattice Algorithm")

    # Original trees are passed to the mapping function for proper split mapping

    # Initialize iteration variables using helper function
    (
        jumping_subtree_solutions_dict,
        current_t1,
        current_t2,
        iteration_count,
    ) = _initialize_iteration_variables(input_tree1, input_tree2)

    while True:
        iteration_count += 1
        jt_logger.subsection(f"Iteration {iteration_count}")

        # Check termination conditions using helper function
        if _check_loop_termination_conditions(current_t1, current_t2):
            jt_logger.info("Trees are now identical - terminating iterations")
            break

        # Run lattice algorithm for current iteration
        solution_sets_this_iter, splits_this_iter_unmapped = lattice_algorithm(
            current_t1, current_t2, leaf_order
        )

        splits_mapped_to_original = map_s_edges_by_jaccard_similarity(
            splits_this_iter_unmapped,
            input_tree1,  # original_t1
            input_tree2,  # original_t2
        )

        # Preserve all jumping subtree solutions by mapping each solution to its corresponding split
        for i, split in enumerate(splits_mapped_to_original):
            if (
                i < len(solution_sets_this_iter) and split is not None
            ):  # Safety check and None check
                solutions = solution_sets_this_iter[i]
                if split not in jumping_subtree_solutions_dict:
                    jumping_subtree_solutions_dict[split] = []
                jumping_subtree_solutions_dict[split].append(solutions)

        # Identify and delete jumping taxa using helper function
        should_break_loop, _ = identify_and_delete_jumping_taxa(
            solution_sets_this_iter, current_t1, current_t2, iteration_count
        )

        if should_break_loop:
            break

        # Summary for this iteration
        total_solutions = sum(len(sol_set) for sol_set in solution_sets_this_iter)
        jt_logger.info(
            f"  Total: {total_solutions} jumping subtree solution(s) in this iteration"
        )

    # Splits have already been mapped to original common splits during accumulation
    return jumping_subtree_solutions_dict


def adapter_iterate_lattice_algorithm(
    input_tree1: Node, input_tree2: Node, leaf_order: List[str] = []
) -> List[Tuple[int, ...]]:
    """
    Backward compatibility adapter for the lattice algorithm.

    Converts the new dictionary format back to the old tuple format
    that existing tests and API expect.

    Args:
        input_tree1: First input tree
        input_tree2: Second input tree
        leaf_order: Order of leaf nodes

    Returns:
        List of tuples representing jumping taxa indices in the old format
    """
    # Get results in new jumping subtree solutions format
    jumping_subtree_solutions: Dict[Partition, List[List[Partition]]] = (
        iterate_lattice_algorithm(input_tree1, input_tree2, leaf_order)
    )

    # Convert to old tuple format for backward compatibility
    jumping_taxa: List[Tuple[int, ...]] = []

    for _, solution_sets in jumping_subtree_solutions.items():
        for solution_set in solution_sets:
            for partition in solution_set:
                # Convert partition indices to tuple format
                indices: Tuple[int, ...] = tuple(sorted(partition.resolve_to_indices()))
                if indices and indices not in jumping_taxa:
                    jumping_taxa.append(indices)

    return jumping_taxa
