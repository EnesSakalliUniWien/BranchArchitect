"""
Updated lattice algorithm functions with proper Pydantic validation.
"""

from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.lattice_solution import LatticeSolutions
from typing import List, Tuple, Set, Dict

# Import lattice modules
from brancharchitect.jumping_taxa.lattice.lattice_construction import (
    construct_sub_lattices,
    build_partition_conflict_matrix,
    are_cover_lists_equivalent,
)
from brancharchitect.jumping_taxa.lattice.matrix_ops import (
    split_matrix,
    solve_matrix_puzzle,
    generalized_meet_product,
)

# Map s-edges back to original common splits before accumulating
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_s_edges_to_original_by_index,
)
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.mapping import (
    sort_lattice_edges_by_subset_hierarchy,
)

# Constants
MAX_ITERATIONS = 5


def process_single_lattice_edge(
    edge: LatticeEdge,
    solutions_manager: LatticeSolutions,
) -> bool:
    """
    Analyze a single LatticeEdge and store solutions in the solutions_manager.

    Returns True if processing is complete for this edge, False otherwise.
    """
    jt_logger.info(f"Processing edge: {edge.split} at visit {edge.visits}")

    # 1. Early exit if covers are equivalent (no conflict)
    if are_cover_lists_equivalent(edge.t1_common_covers, edge.t2_common_covers):
        jt_logger.info(
            f"Skipping {edge.split} at visit {edge.visits} as covers are equivalent."
        )
        return True

    # 2. Early exit if no conflict matrix can be built
    candidate_matrix = build_partition_conflict_matrix(edge)
    if not candidate_matrix:
        return True

    # 3. Determine which solver to use based on matrix type
    matrices = split_matrix(candidate_matrix)
    jt_logger.section("Meet Result Computation")

    if len(matrices) > 1:
        solver = solve_matrix_puzzle
        solver_args = {"matrix1": matrices[0], "matrix2": matrices[1]}
    else:
        solver = generalized_meet_product
        solver_args = {"matrix": matrices[0]}
        jt_logger.matrix(matrices[0])

    # 4. Run the selected solver
    solutions = solver(**solver_args)

    # 5. Handle the results in a unified block
    if not solutions:
        # No solution found, processing is done for this path
        return True

    # Solution was found, add it to the manager
    jt_logger.info(f"Adding solutions for {edge.split} at visit {edge.visits}")
    jt_logger.info(f"Solutions: {solutions}")
    solutions_manager.add_solutions(
        edge.split, solutions, category="solution", visit=edge.visits
    )
    # Return False to indicate that a solution was found and the edge may need re-processing
    return False


def process_iteration(
    sub_lattices: List[LatticeEdge], lattice_solutions_manager: LatticeSolutions
) -> None:
    """
    Process a set of sub-lattices to find solutions.
    """
    processing_stack: List[LatticeEdge] = sub_lattices.copy()
    jt_logger.section("Processing Stack")
    for s_edge_obj_log in processing_stack:  # Renamed to avoid conflict
        jt_logger.info(f"Initial stack processing {s_edge_obj_log.split}")

    while processing_stack:
        s_edge_obj: LatticeEdge = processing_stack.pop()
        s_edge_obj.visits += 1
        jt_logger.info(
            f"s_edge {s_edge_obj.split} updated to visit {s_edge_obj.visits}"
        )

        done: bool = process_single_lattice_edge(s_edge_obj, lattice_solutions_manager)

        if not done:
            solutions_for_visit: List[PartitionSet[Partition]] = (
                lattice_solutions_manager.get_solutions_for_edge_visit(
                    s_edge_obj.split, s_edge_obj.visits
                )
            )

            if solutions_for_visit:
                s_edge_obj.remove_solutions_from_covers(solutions_for_visit)
                processing_stack.append(s_edge_obj)
            else:
                jt_logger.info(
                    f"No solutions for {s_edge_obj.split} at visit {s_edge_obj.visits} after processing; skipping re-add."
                )


@jt_logger.log_execution
def lattice_algorithm(
    input_tree1: Node, input_tree2: Node, leaf_order: List[str]
) -> Tuple[List[List[Partition]], List[Partition]]:
    """
    Execute the lattice algorithm to find jumping taxa between two trees.

    Args:
        input_tree1: First tree
        input_tree2: Second tree
        leaf_order: Order of leaf nodes

    Returns:
        A tuple containing:
        - List of "solution sets" (List[List[Partition]]), where each inner list (PartitionSet)
          represents a group of jumping taxa derived from one s-edge.
        - List of Partition objects representing the s-edges from which these solutions were derived.
    """
    try:
        jt_logger.log_newick_strings(input_tree1, input_tree2)

        lattice_solutions_manager = LatticeSolutions()

        current_s_edges: List[LatticeEdge] | None = construct_sub_lattices(
            input_tree1, input_tree2
        )

        jt_logger.section("Initial Sub-Lattices")

        if current_s_edges:
            # Sort lattice edges by subset hierarchy for consistent processing order
            current_s_edges = sort_lattice_edges_by_subset_hierarchy(
                current_s_edges, input_tree1, input_tree2
            )
            jt_logger.info(
                f"Sorted {len(current_s_edges)} lattice edges by subset hierarchy"
            )

            process_iteration(current_s_edges, lattice_solutions_manager)

        jt_logger.info(
            f"Processed {len(lattice_solutions_manager.solutions_for_s_edge)} s-edges with solutions"
        )

        # Early return if no solutions found
        if not lattice_solutions_manager.solutions_for_s_edge:
            jt_logger.info("No solutions found in lattice solutions manager")
            return [], []

        # solution_sets_list will be List[List[Partition]]
        solution_sets_list: List[List[Partition]] = []
        s_edges_of_solutions_list: List[Partition] = []

        # Group solutions by s_edge (ignoring visit) to find all minimal solutions
        s_edge_to_solutions: Dict[
            Partition, List[Tuple[PartitionSet[Partition], int, int]]
        ] = {}

        for (
            s_edge_partition,  # This is a Partition object
            visit,
        ), category_sols in lattice_solutions_manager.solutions_for_s_edge.items():
            if not category_sols:
                continue

            # Get the single smallest solution for this visit
            smallest_solution: PartitionSet[Partition] | None = (
                lattice_solutions_manager.get_single_smallest_solution(
                    s_edge_partition, visit
                )
            )
            jt_logger.info(
                f"Edge {s_edge_partition} at visit {visit} has solution: {smallest_solution}"
            )

            if smallest_solution:
                # Calculate solution size (total number of taxa)
                solution_size = sum(
                    len(partition.indices) for partition in smallest_solution
                )

                # Collect all solutions for this s_edge
                if s_edge_partition not in s_edge_to_solutions:
                    s_edge_to_solutions[s_edge_partition] = []
                s_edge_to_solutions[s_edge_partition].append(
                    (smallest_solution, solution_size, visit)
                )

        # For each s_edge, keep only the solutions with minimum size
        for s_edge_partition, solutions in s_edge_to_solutions.items():
            # Find minimum solution size
            min_size = min(size for _, size, _ in solutions)

            # Keep only solutions with minimum size
            minimal_solutions = [
                (sol, size, visit) for sol, size, visit in solutions if size == min_size
            ]

            # Add all minimal solutions
            for solution, size, visit in minimal_solutions:
                # Convert PartitionSet to List[Partition]
                selected_solution_set: List[Partition] = list(solution)

                solution_sets_list.append(selected_solution_set)
                s_edges_of_solutions_list.append(s_edge_partition)
                jt_logger.info(
                    f"Selected minimal solution for s_edge {s_edge_partition} (visit {visit}, size {size}): {selected_solution_set}"
                )

        return solution_sets_list, s_edges_of_solutions_list

    except Exception as e:
        from brancharchitect.jumping_taxa.debug import log_stacktrace

        log_stacktrace(e)
        raise Exception(f"Error in lattice_algorithm: {str(e)}")


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
        - s_edge_solutions_dict: Empty dictionary for storing s-edge -> solutions mapping
        - current_t1: Deep copy of first tree
        - current_t2: Deep copy of second tree
        - iteration_count: Starting iteration count (0)
    """
    s_edge_solutions_dict: Dict[Partition, List[List[Partition]]] = {}

    current_t1: Node = input_tree1.deep_copy()
    current_t2: Node = input_tree2.deep_copy()
    iteration_count = 0

    return (
        s_edge_solutions_dict,
        current_t1,
        current_t2,
        iteration_count,
    )


def _check_loop_termination_conditions(
    current_t1: Node,
    current_t2: Node,
    iteration_count: int,
    max_iterations: int,
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
    # Check for maximum iterations exceeded
    if iteration_count > max_iterations:
        jt_logger.error(
            f"Exceeded maximum iterations ({max_iterations}). Breaking loop."
        )
        return True

    # Log current iteration
    jt_logger.section(f"Iteration {iteration_count}")

    # Check for unique splits between trees
    unique_splits: PartitionSet[Partition] = (
        current_t1.to_splits() ^ current_t2.to_splits()
    )

    if not unique_splits:
        jt_logger.info(
            f"Iter {iteration_count}: No unique splits. Loop will terminate."
        )
        return True

    return False


def _identify_and_delete_jumping_taxa(
    solution_sets_this_iter: List[List[Partition]],
    current_t1: Node,
    current_t2: Node,
    iteration_count: int,
) -> Tuple[bool, Set[int]]:
    """
    Identify jumping taxa from solutions and delete them from trees.

    Args:
        solution_sets_this_iter: Solution sets from current iteration
        current_t1: Current state of first tree (modified in-place)
        current_t2: Current state of second tree (modified in-place)
        iteration_count: Current iteration number

    Returns:
        Tuple containing:
        - should_break_loop: Boolean indicating if loop should break
        - taxa_indices_to_delete: Set of taxa indices that were deleted
    """
    # Collect taxa indices from all solution partitions
    taxa_indices_to_delete: Set[int] = set()
    for solution_set in solution_sets_this_iter:
        for sol_partition in solution_set:
            taxa_indices_to_delete.update(sol_partition.resolve_to_indices())

    # Check if there are any taxa to delete
    if not taxa_indices_to_delete:
        jt_logger.warning(
            f"Iter {iteration_count}: Solutions found, but no taxa indices to delete. Breaking."
        )
        return True, taxa_indices_to_delete  # Break loop

    # Log and perform deletion
    jt_logger.info(
        f"Iter {iteration_count}: Deleting taxa indices: {list(taxa_indices_to_delete)}"
    )
    current_t1.delete_taxa(list(taxa_indices_to_delete))
    current_t2.delete_taxa(list(taxa_indices_to_delete))

    # Check if trees have enough leaves to continue
    if len(current_t1.get_leaves()) < 2 or len(current_t2.get_leaves()) < 2:
        jt_logger.info(
            f"Iter {iteration_count}: One or both trees have too few leaves. Stopping."
        )
        return True, taxa_indices_to_delete  # Break loop
    return False, taxa_indices_to_delete  # Continue loop


def iterate_lattice_algorithm(
    input_tree1: Node, input_tree2: Node, leaf_order: List[str] = []
) -> Dict[Partition, List[List[Partition]]]:
    """
    Iteratively apply the lattice algorithm.
    Returns:
        Dictionary mapping s-edge Partitions to their corresponding solution sets.
        Format: {s_edge: [solution_set_1, solution_set_2, ...]}

    Note: Only returns s-edges that exist in the original input trees to ensure
    they can be used for interpolation.
    """
    # Original trees are passed to the mapping function for proper s-edge mapping

    # Initialize iteration variables using helper function
    (
        s_edge_solutions_dict,
        current_t1,
        current_t2,
        iteration_count,
    ) = _initialize_iteration_variables(input_tree1, input_tree2)

    while True:
        iteration_count += 1

        # Check termination conditions using helper function
        if _check_loop_termination_conditions(
            current_t1, current_t2, iteration_count, MAX_ITERATIONS
        ):
            break

        # Run lattice algorithm for current iteration
        solution_sets_this_iter, s_edges_this_iter_unmapped = lattice_algorithm(
            current_t1, current_t2, leaf_order
        )

        if not solution_sets_this_iter:
            jt_logger.info(
                f"Iter {iteration_count}: No solutions found by lattice_algorithm. Stopping."
            )
            break

        s_edges_mapped_to_original = map_s_edges_to_original_by_index(
            s_edges_this_iter_unmapped,
            input_tree1,  # original_t1
            input_tree2,  # original_t2
            current_t1,
            current_t2,
            iteration_count,
        )

        # Preserve all solutions by either mapping to original s-edges or using unmapped s-edges directly
        for i, (original_s_edge, unmapped_s_edge) in enumerate(
            zip(s_edges_mapped_to_original, s_edges_this_iter_unmapped)
        ):
            if i < len(solution_sets_this_iter):
                solution_set = solution_sets_this_iter[i]
                # Use mapped s-edge if available, otherwise use unmapped s-edge to preserve all solutions
                s_edge_key = (
                    original_s_edge if original_s_edge is not None else unmapped_s_edge
                )
                s_edge_solutions_dict.setdefault(s_edge_key, []).append(solution_set)

        print(
            f"Iter {iteration_count}: Found {len(solution_sets_this_iter)} solution sets."
        )

        # Identify and delete jumping taxa using helper function
        should_break_loop, _ = _identify_and_delete_jumping_taxa(
            solution_sets_this_iter, current_t1, current_t2, iteration_count
        )

        if should_break_loop:
            break

    if (
        current_t1.to_splits() != current_t2.to_splits()
        and iteration_count <= MAX_ITERATIONS
    ):
        jt_logger.warning(
            "iterate_lattice_algorithm terminated: No further solutions, but trees still differ."
        )
    elif iteration_count > MAX_ITERATIONS:
        jt_logger.warning(
            "iterate_lattice_algorithm terminated: Exceeded max iterations."
        )
    else:
        jt_logger.info(
            "iterate_lattice_algorithm completed: Trees reconciled or loop condition met."
        )

    # S-edges have already been mapped to original common splits during accumulation
    return s_edge_solutions_dict


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
    # Get results in new dictionary format
    s_edge_solutions: Dict[Partition, List[List[Partition]]] = (
        iterate_lattice_algorithm(input_tree1, input_tree2, leaf_order)
    )

    # Convert to old tuple format for backward compatibility
    jumping_taxa: List[Tuple[int, ...]] = []

    for _, solution_sets in s_edge_solutions.items():
        for solution_set in solution_sets:
            for partition in solution_set:
                # Convert partition indices to tuple format
                indices: Tuple[int, ...] = tuple(sorted(partition.resolve_to_indices()))
                if indices and indices not in jumping_taxa:
                    jumping_taxa.append(indices)

    return jumping_taxa
