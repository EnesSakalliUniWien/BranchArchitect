"""
Updated lattice algorithm functions with proper Pydantic validation.
"""

from typing import List, Tuple, Any
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.lattice_solution import LatticeSolutions

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


def process_single_lattice_edge(
    edge: LatticeEdge,
    solutions_manager: Any,  # Changed from LatticeSolutions to Any
) -> bool:
    """
    Analyze a single LatticeEdge and store solutions in the solutions_manager.
    """
    if are_cover_lists_equivalent(edge.t1_common_covers, edge.t2_common_covers):
        jt_logger.info(
            f"Skipping {edge.split} at visit {edge.visits} as covers are equivalent."
        )
        return True

    candidate_matrix = build_partition_conflict_matrix(edge)
    if not candidate_matrix:
        return True

    # Split matrix
    matrices = split_matrix(candidate_matrix)
    jt_logger.section("Meet Result Computation")

    # Process degenerate case (multiple matrices)
    if len(matrices) > 1:
        solutions = solve_matrix_puzzle(matrix1=matrices[0], matrix2=matrices[1])
        case_label = "degenerate"

        if solutions:
            jt_logger.info(
                f"Adding solutions for {edge.split} at visit {edge.visits}, Case {case_label}"
            )
            jt_logger.info(f"Solutions: {solutions}")

            # Add solutions ensuring they are PartitionSets
            solutions_manager.add_solutions(
                edge.split, solutions, category=case_label, visit=edge.visits
            )
            return False

        return True

    # Process non-degenerate case (single matrix)
    matrix = matrices[0]
    jt_logger.matrix(matrix)

    solutions = generalized_meet_product(matrix=matrix)
    case_label = "non-degenerate"

    if solutions:
        jt_logger.info(
            f"Adding solutions for {edge.split} at visit {edge.visits}, Case {case_label}"
        )
        jt_logger.info(f"Solutions: {solutions}")

        # Add each valid solution
        for solution in solutions:
            if solution:
                solutions_manager.add_solutions(
                    edge.split, [solution], category=case_label, visit=edge.visits
                )

        return False
    else:
        return True


# Modified process_iteration function with Any type annotation for lattice_solutions
def process_iteration(sub_lattices: List[LatticeEdge], lattice_solutions: Any) -> None:
    """
    Process a set of sub-lattices to find solutions.

    Args:
        sub_lattices: List of lattice edges to process
        lattice_solutions: Solution manager to store results
    """
    # Initialize the stack with all edges
    processing_stack = sub_lattices.copy()

    jt_logger.section("Processing Stack")
    for s_edge in processing_stack:
        jt_logger.info(f"Processing {s_edge.split}")

    while processing_stack:
        # Get the next edge to process
        s_edge = processing_stack.pop()
        s_edge.visits += 1

        jt_logger.info(f"s_edge {s_edge.split} updated to visit {s_edge.visits}")

        # Process the edge
        done = process_single_lattice_edge(s_edge, lattice_solutions)

        if not done:
            # Get solutions for this edge and visit
            solutions = lattice_solutions.get_solutions_for_edge_visit(
                s_edge.split, s_edge.visits
            )
            jt_logger.log_solutions_for_sub_lattice(s_edge, solutions)

            # If solutions found, remove them from covers
            if solutions:
                jt_logger.section("Removing Solutions from Covers")

                jt_logger.log_covet(
                    s_edge.t1_common_covers,
                    s_edge.t2_common_covers,
                    "Before",
                    s_edge.encoding,
                )

                s_edge.remove_solutions_from_covers(solutions)

                jt_logger.log_covet(
                    s_edge.t1_common_covers,
                    s_edge.t2_common_covers,
                    "After",
                    s_edge.encoding,
                )

                # Re-add to processing stack
                processing_stack.append(s_edge)
            else:
                jt_logger.info(
                    f"No minimal solutions for {s_edge.split} at visit {s_edge.visits}; skipping re-add."
                )


@jt_logger.log_execution
def lattice_algorithm(
    input_tree1: Node, input_tree2: Node, leaf_order: List[str]
) -> List[Tuple[int, ...]]:
    """
    Execute the lattice algorithm to find jumping taxa between two trees.

    Args:
        input_tree1: First tree
        input_tree2: Second tree
        leaf_order: Order of leaf nodes

    Returns:
        List of tuples representing indices of jumping taxa
    """
    try:
        jt_logger.log_newick_strings(input_tree1, input_tree2)

        # Initialize solution manager
        lattice_solution = LatticeSolutions()

        # Construct sub-lattices
        current_s_edges = construct_sub_lattices(input_tree1, input_tree2)

        # Log initial information
        jt_logger.section("Initial Sub-Lattices")

        # Process sub-lattices if any exist
        if current_s_edges:
            process_iteration(current_s_edges, lattice_solution)

        # Collect final solutions
        solutions_set: List[Tuple[int, ...]] = []

        # Process each s_edge and visit combination
        for (
            s_edge,
            visit,
        ), category_sols in lattice_solution.solutions_for_s_edge.items():
            case_label = next(iter(category_sols.keys())) if category_sols else None

            if case_label:
                # Get minimal solutions for this edge and visit
                minimal_solutions = lattice_solution.get_minimal_by_indices_sum(
                    s_edge, visit
                )

                jt_logger.info(
                    f"Edge {s_edge} at visit {visit} has {case_label} solutions: {minimal_solutions}"
                )

                # Process solutions based on category
                if minimal_solutions:
                    selected_solution = minimal_solutions[0]

                    # Add solution indices to result
                    for index in selected_solution.resolve_to_indices():
                        solutions_set.append(index)

                    jt_logger.info(f"Selected solution: {selected_solution}")

        return solutions_set

    except Exception as e:
        from brancharchitect.jumping_taxa.debug import log_stacktrace

        log_stacktrace(e)
        raise Exception(f"Error in lattice_algorithm: {str(e)}")


def iterate_lattice_algorithm(
    input_tree1: Node, input_tree2: Node, leaf_order: List[str]
) -> List[Tuple[int, ...]]:
    """
    Execute the lattice algorithm iteratively to find jumping taxa between two trees.

    Args:
        input_tree1: First tree
        input_tree2: Second tree
        leaf_order: Order of leaf nodes

    Returns:
        List of tuples representing indices of jumping taxa
    """
    solution_set = []
    t1, t2 = input_tree1.deep_copy(), input_tree2.deep_copy()

    while True:
        t1_unique_splits = t1.to_splits() - t2.to_splits()
        t2_unique_splits = t2.to_splits() - t1.to_splits()

        if len(t1_unique_splits) == 0 and len(t2_unique_splits) == 0:
            break

        solutions = lattice_algorithm(t1, t2, leaf_order)
        solution_set.extend(solutions)

        for solution in solutions:
            t1.delete_taxa([s for s in solution])
            t2.delete_taxa([s for s in solution])

        t1, t2 = t1.deep_copy(), t2.deep_copy()

    return solution_set
