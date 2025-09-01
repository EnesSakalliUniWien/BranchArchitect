"""
Updated lattice algorithm functions with proper Pydantic validation.
"""

from brancharchitect.tree import Node
from typing import List, Tuple
from collections import defaultdict

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge

from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.lattice_solution import LatticeSolutions

# Import lattice modules
from brancharchitect.jumping_taxa.lattice.lattice_construction import (
    construct_sub_lattices,
)

# Map s-edges back to original common splits before accumulating
from brancharchitect.jumping_taxa.lattice.mapping import (
    sort_lattice_edges_by_subset_hierarchy,
)
from brancharchitect.jumping_taxa.lattice.lattice_algorithm import (
    process_single_lattice_edge,
)


def solve_lattice_edges(
    sub_lattices: List[LatticeEdge], lattice_solutions_manager: LatticeSolutions
) -> None:
    """
    Process a set of sub-lattices to find solutions.
    """
    processing_stack: List[LatticeEdge] = sub_lattices.copy()

    while processing_stack:
        s_edge_obj: LatticeEdge = processing_stack.pop()
        s_edge_obj.visits += 1

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
        jt_logger.section("Lattice Algorithm Execution")
        jt_logger.log_newick_strings(input_tree1, input_tree2)

        lattice_solutions_manager = LatticeSolutions()

        current_s_edges: List[LatticeEdge] | None = construct_sub_lattices(
            input_tree1, input_tree2
        )

        jt_logger.subsection("Initial Sub-Lattices")

        current_s_edges = sort_lattice_edges_by_subset_hierarchy(
            current_s_edges, input_tree1, input_tree2
        )

        solve_lattice_edges(current_s_edges, lattice_solutions_manager)

        # solution_sets_list will be List[List[Partition]]
        solution_sets_list: List[List[Partition]] = []
        s_edges_of_solutions_list: List[Partition] = []

        # Group solutions by s_edge (ignoring visit) to find all minimal solutions
        s_edge_to_solutions: defaultdict[
            Partition, List[Tuple[PartitionSet[Partition], int, int]]
        ] = defaultdict(list)

        for (
            s_edge_partition,  # This is a Partition object
            visit,
        ), _ in lattice_solutions_manager.solutions_for_s_edge.items():
            # Get the single smallest solution for this visit
            smallest_solution: PartitionSet[Partition] | None = (
                lattice_solutions_manager.get_single_smallest_solution(
                    s_edge_partition, visit
                )
            )

            if smallest_solution:
                solution_size = sum(
                    len(partition.indices) for partition in smallest_solution
                )
                s_edge_to_solutions[s_edge_partition].append(
                    (smallest_solution, solution_size, visit)
                )

        # For each s_edge, keep only the solutions with minimum size
        for s_edge_partition, solutions in s_edge_to_solutions.items():
            min_size = min(size for _, size, _ in solutions)
            minimal_solutions = [
                (sol, size, visit) for sol, size, visit in solutions if size == min_size
            ]

            # Add all minimal solutions
            for solution, _, visit in minimal_solutions:
                # Convert PartitionSet to List[Partition]
                selected_solution_set: List[Partition] = list(solution)
                solution_sets_list.append(selected_solution_set)
                s_edges_of_solutions_list.append(s_edge_partition)

        return solution_sets_list, s_edges_of_solutions_list

    except Exception as e:
        from brancharchitect.jumping_taxa.debug import log_stacktrace

        log_stacktrace(e)
        raise Exception(f"Error in lattice_algorithm: {str(e)}")
