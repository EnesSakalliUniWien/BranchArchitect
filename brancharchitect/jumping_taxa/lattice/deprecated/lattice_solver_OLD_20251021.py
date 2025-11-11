from __future__ import annotations
from brancharchitect.tree import Node
from typing import List, Tuple
from collections import defaultdict

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)

from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.lattice_solution import LatticeSolutions

# Import lattice modules
from brancharchitect.jumping_taxa.lattice.build_pivot_lattices import (
    construct_sublattices,
)

# Map s-edges back to original common splits before accumulating
from brancharchitect.jumping_taxa.lattice.edge_depth_ordering import (
    sort_lattice_edges_by_subset_hierarchy,
)
from brancharchitect.jumping_taxa.lattice.pivot_edge_problem_solver import (
    process_single_lattice_edge,
)


def solve_lattice_edges(
    sub_lattices: List[PivotEdgeSubproblem],
    lattice_solutions_manager: LatticeSolutions,
    t1: Node,
    t2: Node,
    stop_after_first_solution: bool = True,
) -> None:
    """
    Process a set of sub-lattices to find solutions.

    Args:
        sub_lattices: List of pivot subproblems to process
        lattice_solutions_manager: Manager for storing solutions
        t1: First tree
        t2: Second tree
        stop_after_first_solution: If True, stops processing a pivot after finding
            its first solution. This prevents spurious "nesting solutions" that arise
            from incremental removal artifacts rather than actual jumping taxa.
    """
    processing_stack: List[PivotEdgeSubproblem] = sub_lattices.copy()

    while processing_stack:
        current_pivot_edge: PivotEdgeSubproblem = processing_stack.pop()
        current_pivot_edge.visits += 1

        done: bool = process_single_lattice_edge(
            current_pivot_edge, lattice_solutions_manager, t1, t2
        )

        if not done:
            solutions_for_visit: List[PartitionSet[Partition]] = (
                lattice_solutions_manager.get_solutions_for_edge_visit(
                    current_pivot_edge.pivot_split, current_pivot_edge.visits
                )
            )

            if solutions_for_visit:
                current_pivot_edge.remove_solutions_from_covers(solutions_for_visit)

                # Only re-add to stack if we want to continue processing this pivot
                if not stop_after_first_solution:
                    processing_stack.append(current_pivot_edge)
                else:
                    jt_logger.info(
                        f"[solve_lattice_edges] Stopping after first solution for pivot {current_pivot_edge.pivot_split.bipartition()} (visit {current_pivot_edge.visits})"
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
        jt_logger.section("Lattice Algorithm Execution")
        jt_logger.log_newick_strings(input_tree1, input_tree2)

        lattice_solutions_manager = LatticeSolutions()

        current_pivot_edges: List[PivotEdgeSubproblem] | None = construct_sublattices(
            input_tree1, input_tree2
        )

        jt_logger.subsection("Initial Sub-Lattices")

        current_pivot_edges = sort_lattice_edges_by_subset_hierarchy(
            current_pivot_edges, input_tree1, input_tree2
        )

        solve_lattice_edges(
            current_pivot_edges, lattice_solutions_manager, input_tree1, input_tree2
        )

        # solution_sets_list will be List[List[Partition]]
        solution_sets_list: List[List[Partition]] = []
        pivot_edges_of_solutions_list: List[Partition] = []

        # Group solutions by pivot_edge (ignoring visit) to find all minimal solutions
        # We rank by (fewest elements, smallest partition sizes lexicographically), then deterministic bitmask
        pivot_edge_to_solutions: defaultdict[
            Partition, List[Tuple[PartitionSet[Partition], tuple, int]]
        ] = defaultdict(list)

        for (
            pivot_edge_partition,  # This is a Partition object
            visit,
        ), _ in lattice_solutions_manager.solutions_for_s_edge.items():
            # Get the single smallest solution for this visit
            smallest_solution: PartitionSet[Partition] | None = (
                lattice_solutions_manager.get_single_smallest_solution(
                    pivot_edge_partition, visit
                )
            )

            if smallest_solution:

                def _popcount(x: int) -> int:
                    try:
                        return x.bit_count()  # Python 3.8+
                    except AttributeError:
                        return bin(x).count("1")

                num_parts = len(smallest_solution)
                sizes_tuple = tuple(
                    sorted((_popcount(p.bitmask) for p in smallest_solution))
                )
                mask_tuple = tuple(sorted((p.bitmask for p in smallest_solution)))
                rank_key = (num_parts, sizes_tuple, mask_tuple)
                pivot_edge_to_solutions[pivot_edge_partition].append(
                    (smallest_solution, rank_key, visit)
                )

        # For each pivot_edge, keep one smallest solution per visit (no cross-visit filtering)
        for pivot_edge_partition, solutions in pivot_edge_to_solutions.items():
            kept = 0
            for solution, _rank_key, visit in solutions:
                # Convert PartitionSet to List[Partition]
                selected_solution_set: List[Partition] = list(solution)
                solution_sets_list.append(selected_solution_set)
                pivot_edges_of_solutions_list.append(pivot_edge_partition)
                kept += 1

            # Diagnostics: report kept per edge
            try:
                jt_logger.info(
                    f"[lattice] Edge {pivot_edge_partition}: visits={len(solutions)} kept={kept}"
                )
            except Exception:
                pass

        return solution_sets_list, pivot_edges_of_solutions_list

    except Exception as e:
        from brancharchitect.jumping_taxa.debug import log_stacktrace

        log_stacktrace(e)
        raise Exception(f"Error in lattice_algorithm: {str(e)}")
