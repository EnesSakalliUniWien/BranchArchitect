from __future__ import annotations
from brancharchitect.tree import Node
from typing import List, Tuple, Dict
from collections import defaultdict

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)

from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.registry import (
    SolutionRegistry,
    compute_solution_rank_key,
)

# Import lattice modules
from brancharchitect.jumping_taxa.lattice.build_pivot_lattices import (
    construct_sublattices,
    build_conflict_matrix,
)

# Sort pivot edges by depth-based hierarchy for optimal processing order
from brancharchitect.jumping_taxa.lattice.edge_depth_ordering import (
    sort_pivot_edges_by_subset_hierarchy,
)
from brancharchitect.jumping_taxa.lattice.meet_product_solvers import (
    split_matrix,
    union_split_matrix_results,
    generalized_meet_product,
)
from brancharchitect.jumping_taxa.lattice.mapping.iterative_pivot_mappings import (
    map_iterative_pivot_edges_to_original,
)


def solve_pivot_edges(
    sub_lattices: List[PivotEdgeSubproblem],
    solution_registry: SolutionRegistry,
    t1: Node,
    t2: Node,
    stop_after_first_solution: bool = True,
) -> None:
    """
    Process a set of pivot edge subproblems to find solutions.

    For each edge, builds a conflict matrix, solves it using meet products,
    and stores solutions. Edges can be re-processed after solution removal
    to find nested solutions.

    Args:
        sub_lattices: List of pivot subproblems to process
        solution_registry: SolutionRegistry for storing solutions
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

        jt_logger.info(
            f"Processing edge: {current_pivot_edge.pivot_split} at visit {current_pivot_edge.visits}"
        )

        # Build conflict matrix (decision logic inside build_conflict_matrix)
        candidate_matrix = build_conflict_matrix(current_pivot_edge)

        # Determine which solver to use based on matrix type
        matrices = split_matrix(candidate_matrix)
        jt_logger.section("Meet Result Computation")

        # Run the appropriate solver based on number of matrices
        if len(matrices) > 1:
            solutions = union_split_matrix_results(matrices)
        else:
            solutions = generalized_meet_product(matrices[0])

        # Store solutions for this edge and visit
        jt_logger.info(
            f"Adding solutions for {current_pivot_edge.pivot_split} at visit {current_pivot_edge.visits}"
        )

        jt_logger.info(f"Solutions: {solutions}")

        solution_registry.add_solutions(
            current_pivot_edge.pivot_split,
            solutions,
            category="solution",
            visit=current_pivot_edge.visits,
        )

        # Retrieve solutions for this visit to potentially re-process
        solutions_for_visit: List[PartitionSet[Partition]] = (
            solution_registry.get_solutions_for_edge_visit(
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
                    f"[solve_pivot_edges] Stopping after first solution for pivot "
                    f"{current_pivot_edge.pivot_split.bipartition()} (visit {current_pivot_edge.visits})"
                )


@jt_logger.log_execution
def lattice_algorithm(
    input_tree1: Node,
    input_tree2: Node,
    original_tree1: Node | None = None,
    original_tree2: Node | None = None,
) -> Dict[Partition, List[Partition]]:
    """Execute the lattice algorithm to find jumping taxa between two trees.

    Args:
        input_tree1: First tree (current iteration, possibly pruned)
        input_tree2: Second tree (current iteration, possibly pruned)
        original_tree1: Original unpruned tree 1 (for mapping pivot edges).
                       If None, uses input_tree1 as original.
        original_tree2: Original unpruned tree 2 (for mapping pivot edges).
                       If None, uses input_tree2 as original.

    Returns:
        Dictionary mapping pivot edges (mapped to original trees)
        to their flat solution partitions: {pivot_edge: [partition_1, partition_2, ...]}

    Raises:
        ValueError: If sublattices cannot be constructed from input trees
    """
    jt_logger.section("Lattice Algorithm Execution")
    jt_logger.log_newick_strings(input_tree1, input_tree2)

    # Default to input trees if original trees not provided
    if original_tree1 is None:
        original_tree1 = input_tree1
    if original_tree2 is None:
        original_tree2 = input_tree2

    # 1. Construct sublattices and solve pivot edges
    solution_registry = SolutionRegistry()
    current_pivot_edges = construct_sublattices(input_tree1, input_tree2)

    if not current_pivot_edges:
        raise ValueError(
            "Failed to construct sublattices from input trees. "
            "Trees may be identical or invalid."
        )

    jt_logger.subsection("Initial Sub-Lattices")

    current_pivot_edges = sort_pivot_edges_by_subset_hierarchy(
        current_pivot_edges, input_tree1, input_tree2
    )

    solve_pivot_edges(current_pivot_edges, solution_registry, input_tree1, input_tree2)
    # Build dictionary mapping pivot edges to their solution partitions (flattened)
    solutions_dict: Dict[Partition, List[Partition]] = {}

    # Group solutions by pivot_edge (ignoring visit) to find all minimal solutions
    # We rank by (fewest elements, smallest partition sizes lexicographically), then deterministic bitmask
    pivot_edge_to_solutions: defaultdict[
        Partition,
        List[
            Tuple[
                PartitionSet[Partition],
                Tuple[int, Tuple[int, ...], Tuple[int, ...]],
                int,
            ]
        ],
    ] = defaultdict(list)

    for (
        pivot_edge_partition,  # This is a Partition object
        visit,
    ), _ in solution_registry.solutions_by_pivot_and_iteration.items():
        # Get the single smallest solution for this visit
        smallest_solution = solution_registry.get_single_smallest_solution(
            pivot_edge_partition, visit
        )

        if smallest_solution is None:
            raise ValueError(
                f"No solution found for pivot edge {pivot_edge_partition} at visit {visit}. "
                "This indicates a corrupted solution registry."
            )

        # Compute ranking key for solution comparison
        rank_key = compute_solution_rank_key(smallest_solution)
        pivot_edge_to_solutions[pivot_edge_partition].append(
            (smallest_solution, rank_key, visit)
        )

    # For each pivot_edge, filter by parsimony: keep only best-ranked solutions
    for pivot_edge_partition, solutions in pivot_edge_to_solutions.items():
        # Sort by rank, then choose a single parsimonious solution deterministically
        solutions.sort(key=lambda x: x[1])
        best_solution_set: PartitionSet[Partition] = solutions[0][0]

        # Flatten the chosen solution set into a list of partitions in deterministic order
        flat_partitions: List[Partition] = list(sorted(best_solution_set, key=lambda p: (len(p.indices), p.bitmask)))

        solutions_dict[pivot_edge_partition] = flat_partitions

    # 3. Map pivot edges to original trees
    return _map_solutions_to_original_trees(
        solutions_dict, original_tree1, original_tree2, input_tree1, input_tree2
    )


def _map_solutions_to_original_trees(
    solutions_dict: Dict[Partition, List[Partition]],
    original_tree1: Node,
    original_tree2: Node,
    current_tree1: Node,
    current_tree2: Node,
) -> Dict[Partition, List[Partition]]:
    """
    Map pivot edges from pruned trees to their corresponding splits in original trees.

    Args:
        solutions_dict: Flat solution partitions keyed by pivot edges from pruned trees
        original_tree1: Original unpruned tree 1
        original_tree2: Original unpruned tree 2
        current_tree1: Current pruned tree 1
        current_tree2: Current pruned tree 2

    Returns:
        Flat solution partitions keyed by pivot edges mapped to original trees
    """
    jt_logger.info("[lattice] Mapping pivot edges to original trees...")

    pivot_edges_list = list(solutions_dict.keys())
    solutions_list = [solutions_dict[pivot] for pivot in pivot_edges_list]

    mapped_pivot_edges = map_iterative_pivot_edges_to_original(
        pivot_edges_list,
        original_tree1,
        original_tree2,
        current_tree1,
        current_tree2,
        solutions_list,
    )

    # Build new dictionary with mapped pivot edges
    # Note: map_iterative_pivot_edges_to_original always returns valid Partition objects
    mapped_solutions_dict = {
        mapped_pivot: solutions_dict[pivot_edge]
        for pivot_edge, mapped_pivot in zip(pivot_edges_list, mapped_pivot_edges)
    }

    jt_logger.info(
        f"[lattice] Mapped {len(pivot_edges_list)} pivot edges to original trees"
    )

    return mapped_solutions_dict
