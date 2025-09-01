"""
Updated lattice algorithm functions with proper Pydantic validation.
"""

from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.lattice_solution import LatticeSolutions

# Import lattice modules
from brancharchitect.jumping_taxa.lattice.lattice_construction import (
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
        solver = solve_matrix_puzzle  # Now uses union approach internally
        solver_args = {"matrix1": matrices[0], "matrix2": matrices[1]}
    else:
        solver = generalized_meet_product
        solver_args = {"matrix": matrices[0]}

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
