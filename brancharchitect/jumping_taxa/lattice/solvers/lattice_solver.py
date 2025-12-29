from __future__ import annotations
from brancharchitect.tree import Node
from typing import List, Dict, Set, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)

from brancharchitect.logger import jt_logger
from brancharchitect.jumping_taxa.lattice.types.registry import SolutionRegistry

# Import lattice modules
from brancharchitect.jumping_taxa.lattice.frontiers.build_pivot_lattices import (
    construct_sublattices,
)
from brancharchitect.jumping_taxa.lattice.matrices import (
    build_conflict_matrix,
)

# Sort pivot edges by depth-based hierarchy for optimal processing order
from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
    sort_pivot_edges_by_subset_hierarchy,
)
from brancharchitect.jumping_taxa.lattice.matrices.meet_product_solvers import (
    split_matrix,
    union_split_matrix_results,
    generalized_meet_product,
)
from brancharchitect.jumping_taxa.lattice.mapping.iterative_pivot_mappings import (
    map_single_pivot_edge_to_original,
)


class LatticeSolver:
    """
    Stateful solver for phylogenetic lattice subproblems.

    Encapsulates the full workflow: sublattice construction, sorting,
    solving, and solution selection.

    Usage:
        solver = LatticeSolver(tree1, tree2)
        solutions = solver.solve()  # Returns Dict[Partition, List[Partition]]
    """

    def __init__(
        self,
        tree1: Node,
        tree2: Node,
        original_tree1: Node | None = None,
        original_tree2: Node | None = None,
        _precomputed_original_common_splits: PartitionSet[Partition] | None = None,
    ):
        """
        Initialize solver with input trees.

        Args:
            tree1: First phylogenetic tree (current iteration, possibly pruned)
            tree2: Second phylogenetic tree (current iteration, possibly pruned)
            original_tree1: Original unpruned tree 1. If None, uses tree1.
            original_tree2: Original unpruned tree 2. If None, uses tree2.
            _precomputed_original_common_splits: Pre-computed common splits from original
                trees. Internal use only - avoids recomputation in iterative solving.

        Raises:
            ValueError: If sublattices cannot be constructed (trees may be identical)
        """
        self.tree1 = tree1
        self.tree2 = tree2

        # Set original trees (default to input trees if not provided)
        self.original_tree1 = original_tree1 if original_tree1 is not None else tree1
        self.original_tree2 = original_tree2 if original_tree2 is not None else tree2

        self.registry = SolutionRegistry()

        # Track deleted taxa per iteration (populated by solve_iteratively)
        self.deleted_taxa_per_iteration: List[Set[int]] = []

        # Pre-compute common splits from original trees ONCE (they never change)
        # Use pre-computed value if provided (for iterative solving efficiency)
        if _precomputed_original_common_splits is not None:
            self._original_common_splits = _precomputed_original_common_splits
        else:
            self._original_common_splits = (
                self.original_tree1.to_splits() & self.original_tree2.to_splits()
            )

        # Construct sublattices
        self.pivot_edges = construct_sublattices(tree1, tree2)

        if not self.pivot_edges and not jt_logger.disabled:
            jt_logger.info("No sublattices constructed. Trees may be identical.")

        # Sort by depth-based hierarchy for optimal processing
        self.pivot_edges = sort_pivot_edges_by_subset_hierarchy(
            self.pivot_edges, tree1, tree2
        )

        # Processing stack (copy to preserve original order)
        self.processing_stack: List[PivotEdgeSubproblem] = self.pivot_edges.copy()

    @jt_logger.log_execution
    def solve(
        self, stop_after_first_solution: bool = True
    ) -> Dict[Partition, List[Partition]]:
        """
        Process all pivot edge subproblems and return best solutions.

        Args:
            stop_after_first_solution: If True, stops processing a pivot after finding
                its first solution.

        Returns:
            Dictionary mapping pivot edges (mapped to original trees)
            to their flattened, sorted solution partitions.
        """
        if not jt_logger.disabled:
            jt_logger.section("Lattice Algorithm Execution")
            jt_logger.log_newick_strings(self.tree1, self.tree2)
            jt_logger.subsection("Initial Sub-Lattices")

        while self.processing_stack:
            self._process_next_pivot(stop_after_first_solution)

        # Solutions are already mapped to original trees during processing
        return self.registry.select_best_solutions()

    def _process_next_pivot(self, stop_after_first_solution: bool) -> None:
        """Process a single pivot edge subproblem from the stack."""
        current_pivot_edge: PivotEdgeSubproblem = self.processing_stack.pop()
        current_pivot_edge.visits += 1

        solutions = self._solve_pivot_edge(current_pivot_edge)

        # Store solutions for this edge and visit
        if not jt_logger.disabled:
            jt_logger.info(
                f"Processing edge: {current_pivot_edge.pivot_split} at visit {current_pivot_edge.visits}"
            )
            jt_logger.debug(
                f"Solutions for {current_pivot_edge.pivot_split} (visit {current_pivot_edge.visits}): {solutions}"
            )
            jt_logger.info(
                f"  ✓ Added {len(solutions)} solution(s) for pivot {current_pivot_edge.pivot_split.bipartition()}"
            )

        # Map pivot edge to original trees IMMEDIATELY when solutions are found
        if solutions:
            mapped_pivot = map_single_pivot_edge_to_original(
                current_pivot_edge.pivot_split,
                self._original_common_splits,
                solutions,
            )
        else:
            # No solutions - use the pivot edge as-is (will be filtered out anyway)
            mapped_pivot = current_pivot_edge.pivot_split

        self.registry.add_solutions(
            mapped_pivot,  # Use mapped pivot edge instead of pruned one
            solutions,
            category="solution",
            visit=current_pivot_edge.visits,
        )

        # Retrieve solutions for this visit to potentially re-process
        solutions_for_visit: List[PartitionSet[Partition]] = (
            self.registry.get_solutions_for_edge_visit(
                mapped_pivot, current_pivot_edge.visits
            )
        )

        if solutions_for_visit:
            current_pivot_edge.remove_solutions_from_covers(solutions_for_visit)

            # Only re-add to stack if we want to continue processing this pivot
            if not stop_after_first_solution:
                self.processing_stack.append(current_pivot_edge)
            else:
                if not jt_logger.disabled:
                    jt_logger.info(
                        f"[LatticeSolver] Stopping after first solution for pivot "
                        f"{current_pivot_edge.pivot_split.bipartition()} (visit {current_pivot_edge.visits})"
                    )

    def _solve_pivot_edge(
        self, pivot_edge: PivotEdgeSubproblem
    ) -> List[PartitionSet[Partition]]:
        """
        Builds conflict matrix and solves for the given pivot edge.
        """
        # Build conflict matrix (decision logic inside build_conflict_matrix)
        candidate_matrix = build_conflict_matrix(pivot_edge)

        # Determine which solver to use based on matrix type
        matrices = split_matrix(candidate_matrix)
        if not jt_logger.disabled:
            jt_logger.section("Meet Result Computation")

        # Run the appropriate solver based on number of matrices
        if len(matrices) > 1:
            return union_split_matrix_results(matrices)
        else:
            return generalized_meet_product(matrices[0])

    @jt_logger.log_execution
    def solve_iteratively(
        self,
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
        if not jt_logger.disabled:
            jt_logger.section("Iterative Lattice Algorithm")

        # Initialize iteration variables
        jumping_subtree_solutions_dict: Dict[Partition, List[Partition]] = {}
        self.deleted_taxa_per_iteration = []  # Reset for this solve

        # Use copies of the initial trees associated with this solver
        # The solver instance itself represents the "problem" of solving for (tree1, tree2)
        current_t1: Node = self.tree1.deep_copy()
        current_t2: Node = self.tree2.deep_copy()

        # Use the original trees stored in self
        orig_t1 = self.original_tree1
        orig_t2 = self.original_tree2

        iteration_count = 0

        while True:
            iteration_count += 1

            if not jt_logger.disabled:
                jt_logger.subsection(f"Iteration {iteration_count}")

            # Check if trees are now identical (using Node.__eq__ which compares full topology)
            if current_t1 == current_t2:
                if not jt_logger.disabled:
                    jt_logger.info("Trees are now identical - terminating iterations")
                break

            # Run lattice algorithm - it now returns a dictionary with mapped pivot edges
            # Create a new solver instance for the current iteration's trees
            # Pass pre-computed original common splits to avoid recomputation
            solver = LatticeSolver(
                current_t1,
                current_t2,
                orig_t1,
                orig_t2,
                _precomputed_original_common_splits=self._original_common_splits,
            )
            solutions_dict_this_iter = solver.solve()

            # Accumulate solutions (flat partitions) from this iteration into the global dictionary
            for split, partitions in solutions_dict_this_iter.items():
                jumping_subtree_solutions_dict.setdefault(split, []).extend(partitions)

            # For deletion, we need a flat list of partitions for this iteration
            partitions_this_iter = [
                part for parts in solutions_dict_this_iter.values() for part in parts
            ]

            # Identify and delete jumping taxa using helper function
            should_break_loop = self._identify_and_delete_jumping_taxa(
                partitions_this_iter, current_t1, current_t2, iteration_count
            )

            if should_break_loop:
                break

            if not jt_logger.disabled:
                # Summary for this iteration
                total_solutions = len(partitions_this_iter)
                jt_logger.info(
                    f"  Total: {total_solutions} jumping subtree solution(s) in this iteration"
                )

        # Note: Dict order may not be topologically sorted across iterations.
        # Callers should sort if needed using topological_sort_edges.
        return jumping_subtree_solutions_dict, self.deleted_taxa_per_iteration

    def _identify_and_delete_jumping_taxa(
        self,
        solution_elements_this_iter: List[Partition],
        current_t1: Node,
        current_t2: Node,
        iteration_count: int,
    ) -> bool:
        """
        Identify jumping taxa from solutions and delete them from trees.

        Args:
            solution_elements_this_iter: Flat list of solution partitions from this iteration
            current_t1: Current state of first tree (modified in-place)
            current_t2: Current state of second tree (modified in-place)
            iteration_count: Current iteration number

        Returns:
            should_break_loop: Boolean indicating if loop should break
        """
        # Collect taxa indices from all solution partitions
        taxa_indices_to_delete: Set[int] = set()

        for sol_partition in solution_elements_this_iter:
            taxa_indices_to_delete.update(sol_partition.resolve_to_indices())

        # Check if there are any taxa to delete
        if not taxa_indices_to_delete:
            if not jt_logger.disabled:
                jt_logger.warning(
                    f"Iter {iteration_count}: Solutions found, but no taxa indices to delete. Breaking."
                )
            return True  # Break loop

        # Track which taxa were deleted in this iteration
        self.deleted_taxa_per_iteration.append(taxa_indices_to_delete)

        # Perform deletion
        current_t1.delete_taxa(list(taxa_indices_to_delete))
        current_t2.delete_taxa(list(taxa_indices_to_delete))

        if not jt_logger.disabled:
            jt_logger.debug(
                f"Iter {iteration_count}: Deleted {len(taxa_indices_to_delete)} taxa"
            )

        # Check if trees have enough leaves to continue
        t1_leaf_count = len(current_t1.get_leaves())
        t2_leaf_count = len(current_t2.get_leaves())
        if t1_leaf_count < 2 or t2_leaf_count < 2:
            if not jt_logger.disabled:
                jt_logger.info(
                    f"Iter {iteration_count}: One or both trees have too few leaves. Stopping."
                )
            return True  # Break loop

        return False  # Continue loop
