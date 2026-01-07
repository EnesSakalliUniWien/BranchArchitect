from __future__ import annotations
from brancharchitect.tree import Node
from typing import List, Dict, Set, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)

from brancharchitect.logger import jt_logger
from brancharchitect.logger.formatting import format_partition_set
from brancharchitect.jumping_taxa.lattice.types.registry import SolutionRegistry

# Import lattice modules
from brancharchitect.jumping_taxa.lattice.frontiers.construct_pivot_edge_problems import (
    construct_pivot_edge_problems,
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

    Encapsulates the full workflow: pivot edge construction, sorting,
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
            ValueError: If pivot edge subproblems cannot be constructed (trees may be identical)
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

        # Working copies of trees that get pruned during iterative solving
        self.current_t1: Node = self.tree1.deep_copy()
        self.current_t2: Node = self.tree2.deep_copy()

        # Build initial processing state
        self._build_processing_state()

    def _build_processing_state(self):
        """
        Build or rebuild the mutable processing state for solve().

        This constructs pivot edge subproblems from current_t1/current_t2,
        sorts them, and initializes the processing stack and registry.
        Called once in __init__ and again at the start of each iteration
        in solve_iteratively.

        Returns:
            True if there are pivot edges to process, False if trees are identical.
        """
        # Fresh registry for this solve pass
        self.registry = SolutionRegistry()

        # Construct pivot edge subproblems from current working trees
        self.pivot_edges = construct_pivot_edge_problems(
            self.current_t1, self.current_t2
        )

        if not self.pivot_edges:
            if not jt_logger.disabled:
                jt_logger.info(
                    "No pivot edge subproblems constructed. Trees are identical."
                )
            self.processing_stack = []
            return False

        # Sort by depth-based hierarchy for optimal processing
        self.pivot_edges = sort_pivot_edges_by_subset_hierarchy(
            self.pivot_edges, self.current_t1, self.current_t2
        )

        # Processing stack (initialize in reverse so pop() yields correct order)
        # We want to process from start of sorted list (subsets) to end (supersets)
        # Since stack.pop() takes from the end, we reverse the list first.
        self.processing_stack: List[PivotEdgeSubproblem] = list(
            reversed(self.pivot_edges)
        )
        return True

    def solve(self) -> Dict[Partition, List[Partition]]:
        """
        Process all pivot edge subproblems and return best solutions.

        Returns:
            Dictionary mapping pivot edges (mapped to original trees)
            to their flattened, sorted solution partitions.
        """
        if not jt_logger.disabled:
            jt_logger.subsection("Lattice Algorithm Execution")
            jt_logger.log_newick_strings(self.current_t1, self.current_t2)
            jt_logger.info("Initial Sub-Lattices")

        while self.processing_stack:
            self._process_next_pivot()

        # Solutions are already mapped to original trees during processing
        return self.registry.select_best_solutions()

    def _process_next_pivot(self) -> None:
        """Process a single pivot edge subproblem from the stack."""
        current_pivot_edge: PivotEdgeSubproblem = self.processing_stack.pop()
        if not jt_logger.disabled:
            jt_logger.info(f"Processing pivot: {current_pivot_edge.pivot_split}")

        current_pivot_edge.visits += 1
        solutions = self._solve_pivot_edge(current_pivot_edge)

        self._handle_pivot_solutions(current_pivot_edge, solutions)

    def _handle_pivot_solutions(
        self,
        current_pivot_edge: PivotEdgeSubproblem,
        solutions: List[PartitionSet[Partition]],
    ) -> None:
        """
        Handle the solutions found for a pivot edge: map them and register them.

        Note: We do NOT re-queue the pivot edge here. If there are secondary conflicts
        (e.g. overlaps hidden by nesting), they will be caught in the next iteration
        of solve_iteratively() after the current solutions are applied and trees are pruned.
        """
        # 1. Handle case with no solutions
        if not solutions:
            self.registry.add_solutions(
                current_pivot_edge.pivot_split,
                [],
                category="solution",
                visit=current_pivot_edge.visits,
            )
            return

        # 2. Process found solutions
        if not jt_logger.disabled:
            jt_logger.info(
                f"Found {len(solutions)} solutions for Pivot Split {current_pivot_edge.pivot_split}:"
            )
            jt_logger.info(
                "These solutions represent potential jumping taxa sets for this subproblem."
            )
            for i, sol in enumerate(solutions):
                jt_logger.info(f"  Solution {i + 1}: {format_partition_set(sol)}")

        # Map each solution individually to preserve specificity
        for solution in solutions:
            mapped_pivot = map_single_pivot_edge_to_original(
                current_pivot_edge.pivot_split,
                self._original_common_splits,
                [solution],
            )

            self.registry.add_solutions(
                mapped_pivot,
                [solution],
                category="solution",
                visit=current_pivot_edge.visits,
            )

    def _solve_pivot_edge(
        self, pivot_edge: PivotEdgeSubproblem
    ) -> List[PartitionSet[Partition]]:
        """
        Builds conflict matrix and solves for the given pivot edge.
        """
        # 1. Build conflict matrix
        candidate_matrix = build_conflict_matrix(pivot_edge)
        if not candidate_matrix:
            return []

        # 2. Decompose matrix into independent sub-problems
        sub_matrices = split_matrix(candidate_matrix)
        if not sub_matrices:
            return []

        if not jt_logger.disabled:
            jt_logger.info("Computing meet results...")

        # 3. Solve: Direct solve for single matrix, or union for multiple
        if len(sub_matrices) == 1:
            return generalized_meet_product(sub_matrices[0])

        return union_split_matrix_results(sub_matrices)

    def solve_iteratively(
        self,
        max_iters: int = 100,
    ) -> Tuple[Dict[Partition, List[Partition]], List[Set[int]]]:
        """
        Iteratively apply the lattice algorithm to find jumping taxa solutions.
        Returns a tuple of:
          - Dict[Partition, List[Partition]] mapping each pivot edge to a flat list of
            solution partitions (jumping taxa groups) selected by parsimony.
          - List[Set[int]] of taxa indices actually deleted in each iteration.

        Note: Only returns splits mapped to the original input trees to ensure
        usability in interpolation.

        Args:
            max_iters: Maximum number of iterations to perform to prevent infinite loops.
                      Defaults to 100.
        """
        if not jt_logger.disabled:
            jt_logger.section("Iterative Lattice Algorithm")

        # Initialize iteration variables
        jumping_subtree_solutions_dict: Dict[Partition, List[Partition]] = {}
        self.deleted_taxa_per_iteration = []  # Reset for this solve

        iteration_count = 0

        while iteration_count < max_iters:
            iteration_count += 1
            if not jt_logger.disabled:
                jt_logger.subsection(f"Iteration {iteration_count}")

            # Check if trees are now identical (using Node.__eq__ which compares full topology)
            if self.current_t1 == self.current_t2:
                break

            # Rebuild processing state for current trees
            has_work = self._build_processing_state()

            if not has_work:
                # No pivot edges means trees are topologically identical
                break

            solutions_dict_this_iter = self.solve()

            # Accumulate solutions (flat partitions) from this iteration into the global dictionary
            for split, partitions in solutions_dict_this_iter.items():
                jumping_subtree_solutions_dict.setdefault(split, []).extend(partitions)

            # Identify and delete jumping taxa
            should_break_loop = self._identify_and_delete_jumping_taxa(
                solutions_dict_this_iter, iteration_count
            )

            if should_break_loop:
                break

        # Note: Dict order may not be topologically sorted across iterations.
        # Callers should sort if needed using topological_sort_edges.
        return jumping_subtree_solutions_dict, self.deleted_taxa_per_iteration

    def _identify_and_delete_jumping_taxa(
        self,
        solutions_dict_this_iter: Dict[Partition, List[Partition]],
        iteration_count: int,
    ) -> bool:
        """
        Identify jumping taxa from solutions and delete them from trees.

        Args:
            solution_elements_this_iter: Flat list of solution partitions from this iteration
            iteration_count: Current iteration number

        Returns:
            should_break_loop: Boolean indicating if loop should break
        """

        # Collect taxa indices from all solution partitions
        taxa_indices_to_delete: Set[int] = set()

        for pivot_split, solutions in solutions_dict_this_iter.items():
            for sol_partition in solutions:
                indices = sol_partition.resolve_to_indices()
                taxa_indices_to_delete.update(indices)
                if not jt_logger.disabled:
                    jt_logger.info(
                        f"Deleting taxa {format_partition_set([sol_partition])} "
                        f"because they appear in the solution for Pivot Split {pivot_split}"
                    )

        # Check if there are any taxa to delete
        if not taxa_indices_to_delete:
            return True  # Break loop

        # Track which taxa were deleted in this iteration
        self.deleted_taxa_per_iteration.append(taxa_indices_to_delete)

        # Perform deletion (sorted for determinism)
        indices = sorted(taxa_indices_to_delete)
        self.current_t1.delete_taxa(indices)
        self.current_t2.delete_taxa(indices)

        if not jt_logger.disabled:
            jt_logger.debug(
                f"Iter {iteration_count}: Deleted {len(taxa_indices_to_delete)} taxa"
            )

        # Check if trees have enough leaves to continue
        t1_leaf_count = len(self.current_t1.get_leaves())
        t2_leaf_count = len(self.current_t2.get_leaves())
        if t1_leaf_count < 2 or t2_leaf_count < 2:
            return True  # Break loop

        return False  # Continue loop
