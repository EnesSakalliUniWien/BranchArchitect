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
from brancharchitect.jumping_taxa.lattice.mapping.solution_mapping import (
    map_solutions_to_common_subtrees,
)
from brancharchitect.jumping_taxa.lattice.solvers.identify_jumping_taxa import (
    identify_and_delete_jumping_taxa,
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
        self._original_common_splits_with_leaves = self.original_tree1.to_splits(
            with_leaves=True
        ) & self.original_tree2.to_splits(with_leaves=True)

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

    def solve(self, map_solutions: bool = True) -> Dict[Partition, List[Partition]]:
        """
        Process all pivot edge subproblems and return best solutions.

        Returns:
            Dictionary mapping pivot edges (mapped to original trees)
            to their flattened, sorted solution partitions.

        Args:
            map_solutions: If True, map solution partitions to original common subtrees.
                If False, keep solutions in current-tree space (used for pruning steps).
        """
        if not jt_logger.disabled:
            jt_logger.subsection("Lattice Algorithm Execution")
            jt_logger.log_newick_strings(self.current_t1, self.current_t2)
            jt_logger.info("Initial Sub-Lattices")

        while self.processing_stack:
            self._process_next_pivot()

        selected = self.registry.select_best_solutions()
        mapped_pivots = self._map_selected_pivots(selected)

        if not map_solutions:
            return mapped_pivots
        return map_solutions_to_common_subtrees(
            mapped_pivots,
            self.original_tree1,
            self.original_tree2,
        )

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
        Handle the solutions found for a pivot edge: register them under the
        current pivot split and map only after selection.

        Note: We do NOT re-queue the pivot edge here. If there are secondary conflicts
        (e.g. overlaps hidden by nesting), they will be caught in the next iteration
        of solve_iteratively() after the current solutions are applied and trees are pruned.
        """
        if not solutions:
            self.registry.add_no_solution(
                current_pivot_edge.pivot_split,
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

        for solution in solutions:
            # CRITICAL: Snapshot the solution before adding to registry.
            # remove_solutions_from_covers() will likely be called on the *original* objects.
            # While PartitionSet internal storage (bitmasks) is usually robust,
            # explicit snapshotting protects against any mutable aliasing bugs.
            solution_snapshot = solution.copy()

            self.registry.add_solutions(
                current_pivot_edge.pivot_split,
                [solution_snapshot],
                category="solution",
                visit=current_pivot_edge.visits,
            )

        # Remove solved partitions from covers and re-queue if conflicts remain.
        current_pivot_edge.remove_solutions_from_covers(solutions)
        if current_pivot_edge.has_remaining_conflicts():
            self.processing_stack.append(current_pivot_edge)

    def _map_selected_pivots(
        self,
        selected_solutions: Dict[Partition, List[Partition]],
    ) -> Dict[Partition, List[Partition]]:
        """
        Map pivot edges to original trees after selecting best solutions.

        Uses Φ(p, best_solution) so each pivot contributes a single mapped key.
        """
        mapped: Dict[Partition, List[Partition]] = {}

        for pivot_edge, partitions in selected_solutions.items():
            mapped_pivot = map_single_pivot_edge_to_original(
                pivot_edge,
                self._original_common_splits,
                partitions,
            )

            mapped.setdefault(mapped_pivot, []).extend(partitions)

        return mapped

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
            max_iters: Retained for compatibility; no longer enforced.
        """
        if not jt_logger.disabled:
            jt_logger.section("Iterative Lattice Algorithm")

        # Initialize iteration variables
        jumping_subtree_solutions_dict: Dict[Partition, List[Partition]] = {}
        self.deleted_taxa_per_iteration = []  # Reset for this solve

        iteration_count = 0

        while True:
            if not jt_logger.disabled:
                jt_logger.subsection(f"Iteration {iteration_count + 1}")

            # Check if trees are now identical (using Node.__eq__ which compares full topology)
            if self.current_t1 == self.current_t2:
                break

            iteration_count += 1

            # Rebuild processing state for current trees
            has_work = self._build_processing_state()
            if not has_work:
                raise RuntimeError(
                    "No pivot edges constructed but trees are not isomorphic."
                )

            solutions_dict_this_iter = self.solve(map_solutions=False)

            # Accumulate solutions (flat partitions) from this iteration into the global dictionary
            for split, partitions in solutions_dict_this_iter.items():
                jumping_subtree_solutions_dict.setdefault(split, []).extend(partitions)

            # Identify and delete jumping taxa
            should_break_loop = identify_and_delete_jumping_taxa(
                self.current_t1,
                self.current_t2,
                self.deleted_taxa_per_iteration,
                solutions_dict_this_iter,
                iteration_count,
            )

            if should_break_loop:
                if self.current_t1 == self.current_t2:
                    break
                # Provide more detailed error information
                t1_splits = self.current_t1.to_splits()
                t2_splits = self.current_t2.to_splits()
                only_in_t1 = t1_splits - t2_splits
                only_in_t2 = t2_splits - t1_splits

                raise RuntimeError(
                    f"Stopping condition reached before tree isomorphism. "
                    f"Pivot edges: {len(self.pivot_edges)}, "
                    f"Solutions this iter: {len(solutions_dict_this_iter)}, "
                    f"Splits only in T1: {len(only_in_t1)}, "
                    f"Splits only in T2: {len(only_in_t2)}. "
                    f"This may indicate structural differences that cannot be resolved by the lattice algorithm."
                )

        # Note: Dict order may not be topologically sorted across iterations.
        # Callers should sort if needed using topological_sort_edges.
        # Map moving subtrees to common splits so interpolation doesn't target missing nodes.
        mapped_solutions_dict = map_solutions_to_common_subtrees(
            jumping_subtree_solutions_dict,
            self.original_tree1,
            self.original_tree2,
        )

        # verify_mapped_solutions_prune(
        #    self.original_tree1,
        #    self.original_tree2,
        #    self.current_t1,
        #    self.current_t2,
        #    mapped_solutions_dict,
        # )

        return mapped_solutions_dict, self.deleted_taxa_per_iteration
