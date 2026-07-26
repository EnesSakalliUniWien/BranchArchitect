from __future__ import annotations
import logging
import time
from itertools import product
from brancharchitect.tree import Node
from typing import List, Dict, Set, Tuple, cast

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.jumping_taxa.lattice.types.child_frontiers import ChildFrontiers

from brancharchitect.logger import jt_logger
from brancharchitect.logger.formatting import format_partition_set
from brancharchitect.jumping_taxa.lattice.types.registry import (
    SolutionRegistry,
    compute_solution_rank_key,
)

# Import lattice modules
from brancharchitect.jumping_taxa.lattice.frontiers.construct_pivot_edge_problems import (
    construct_pivot_edge_problems,
)
from brancharchitect.jumping_taxa.lattice.matrices import (
    build_conflict_matrix,
)
from brancharchitect.jumping_taxa.lattice.matrices.types import PMatrix

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

logger = logging.getLogger(__name__)


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

        # Mutable frontier stack for pivot processing (pop from end for ordered traversal)
        self.processing_stack: List[PivotEdgeSubproblem] = []

        # Build initial processing state
        self._build_processing_state()

    def _build_processing_state(self) -> bool:
        """
        Build or rebuild the mutable processing state for solve().

        This constructs pivot edge subproblems from current_t1/current_t2,
        sorts them, and initializes the processing stack and registry.
        Called once in __init__ and again at the start of each iteration
        in solve_iteratively.

        Returns:
            True if there are pivot edges to process, False if trees are identical.
        """
        t_total_start = time.perf_counter()

        # Fresh registry for this solve pass
        self.registry = SolutionRegistry()

        # Construct pivot edge subproblems from current working trees
        t_construct_start = time.perf_counter()
        self.pivot_edges = construct_pivot_edge_problems(
            self.current_t1, self.current_t2
        )
        construct_elapsed = time.perf_counter() - t_construct_start

        if not self.pivot_edges:
            if not jt_logger.disabled:
                jt_logger.info(
                    "No pivot edge subproblems constructed. Trees are identical."
                )
            self.processing_stack = []
            logger.info(
                "[PhaseTimer] lattice_build_processing_state pivots=0 construct=%.3fs total=%.3fs",
                construct_elapsed,
                time.perf_counter() - t_total_start,
            )
            return False

        # Sort by depth-based hierarchy for optimal processing
        t_sort_start = time.perf_counter()
        self.pivot_edges = sort_pivot_edges_by_subset_hierarchy(
            self.pivot_edges, self.current_t1, self.current_t2
        )
        sort_elapsed = time.perf_counter() - t_sort_start

        # Processing stack (initialize in reverse so pop() yields correct order)
        # We want to process from start of sorted list (subsets) to end (supersets)
        # Since stack.pop() takes from the end, we reverse the list first.
        self.processing_stack = list(
            reversed(self.pivot_edges)
        )
        logger.info(
            "[PhaseTimer] lattice_build_processing_state pivots=%d construct=%.3fs sort=%.3fs total=%.3fs",
            len(self.pivot_edges),
            construct_elapsed,
            sort_elapsed,
            time.perf_counter() - t_total_start,
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
        solutions = self._solve_pivot_edge(
            current_pivot_edge,
            include_top_containment=current_pivot_edge.visits == 1,
        )

        self._handle_pivot_solutions(current_pivot_edge, solutions)

    def _handle_pivot_solutions(
        self,
        current_pivot_edge: PivotEdgeSubproblem,
        solutions: List[PartitionSet[Partition]],
    ) -> None:
        """
        Handle the solutions found for a pivot edge: register them under the
        current pivot split and map only after selection.

        Candidate solution sets from one solve pass are alternatives for the
        next accepted move. Rank each move by its completed residual sequence,
        remove only the selected move from the mutable covers, and re-queue the
        pivot if residual conflicts remain.
        """
        valid_solutions = self._valid_solution_candidates(current_pivot_edge, solutions)

        if not valid_solutions:
            self.registry.add_no_solution(
                current_pivot_edge.pivot_split,
                category="solution",
                visit=current_pivot_edge.visits,
            )
            return

        # 2. Process found candidate solutions
        if not jt_logger.disabled:
            jt_logger.info(
                f"Found {len(valid_solutions)} solutions for Pivot Split {current_pivot_edge.pivot_split}:"
            )
            jt_logger.info(
                "These solutions represent potential jumping taxa sets for this subproblem."
            )
            for i, sol in enumerate(valid_solutions):
                jt_logger.info(f"  Solution {i + 1}: {format_partition_set(sol)}")

        accepted_solution = self._best_completion_sequence_for_pivot(
            current_pivot_edge, valid_solutions
        )[0]
        # CRITICAL: Snapshot the solution before adding to registry.
        # remove_solutions_from_covers() mutates frontier structures after this.
        solution_snapshot = accepted_solution.copy()

        self.registry.add_solutions(
            current_pivot_edge.pivot_split,
            [solution_snapshot],
            category="solution",
            visit=current_pivot_edge.visits,
        )

        # Remove the accepted solution from covers and re-queue if conflicts remain.
        current_pivot_edge.remove_solutions_from_covers([accepted_solution])
        if current_pivot_edge.has_remaining_conflicts():
            self.processing_stack.append(current_pivot_edge)

    def _best_completion_sequence_for_pivot(
        self,
        pivot_edge: PivotEdgeSubproblem,
        solutions: List[PartitionSet[Partition]],
        seen_states: frozenset[tuple] | None = None,
        residual_solution_cache: (
            dict[tuple, List[PartitionSet[Partition]]] | None
        ) = None,
    ) -> List[PartitionSet[Partition]]:
        """Return the best candidate sequence after recursively solving residuals."""
        if seen_states is None:
            seen_states = frozenset()
        if residual_solution_cache is None:
            residual_solution_cache = {}

        ranked_sequences: list[tuple[tuple, list[PartitionSet[Partition]]]] = []
        valid_solutions = self._valid_solution_candidates(pivot_edge, solutions)

        for solution in sorted(valid_solutions, key=compute_solution_rank_key):
            branch = self._clone_pivot_edge_subproblem(pivot_edge)
            before_state = self._pivot_frontier_state_key(branch)
            branch.remove_solutions_from_covers([solution])
            after_state = self._pivot_frontier_state_key(branch)

            sequence = [solution.copy()]
            complete = not branch.has_remaining_conflicts()

            if (
                not complete
                and after_state != before_state
                and after_state not in seen_states
            ):
                if after_state not in residual_solution_cache:
                    residual_solution_cache[after_state] = self._solve_pivot_edge(
                        branch,
                        include_top_containment=False,
                    )
                residual_solutions = residual_solution_cache[after_state]
                if residual_solutions:
                    sequence.extend(
                        self._best_completion_sequence_for_pivot(
                            branch,
                            residual_solutions,
                            seen_states | {after_state},
                            residual_solution_cache,
                        )
                    )
                    completion_probe = self._clone_pivot_edge_subproblem(branch)
                    completion_probe.remove_solutions_from_covers(sequence[1:])
                    complete = not completion_probe.has_remaining_conflicts()

            ranked_sequences.append(
                (
                    self._solution_sequence_rank_key(
                        sequence, pivot_edge.pivot_split.encoding, complete
                    ),
                    sequence,
                )
            )

        if not ranked_sequences:
            return []

        return min(ranked_sequences, key=lambda item: item[0])[1]

    def _valid_solution_candidates(
        self,
        pivot_edge: PivotEdgeSubproblem,
        solutions: List[PartitionSet[Partition]],
    ) -> List[PartitionSet[Partition]]:
        return [
            solution
            for solution in solutions
            if self._is_valid_solution_candidate(pivot_edge, solution)
        ]

    def _is_valid_solution_candidate(
        self,
        pivot_edge: PivotEdgeSubproblem,
        solution: PartitionSet[Partition],
    ) -> bool:
        if pivot_edge.pivot_split in solution:
            return False

        excluded_mask = 0
        for partition in pivot_edge.excluded_partitions:
            excluded_mask |= partition.bitmask

        solution_mask = 0
        for partition in solution:
            solution_mask |= partition.bitmask

        if not hasattr(self, "current_t1") or not hasattr(self, "current_t2"):
            return True

        trees: tuple[Node, Node] = (
            cast(Node, self.current_t1),
            cast(Node, self.current_t2),
        )

        for partition in solution:
            for tree in trees:
                node = tree.find_node_by_split(partition)
                if node is None or node.parent is None:
                    return False

        selected_mask = excluded_mask | solution_mask
        for tree in trees:
            root_mask = tree.split_indices.bitmask
            if root_mask and (root_mask & ~selected_mask) == 0:
                return False

        return True

    @staticmethod
    def _solution_sequence_rank_key(
        sequence: List[PartitionSet[Partition]],
        encoding: dict[str, int],
        complete: bool,
    ) -> tuple:
        combined: PartitionSet[Partition] = PartitionSet(encoding=encoding)
        for solution in sequence:
            combined.update(solution)

        sequence_key = tuple(
            tuple(sorted(partition.bitmask for partition in solution))
            for solution in sequence
        )
        num_partitions, total_taxa, sizes, bitmasks = compute_solution_rank_key(
            combined
        )
        return (
            0 if complete else 1,
            num_partitions,
            total_taxa,
            sizes,
            -len(sequence),
            bitmasks,
            sequence_key,
        )

    @staticmethod
    def _clone_pivot_edge_subproblem(
        pivot_edge: PivotEdgeSubproblem,
    ) -> PivotEdgeSubproblem:
        clone = PivotEdgeSubproblem(
            pivot_split=pivot_edge.pivot_split,
            tree1_node=pivot_edge.tree1_node,
            tree2_node=pivot_edge.tree2_node,
            tree1_child_frontiers=LatticeSolver._copy_child_frontiers(
                pivot_edge.tree1_child_frontiers
            ),
            tree2_child_frontiers=LatticeSolver._copy_child_frontiers(
                pivot_edge.tree2_child_frontiers
            ),
            child_subtree_splits_across_trees=(
                pivot_edge.child_subtree_splits_across_trees.copy()
            ),
            encoding=pivot_edge.encoding,
            excluded_partitions=pivot_edge.excluded_partitions.copy(),
        )
        clone.visits = pivot_edge.visits
        return clone

    @staticmethod
    def _copy_child_frontiers(
        child_frontiers_by_split: Dict[Partition, ChildFrontiers],
    ) -> Dict[Partition, ChildFrontiers]:
        return {
            split: ChildFrontiers(
                shared_top_splits=child_frontiers.shared_top_splits.copy(),
                bottom_partition_map={
                    bottom: frontiers.copy()
                    for bottom, frontiers in child_frontiers.bottom_partition_map.items()
                },
            )
            for split, child_frontiers in child_frontiers_by_split.items()
        }

    @staticmethod
    def _pivot_frontier_state_key(
        pivot_edge: PivotEdgeSubproblem,
    ) -> tuple[tuple[str, int, str, tuple[int, ...]], ...]:
        entries: list[tuple[str, int, str, tuple[int, ...]]] = []
        for side_name, child_frontiers_by_split in (
            ("tree1", pivot_edge.tree1_child_frontiers),
            ("tree2", pivot_edge.tree2_child_frontiers),
        ):
            for split, child_frontiers in sorted(
                child_frontiers_by_split.items(), key=lambda item: item[0].bitmask
            ):
                entries.append(
                    (
                        side_name,
                        split.bitmask,
                        "top",
                        tuple(
                            sorted(
                                partition.bitmask
                                for partition in child_frontiers.shared_top_splits
                            )
                        ),
                    )
                )
                for bottom, frontiers in sorted(
                    child_frontiers.bottom_partition_map.items(),
                    key=lambda item: item[0].bitmask,
                ):
                    entries.append(
                        (
                            side_name,
                            split.bitmask,
                            f"bottom:{bottom.bitmask}",
                            tuple(sorted(partition.bitmask for partition in frontiers)),
                        )
                    )
        return tuple(entries)

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
        self,
        pivot_edge: PivotEdgeSubproblem,
        include_top_containment: bool = True,
    ) -> List[PartitionSet[Partition]]:
        """
        Builds conflict matrix and solves for the given pivot edge.
        """
        # 1. Build conflict matrix
        candidate_matrix = build_conflict_matrix(
            pivot_edge,
            include_top_containment=include_top_containment,
        )
        if not candidate_matrix:
            return []

        direct_alternatives = self._direct_self_meet_alternatives(candidate_matrix)
        if direct_alternatives is not None:
            return direct_alternatives

        if self._has_direct_self_meet_rows(candidate_matrix):
            return self._solve_mixed_direct_candidate_matrix(candidate_matrix)

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

    @staticmethod
    def _has_direct_self_meet_rows(matrix: PMatrix) -> bool:
        return any(LatticeSolver._is_direct_self_meet_row(row) for row in matrix)

    @staticmethod
    def _is_direct_self_meet_row(row: list[PartitionSet[Partition]]) -> bool:
        return len(row) == 2 and row[0] == row[1]

    def _solve_mixed_direct_candidate_matrix(
        self,
        matrix: PMatrix,
    ) -> List[PartitionSet[Partition]]:
        """
        Solve a matrix that mixes overlap rows with direct witness rows.

        Direct rows [S, S] are candidate witnesses, not extra columns in the
        meet-product matrix. Solving them as ordinary rows can accidentally turn
        one overlap row plus one direct row into a 2x2 square and lose the
        overlap row's own meet witness.
        """
        component_candidates: list[list[PartitionSet[Partition]]] = []

        for component in split_matrix(matrix):
            candidates = self._mixed_component_candidates(component)
            if not candidates:
                return []
            component_candidates.append(candidates)

        if len(component_candidates) == 1:
            return component_candidates[0]

        return self._cartesian_component_candidates(component_candidates)

    def _mixed_component_candidates(
        self,
        matrix: PMatrix,
    ) -> List[PartitionSet[Partition]]:
        direct_candidates: list[PartitionSet[Partition]] = []
        meet_rows: PMatrix = []

        for row in matrix:
            if self._is_direct_self_meet_row(row):
                candidate = row[0].maximal_elements()
                if candidate:
                    direct_candidates.append(candidate)
            else:
                meet_rows.append(row)

        meet_candidates = generalized_meet_product(meet_rows) if meet_rows else []
        return self._deduplicate_solution_candidates(
            meet_candidates + direct_candidates
        )

    @staticmethod
    def _cartesian_component_candidates(
        component_candidates: list[list[PartitionSet[Partition]]],
    ) -> List[PartitionSet[Partition]]:
        final_solutions: list[PartitionSet[Partition]] = []
        seen: set[tuple[int, ...]] = set()

        for combination in product(*component_candidates):
            all_partitions: set[Partition] = set()
            for solution in combination:
                all_partitions.update(solution)

            if not all_partitions:
                continue

            encoding = next(iter(all_partitions)).encoding
            combined: PartitionSet[Partition] = PartitionSet(
                all_partitions, encoding=encoding
            )
            key = tuple(sorted(partition.bitmask for partition in combined))
            if key in seen:
                continue
            seen.add(key)
            final_solutions.append(combined)

        return sorted(final_solutions, key=compute_solution_rank_key)

    @staticmethod
    def _deduplicate_solution_candidates(
        solutions: list[PartitionSet[Partition]],
    ) -> List[PartitionSet[Partition]]:
        deduplicated: list[PartitionSet[Partition]] = []
        seen: set[tuple[int, ...]] = set()

        for solution in sorted(solutions, key=compute_solution_rank_key):
            key = tuple(sorted(partition.bitmask for partition in solution))
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(solution)

        return deduplicated

    @staticmethod
    def _direct_self_meet_alternatives(
        matrix: PMatrix,
    ) -> List[PartitionSet[Partition]] | None:
        """
        Return direct alternatives for matrices made only of self-meet rows.

        Rows of the form [S, S] are already solved witnesses. When every row is
        self-meet, the rows are alternatives, not independent constraints to
        combine by Cartesian product.
        """
        alternatives: list[PartitionSet[Partition]] = []
        seen: set[tuple[int, ...]] = set()

        for row in matrix:
            if len(row) != 2 or row[0] != row[1]:
                return None

            alternative = row[0].maximal_elements()
            if not alternative:
                continue

            key = tuple(sorted(partition.bitmask for partition in alternative))
            if key in seen:
                continue
            seen.add(key)
            alternatives.append(alternative)

        return sorted(alternatives, key=compute_solution_rank_key)

    def solve_iteratively(
        self,
        max_iters: int = 100,
    ) -> Tuple[Dict[Partition, List[Partition]], List[Set[int]]]:
        """
        Iteratively apply the lattice algorithm to find jumping taxa solutions.
        Returns a tuple of:
          - Dict[Partition, List[Partition]] mapping each pivot edge to a flat list of
            solution partitions selected by group-first ranking.
          - List[Set[int]] of taxa indices actually deleted in each iteration.

        Note: Only returns splits mapped to the original input trees to ensure
        usability in interpolation.

        Args:
            max_iters: Retained for compatibility; no longer enforced.
        """
        if not jt_logger.disabled:
            jt_logger.section("Iterative Lattice Algorithm")

        t_total_start = time.perf_counter()
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

            t_iteration_start = time.perf_counter()

            # Rebuild processing state for current trees
            t_build_start = time.perf_counter()
            has_work = self._build_processing_state()
            build_elapsed = time.perf_counter() - t_build_start
            if not has_work:
                raise RuntimeError(
                    "No pivot edges constructed but trees are not isomorphic."
                )

            t_solve_start = time.perf_counter()
            solutions_dict_this_iter = self.solve(map_solutions=False)
            solve_elapsed = time.perf_counter() - t_solve_start

            # Accumulate solutions (flat partitions) from this iteration into the global dictionary
            for split, partitions in solutions_dict_this_iter.items():
                jumping_subtree_solutions_dict.setdefault(split, []).extend(partitions)

            # Identify and delete jumping taxa
            t_delete_start = time.perf_counter()
            should_break_loop = identify_and_delete_jumping_taxa(
                self.current_t1,
                self.current_t2,
                self.deleted_taxa_per_iteration,
                solutions_dict_this_iter,
                iteration_count,
            )
            delete_elapsed = time.perf_counter() - t_delete_start
            logger.info(
                "[PhaseTimer] lattice_iteration iteration=%d pivots=%d solutions=%d build=%.3fs solve=%.3fs delete=%.3fs total=%.3fs",
                iteration_count,
                len(self.pivot_edges),
                len(solutions_dict_this_iter),
                build_elapsed,
                solve_elapsed,
                delete_elapsed,
                time.perf_counter() - t_iteration_start,
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
        t_map_start = time.perf_counter()
        mapped_solutions_dict = map_solutions_to_common_subtrees(
            jumping_subtree_solutions_dict,
            self.original_tree1,
            self.original_tree2,
        )
        logger.info(
            "[PhaseTimer] lattice_map_solutions pivots=%d %.3fs",
            len(jumping_subtree_solutions_dict),
            time.perf_counter() - t_map_start,
        )
        logger.info(
            "[PhaseTimer] lattice_solve_iteratively iterations=%d mapped_pivots=%d total=%.3fs",
            iteration_count,
            len(mapped_solutions_dict),
            time.perf_counter() - t_total_start,
        )

        return mapped_solutions_dict, self.deleted_taxa_per_iteration
