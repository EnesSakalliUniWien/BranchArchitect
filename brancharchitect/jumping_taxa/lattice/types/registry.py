from __future__ import annotations
from typing import Dict, List, Tuple, Union, Sequence
from brancharchitect.elements.partition_set import Partition, PartitionSet
from brancharchitect.logger import jt_logger

"""
Solution registry for the lattice algorithm.

This module provides SolutionRegistry, which stores and manages jumping taxa
solutions organized by pivot edge and iteration number.
"""


# ============================================================================
# Utility Functions
# ============================================================================


def compute_solution_rank_key(
    solution: PartitionSet[Partition],
) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute ranking key for solution comparison.

    Solutions are ranked by (in order of priority):
    1. Fewest partitions (simpler = fewer jumping taxa groups/subtrees)
    2. Smallest total taxa count (sum of partition sizes) - Maximum Parsimony
    3. Smallest individual partition sizes (tie-breaker)
    4. Deterministic bitmask ordering (reproducibility)

    Args:
        solution: PartitionSet to rank

    Returns:
        Tuple of (num_partitions, total_taxa, sorted_sizes, sorted_bitmasks)
        Lower values indicate "better" (more parsimonious) solutions

    Example:
        >>> sol1 = PartitionSet([Partition({A}), Partition({B}), Partition({C})])
        >>> sol2 = PartitionSet([Partition({A,B,C})])
        >>> compute_solution_rank_key(sol1)  # (3, 3, (1,1,1), (...))
        >>> compute_solution_rank_key(sol2)  # (1, 3, (3,), (...))
        >>> # sol2 has fewer partitions (1 vs 3), so sol2 ranks better
    """
    num_parts = len(solution)
    sizes_tuple = tuple(sorted(p.size for p in solution))
    total_taxa = sum(sizes_tuple)
    mask_tuple = tuple(sorted((p.bitmask for p in solution)))
    return (num_parts, total_taxa, sizes_tuple, mask_tuple)


class SolutionRegistry:
    """
    Registry for storing and querying jumping taxa solutions.

    This class manages solutions discovered by the lattice algorithm,
    organizing them by pivot edge (split), iteration number, and category.
    It provides methods to add solutions and retrieve them in various formats.

    The registry supports multiple iterations of the lattice algorithm,
    with each iteration potentially discovering new solutions for the same
    or different pivot edges.
    """

    def __init__(self):
        """
        Initialize a new SolutionRegistry instance.

        Solutions are stored in a dictionary keyed by (pivot_edge, iteration) tuples,
        with values being dictionaries mapping categories to lists of solutions.
        """
        self.solutions_by_pivot_and_iteration: Dict[
            Tuple[Partition, int], Dict[str, List[PartitionSet[Partition]]]
        ] = {}

    def add_solution(
        self,
        pivot_edge: Partition,
        solution: Union[PartitionSet[Partition], Partition],
        category: str,
        visit: int,
    ) -> None:
        """
        Add a single solution.

        Args:
            s_edge: The edge this solution belongs to
            solution: The solution to add (PartitionSet or Partition)
            category: Category label for the solution (e.g., 'degenerate')
            visit: Visit number when this solution was found

        Raises:
            ValueError: If solution is not a PartitionSet or Partition
        """
        key: Tuple[Partition, int] = (pivot_edge, visit)

        if isinstance(solution, PartitionSet):
            # IMPORTANT: Copy the solution to avoid it being modified by remove_solutions_from_covers
            solution_copy = solution.copy()
            self.solutions_by_pivot_and_iteration.setdefault(key, {}).setdefault(
                category, []
            ).append(solution_copy)
        else:
            # Wrap Partition in a PartitionSet
            wrapped: PartitionSet[Partition] = PartitionSet(
                {solution}, encoding=solution.encoding, name=f"wrapped_{category}"
            )
            self.solutions_by_pivot_and_iteration.setdefault(key, {}).setdefault(
                category, []
            ).append(wrapped)

    def add_solutions(
        self,
        pivot_edge: Partition,
        solutions: Sequence[Union[PartitionSet[Partition], Partition, set[Partition]]],
        category: str,
        visit: int = 0,
    ) -> None:
        """
        Add multiple solutions.

        Args:
            s_edge: The edge these solutions belong to
            solutions: List of solutions to add
            category: Category label for the solutions
            visit: Visit number when these solutions were found
        """
        for sol in solutions:
            # Handle regular Python sets
            if isinstance(sol, set) and not isinstance(sol, PartitionSet):
                # Try to find a lookup from the edge
                encoding = getattr(pivot_edge, "encoding", {})
                sol_partitionset: PartitionSet[Partition] = PartitionSet(
                    sol, encoding=encoding, name=f"converted_{category}"
                )
                self.add_solution(pivot_edge, sol_partitionset, category, visit)
            else:
                self.add_solution(pivot_edge, sol, category, visit)

    def get_solutions_for_edge_visit(
        self, pivot_edge: Partition, visit: int
    ) -> List[PartitionSet[Partition]]:
        """
        Get all solutions for a specific edge and visit.

        Args:
            s_edge: The edge to query
            visit: The visit number

        Returns:
            List of solutions for the specified edge and visit
        """
        solutions: Dict[str, List[PartitionSet[Partition]]] = (
            self.solutions_by_pivot_and_iteration.get((pivot_edge, visit), {})
        )
        if not jt_logger.disabled:
            jt_logger.info(
                f"[get_solutions_for_edge_visit] pivot_edge={pivot_edge}, visit={visit}"
            )
            jt_logger.info(
                f"[get_solutions_for_edge_visit] solutions dict: {solutions}"
            )
            jt_logger.info(
                f"[get_solutions_for_edge_visit] dict keys: {list(solutions.keys())}"
            )
        result: List[PartitionSet[Partition]] = []
        for sols in solutions.values():
            if not jt_logger.disabled:
                jt_logger.info(
                    f"[get_solutions_for_edge_visit] Extending with sols: {sols}, type: {type(sols)}"
                )
            result.extend(sols)
        if not jt_logger.disabled:
            jt_logger.info(
                f"[get_solutions_for_edge_visit] Final result length: {len(result)}"
            )
            jt_logger.info(f"[get_solutions_for_edge_visit] Final result: {result}")
        return result

    def get_single_smallest_solution(
        self, pivot_edge: Partition, visit: int
    ) -> PartitionSet[Partition] | None:
        """
        Get the single smallest solution for a specific edge and visit.

        Selection criteria (in order):
        1) Fewest elements (partitions) in the solution
        2) Smallest partition sizes (compare sorted size tuples lexicographically)
        3) Deterministic fallback: lexicographically smallest tuple of partition bitmasks

        Args:
            s_edge: The edge to query
            visit: The visit number

        Returns:
            The single smallest solution, or None if no solutions exist
        """
        solutions = self.get_solutions_for_edge_visit(pivot_edge, visit)

        if not solutions:
            return None

        if len(solutions) == 1:
            return solutions[0]

        return min(solutions, key=compute_solution_rank_key)

    def select_best_solutions(self) -> Dict[Partition, List[Partition]]:
        """
        Select the best (most parsimonious) solutions for each pivot edge.

        For each pivot edge across all visits, finds the single most parsimonious
        solution based on ranking criteria (fewest taxa, fewest partitions, etc.)

        Returns:
            Dictionary mapping pivot edges to their flattened, sorted solution partitions.
        """
        from collections import defaultdict

        solutions_dict: Dict[Partition, List[Partition]] = {}

        # Group solutions by pivot_edge (ignoring visit) to find all minimal solutions
        pivot_edge_to_solutions: defaultdict[
            Partition,
            List[
                Tuple[
                    PartitionSet[Partition],
                    Tuple[int, int, Tuple[int, ...], Tuple[int, ...]],
                    int,
                ]
            ],
        ] = defaultdict(list)

        for (
            pivot_edge_partition,
            visit,
        ), _ in self.solutions_by_pivot_and_iteration.items():
            smallest_solution = self.get_single_smallest_solution(
                pivot_edge_partition, visit
            )

            if smallest_solution is None:
                raise ValueError(
                    f"No solution found for pivot edge {pivot_edge_partition} at visit {visit}. "
                    "This indicates a corrupted solution registry."
                )

            rank_key = compute_solution_rank_key(smallest_solution)
            pivot_edge_to_solutions[pivot_edge_partition].append(
                (smallest_solution, rank_key, visit)
            )

        # For each pivot_edge, keep only best-ranked solution
        for pivot_edge_partition, solutions in pivot_edge_to_solutions.items():
            solutions.sort(key=lambda x: x[1])
            best_solution_set: PartitionSet[Partition] = solutions[0][0]

            # Flatten into deterministically ordered list
            flat_partitions: List[Partition] = list(
                sorted(best_solution_set, key=lambda p: (len(p.indices), p.bitmask))
            )

            solutions_dict[pivot_edge_partition] = flat_partitions

        return solutions_dict

    def get_all_candidates_by_pivot(
        self,
    ) -> Dict[Partition, List[PartitionSet[Partition]]]:
        """
        Retrieve all unique candidate solutions for each pivot edge.

        Returns:
            Dictionary mapping pivot edges to a list of unique PartitionSet solutions,
            sorted by rank (best first).
        """
        from collections import defaultdict

        candidates: Dict[Partition, List[PartitionSet[Partition]]] = {}

        # Temporary storage to deduplicate by content (bitmask set)
        # Map: pivot -> set of frozen_bitmasks
        seen_solutions: defaultdict[Partition, set[tuple[int, ...]]] = defaultdict(set)

        # Map: pivot -> list of (solution, rank)
        raw_candidates: defaultdict[
            Partition,
            List[
                Tuple[
                    PartitionSet[Partition],
                    Tuple[int, int, Tuple[int, ...], Tuple[int, ...]],
                ]
            ],
        ] = defaultdict(list)

        for (pivot, visit), _ in self.solutions_by_pivot_and_iteration.items():
            visit_solutions = self.get_solutions_for_edge_visit(pivot, visit)

            for sol in visit_solutions:
                # Deduplicate based on bitmasks
                sol_mask = tuple(sorted(p.bitmask for p in sol))
                if sol_mask in seen_solutions[pivot]:
                    continue
                seen_solutions[pivot].add(sol_mask)

                rank = compute_solution_rank_key(sol)
                raw_candidates[pivot].append((sol, rank))

        # Sort and store
        for pivot, items in raw_candidates.items():
            # Sort by rank (best first)
            items.sort(key=lambda x: x[1])
            candidates[pivot] = [item[0] for item in items]

        return candidates
