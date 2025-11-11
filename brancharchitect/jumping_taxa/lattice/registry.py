from __future__ import annotations
from typing import Dict, List, Tuple, Union, Sequence
from brancharchitect.elements.partition_set import Partition, PartitionSet
from brancharchitect.jumping_taxa.debug import jt_logger

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
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute ranking key for solution comparison.

    Solutions are ranked by:
    1. Fewest partitions (simpler = fewer jumping taxa groups)
    2. Smallest partition sizes (smaller = fewer taxa per group)
    3. Deterministic bitmask ordering (reproducibility)

    Args:
        solution: PartitionSet to rank

    Returns:
        Tuple of (num_partitions, sorted_sizes, sorted_bitmasks)
        Lower values indicate "better" (simpler) solutions

    Example:
        >>> sol1 = PartitionSet([Partition({A}), Partition({B}), Partition({C})])
        >>> sol2 = PartitionSet([Partition({A,B,C})])
        >>> compute_solution_rank_key(sol1)  # (3, (1,1,1), (...))
        >>> compute_solution_rank_key(sol2)  # (1, (3,), (...))
        >>> # sol2 ranks better: fewer partitions
    """
    num_parts = len(solution)
    sizes_tuple = tuple(sorted((p.bitmask.bit_count() for p in solution)))
    mask_tuple = tuple(sorted((p.bitmask for p in solution)))
    return (num_parts, sizes_tuple, mask_tuple)


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
        jt_logger.info(f"[get_solutions_for_edge_visit] pivot_edge={pivot_edge}, visit={visit}")
        jt_logger.info(f"[get_solutions_for_edge_visit] solutions dict: {solutions}")
        jt_logger.info(
            f"[get_solutions_for_edge_visit] dict keys: {list(solutions.keys())}"
        )
        result: List[PartitionSet[Partition]] = []
        for sols in solutions.values():
            jt_logger.info(
                f"[get_solutions_for_edge_visit] Extending with sols: {sols}, type: {type(sols)}"
            )
            result.extend(sols)
        jt_logger.info(
            f"[get_solutions_for_edge_visit] Final result length: {len(result)}"
        )
        jt_logger.info(f"[get_solutions_for_edge_visit] Final result: {result}")
        return result

    def minimal_by_indices_sum(
        self, solutions: List[PartitionSet[Partition]]
    ) -> List[PartitionSet[Partition]]:
        """
        Find solutions with the minimal sum of indices lengths.

        Args:
            solutions: List of solutions to analyze

        Returns:
            List of solutions with the minimal indices sum
        """
        if not solutions:
            return []

        # Calculate indices sum for each solution
        def calculate_indices_sum(solution: PartitionSet[Partition]) -> int:
            """Calculate the sum of lengths of all partition indices in the solution."""
            return sum(len(partition.indices) for partition in solution)

        try:
            # Find the minimum sum
            min_sum = min(calculate_indices_sum(s) for s in solutions)

            # Return solutions with the minimum sum
            return [s for s in solutions if calculate_indices_sum(s) == min_sum]
        except Exception as e:
            # Log error and return empty list
            print(f"Error in minimal_by_indices_sum: {e}")
            return []

    def get_minimal_by_indices_sum(
        self, s_edge: Partition, visit: int
    ) -> List[PartitionSet[Partition]]:
        """
        Get solutions for a specific edge and visit with minimal indices sum.

        This is a convenience method that combines get_solutions_for_edge_visit and minimal_by_indices_sum.

        Args:
            s_edge: The edge to query
            visit: The visit number

        Returns:
            List of solutions with minimal indices sum
        """
        solutions: List[PartitionSet[Partition]] = self.get_solutions_for_edge_visit(
            s_edge, visit
        )
        return self.minimal_by_indices_sum(solutions)

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

        def solution_key(
            sol: PartitionSet[Partition],
        ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
            def _popcount(x: int) -> int:
                try:
                    return x.bit_count()
                except AttributeError:
                    return bin(x).count("1")

            num_parts = len(sol)
            sizes_tuple = tuple(sorted((_popcount(p.bitmask) for p in sol)))
            mask_tuple = tuple(sorted((p.bitmask for p in sol)))
            return (num_parts, sizes_tuple, mask_tuple)

        return min(solutions, key=solution_key)

    def get_solutions_grouped_by_visit_and_edge(
        self,
    ) -> Dict[int, Dict[Partition, List[PartitionSet[Partition]]]]:
        """
        Group all solutions first by iteration, then by pivot edge.

        Returns:
            Dictionary mapping iteration numbers to dictionaries mapping pivot edges to lists of solutions
        """
        grouped: Dict[int, Dict[Partition, List[PartitionSet[Partition]]]] = {}
        for (
            pivot_edge,
            iteration,
        ), category_solutions in self.solutions_by_pivot_and_iteration.items():
            for sols in category_solutions.values():
                grouped.setdefault(iteration, {}).setdefault(pivot_edge, []).extend(
                    sols
                )
        return grouped

    def get_solutions_by_edge(self) -> Dict[Partition, List[Partition]]:
        """
        Get all solutions organized by pivot edge for use in tree interpolation.

        Flattens iteration and category structure to provide a simple mapping
        from each pivot edge to a flat list of solution partitions.

        Returns:
            Dictionary mapping pivot edges to partitions involved in solutions:
            {pivot_edge: [partition_1, partition_2, ...]}
        """
        result: Dict[Partition, List[Partition]] = {}

        for (
            pivot_edge,
            iteration,
        ), category_solutions in self.solutions_by_pivot_and_iteration.items():
            for sols in category_solutions.values():
                for solution_set in sols:
                    for part in solution_set:
                        result.setdefault(pivot_edge, []).append(part)

        return result
