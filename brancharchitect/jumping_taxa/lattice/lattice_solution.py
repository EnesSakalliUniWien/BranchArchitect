from __future__ import annotations
from typing import Dict, List, Tuple, Union, Sequence
from brancharchitect.elements.partition_set import Partition, PartitionSet

"""
Simplified LatticeSolutions class with no external dependencies.
"""


class LatticeSolutions:
    """
    Storage and query manager for solutions in the lattice algorithm.

    This class stores solutions organized by edge, visit, and category.
    It provides methods to add and query solutions with a focus on
    finding minimal solutions for specific edges and visits.

    Simplified from the original implementation to focus on core functionality.
    """

    def __init__(self):
        """
        Initialize a new LatticeSolutions instance.

        Solutions are stored in a dictionary keyed by (edge, visit) tuples,
        with values being dictionaries mapping categories to lists of solutions.
        """
        self.solutions_for_s_edge: Dict[
            Tuple[Partition, int], Dict[str, List[PartitionSet[Partition]]]
        ] = {}

    def add_solution(
        self,
        s_edge: Partition,
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
        key: Tuple[Partition, int] = (s_edge, visit)

        if isinstance(solution, PartitionSet):
            self.solutions_for_s_edge.setdefault(key, {}).setdefault(
                category, []
            ).append(solution)
        else:
            # Wrap Partition in a PartitionSet
            wrapped: PartitionSet[Partition] = PartitionSet(
                {solution}, encoding=solution.encoding, name=f"wrapped_{category}"
            )
            self.solutions_for_s_edge.setdefault(key, {}).setdefault(
                category, []
            ).append(wrapped)

    def add_solutions(
        self,
        s_edge: Partition,
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
                encoding = getattr(s_edge, "encoding", {})
                sol_partitionset: PartitionSet[Partition] = PartitionSet(
                    sol, encoding=encoding, name=f"converted_{category}"
                )
                self.add_solution(s_edge, sol_partitionset, category, visit)
            else:
                self.add_solution(s_edge, sol, category, visit)

    def get_solutions_for_edge_visit(
        self, s_edge: Partition, visit: int
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
            self.solutions_for_s_edge.get((s_edge, visit), {})
        )
        result: List[PartitionSet[Partition]] = []
        for sols in solutions.values():
            result.extend(sols)
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
        self, s_edge: Partition, visit: int
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
        solutions = self.get_solutions_for_edge_visit(s_edge, visit)

        if not solutions:
            return None

        if len(solutions) == 1:
            return solutions[0]

        def solution_key(sol: PartitionSet[Partition]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
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
        Group all solutions first by visit, then by edge.

        Returns:
            Dictionary mapping visit numbers to dictionaries mapping edges to lists of solutions
        """
        grouped: Dict[int, Dict[Partition, List[PartitionSet[Partition]]]] = {}
        for (s_edge, visit), category_solutions in self.solutions_for_s_edge.items():
            for sols in category_solutions.values():
                grouped.setdefault(visit, {}).setdefault(s_edge, []).extend(sols)
        return grouped
