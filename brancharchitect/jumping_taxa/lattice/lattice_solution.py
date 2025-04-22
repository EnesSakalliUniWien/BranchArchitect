"""
Simplified LatticeSolutions class with proper Pydantic configuration.
"""
from typing import Dict, List, Tuple, Union
from brancharchitect.partition_set import Partition, PartitionSet
from pydantic import ConfigDict

class LatticeSolutions:
    """
    Storage and query manager for solutions in the lattice algorithm.
    
    This class stores solutions organized by edge, visit, and category.
    It provides methods to add and query solutions with a focus on
    finding minimal solutions for specific edges and visits.
    
    Simplified from the original implementation to focus on core functionality.
    """
    # Add Pydantic config to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Custom JSON schema generation to bypass schema errors
    @classmethod
    def __get_pydantic_json_schema__(cls, **kwargs):
        return {"title": "LatticeSolutions", "type": "object"}
    
    # Custom core schema generation to bypass pydantic-core schema errors
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema
        return core_schema.none_schema()
    
    def __init__(self):
        """
        Initialize a new LatticeSolutions instance.
        
        Solutions are stored in a dictionary keyed by (edge, visit) tuples,
        with values being dictionaries mapping categories to lists of solutions.
        """
        self.solutions_for_s_edge: Dict[Tuple[Partition, int], Dict[str, List[PartitionSet]]] = {}
    
    def add_solution(self, s_edge: Partition, solution: Union[PartitionSet, Partition], 
                     category: str, visit: int) -> None:
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
        key = (s_edge, visit)
        
        if isinstance(solution, PartitionSet):    
            self.solutions_for_s_edge.setdefault(key, {}).setdefault(category, []).append(solution)
        elif isinstance(solution, Partition):
            # Wrap Partition in a PartitionSet
            wrapped = PartitionSet({solution}, look_up=solution.lookup, name=f"wrapped_{category}")
            self.solutions_for_s_edge.setdefault(key, {}).setdefault(category, []).append(wrapped)
        else:
            raise ValueError(f"Solution must be a PartitionSet or Partition, got {type(solution)}")

    def add_solutions(self, s_edge: Partition, solutions: List, 
                      category: str, visit: int = 0) -> None:
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
                lookup = getattr(s_edge, "lookup", {})
                sol_partitionset = PartitionSet(sol, look_up=lookup, name=f"converted_{category}")
                self.add_solution(s_edge, sol_partitionset, category, visit)
            else:
                self.add_solution(s_edge, sol, category, visit)

    def get_solutions_for_edge_visit(self, s_edge: Partition, visit: int) -> List[PartitionSet]:
        """
        Get all solutions for a specific edge and visit.
        
        Args:
            s_edge: The edge to query
            visit: The visit number
        
        Returns:
            List of solutions for the specified edge and visit
        """
        solutions = self.solutions_for_s_edge.get((s_edge, visit), {})
        result = []
        for sols in solutions.values():
            result.extend(sols)
        return result

    def query_edge_visit(self, s_edge: Partition, visit: int) -> List[PartitionSet]:
        """
        Query solutions for a specific edge and visit combination.
        
        Args:
            s_edge: The edge to query
            visit: The visit number
        
        Returns:
            List of solutions for the specified edge and visit
        """
        return self.get_solutions_for_edge_visit(s_edge, visit)

    def minimal_by_indices_sum(self, solutions: List[PartitionSet]) -> List[PartitionSet]:
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
        def calculate_indices_sum(solution: PartitionSet) -> int:
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
    
    def get_minimal_by_indices_sum(self, s_edge: Partition, visit: int) -> List[PartitionSet]:
        """
        Get solutions for a specific edge and visit with minimal indices sum.
        
        This is a convenience method that combines query_edge_visit and minimal_by_indices_sum.
        
        Args:
            s_edge: The edge to query
            visit: The visit number
            
        Returns:
            List of solutions with minimal indices sum
        """
        solutions = self.query_edge_visit(s_edge, visit)
        return self.minimal_by_indices_sum(solutions)

    def get_solutions_grouped_by_visit_and_edge(self) -> Dict[int, Dict[Partition, List[PartitionSet]]]:
        """
        Group all solutions first by visit, then by edge.

        Returns:
            Dictionary mapping visit numbers to dictionaries mapping edges to lists of solutions
        """
        grouped = {}
        for (s_edge, visit), category_solutions in self.solutions_for_s_edge.items():
            for sols in category_solutions.values():
                grouped.setdefault(visit, {}).setdefault(s_edge, []).extend(sols)
        return grouped