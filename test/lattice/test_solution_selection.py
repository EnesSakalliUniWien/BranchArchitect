
import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.registry import SolutionRegistry
from brancharchitect.jumping_taxa.lattice.solvers.pivot_edge_solver import _select_best_solutions

class TestSolutionSelection:
    def test_select_best_solutions_parsimony(self):
        """Test that the selector picks the most parsimonious solution."""
        registry = SolutionRegistry()
        pivot = Partition((1, 2, 3), encoding={"A": 1, "B": 2, "C": 3})

        # Solution 1: 3 partitions (worse)
        sol1 = PartitionSet({
            Partition((1,), encoding=pivot.encoding),
            Partition((2,), encoding=pivot.encoding),
            Partition((3,), encoding=pivot.encoding)
        }, encoding=pivot.encoding)

        # Solution 2: 1 partition (better)
        sol2 = PartitionSet({
            Partition((1, 2, 3), encoding=pivot.encoding)
        }, encoding=pivot.encoding)

        registry.add_solutions(pivot, [sol1], "test", visit=1)
        registry.add_solutions(pivot, [sol2], "test", visit=2) # Different visit to simulate finding a better one later

        # We need to simulate the registry having multiple visits for the same pivot
        # The selector iterates over (pivot, visit) keys.
        # But wait, the selector logic in _select_best_solutions iterates over ALL (pivot, visit) pairs.
        # Then it groups them by pivot.

        results = _select_best_solutions(registry)

        assert pivot in results
        # Should pick sol2 (length 1) over sol1 (length 3)
        assert len(results[pivot]) == 1
        assert results[pivot][0] == list(sol2)[0]

    def test_select_best_solutions_tie_breaking(self):
        """Test deterministic tie-breaking based on partition sizes/bitmasks."""
        registry = SolutionRegistry()
        pivot = Partition((1, 2), encoding={"A": 1, "B": 2})

        # Solution A: Partition((1,))
        sol_a = PartitionSet({Partition((1,), encoding=pivot.encoding)}, encoding=pivot.encoding)

        # Solution B: Partition((2,))
        sol_b = PartitionSet({Partition((2,), encoding=pivot.encoding)}, encoding=pivot.encoding)

        # Both have 1 partition of size 1. Tie break on bitmask.
        # (1,) has bitmask 2^1 = 2
        # (2,) has bitmask 2^2 = 4
        # Lower bitmask should win?
        # The code sorts by `p.bitmask`.

        registry.add_solutions(pivot, [sol_b], "test", visit=1)
        registry.add_solutions(pivot, [sol_a], "test", visit=2)

        results = _select_best_solutions(registry)

        # logic: solutions.sort(key=lambda x: x[1])
        # x[1] is rank_key = (num_parts, sorted_sizes, sorted_bitmasks)

        assert pivot in results
        assert len(results[pivot]) == 1

        # sol_a (bitmask 2) should come before sol_b (bitmask 4)
        assert results[pivot][0].bitmask == 2
