from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.registry import SolutionRegistry


class TestSolutionSelection:
    def test_select_best_solutions_parsimony(self):
        """Test that the selector picks the most parsimonious solution."""
        registry = SolutionRegistry()
        pivot = Partition((1, 2, 3), encoding={"A": 1, "B": 2, "C": 3})

        # Solution 1: 3 partitions (worse)
        sol1 = PartitionSet(
            {
                Partition((1,), encoding=pivot.encoding),
                Partition((2,), encoding=pivot.encoding),
                Partition((3,), encoding=pivot.encoding),
            },
            encoding=pivot.encoding,
        )

        # Solution 2: 1 partition (better)
        sol2 = PartitionSet(
            {Partition((1, 2, 3), encoding=pivot.encoding)}, encoding=pivot.encoding
        )

        registry.add_solutions(pivot, [sol1], "test", visit=1)
        registry.add_solutions(pivot, [sol2], "test", visit=2)

        results = registry.select_best_solutions()

        assert pivot in results
        # Should pick sol2 (length 1) over sol1 (length 3)
        assert len(results[pivot]) == 1
        assert results[pivot][0] == list(sol2)[0]

    def test_select_best_solutions_tie_breaking(self):
        """Test deterministic tie-breaking based on partition sizes/bitmasks."""
        registry = SolutionRegistry()
        pivot = Partition((1, 2), encoding={"A": 1, "B": 2})

        # Solution A: Partition((1,))
        sol_a = PartitionSet(
            {Partition((1,), encoding=pivot.encoding)}, encoding=pivot.encoding
        )

        # Solution B: Partition((2,))
        sol_b = PartitionSet(
            {Partition((2,), encoding=pivot.encoding)}, encoding=pivot.encoding
        )

        registry.add_solutions(pivot, [sol_b], "test", visit=1)
        registry.add_solutions(pivot, [sol_a], "test", visit=2)

        results = registry.select_best_solutions()

        assert pivot in results
        assert len(results[pivot]) == 1

        # sol_a (bitmask 2) should come before sol_b (bitmask 4)
        assert results[pivot][0].bitmask == 2
