from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.registry import SolutionRegistry


class TestSolutionSelection:
    def test_select_best_solutions_prefers_fewer_jumping_groups_first(self):
        """The local objective is group-first, then total taxa."""
        registry = SolutionRegistry()
        pivot = Partition((0, 1, 2, 3), encoding={"A": 0, "B": 1, "C": 2, "D": 3})

        # Solution 1: 2 partitions and 2 total taxa.
        sol1 = PartitionSet(
            {
                Partition((0,), encoding=pivot.encoding),
                Partition((1,), encoding=pivot.encoding),
            },
            encoding=pivot.encoding,
        )

        # Solution 2: 1 partition and 3 total taxa, but not the pivot itself.
        sol2 = PartitionSet(
            {Partition((0, 1, 2), encoding=pivot.encoding)}, encoding=pivot.encoding
        )

        registry.add_solutions(pivot, [sol1, sol2], "test", visit=1)

        results = registry.select_best_solutions()

        assert pivot in results
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

        registry.add_solutions(pivot, [sol_b, sol_a], "test", visit=1)

        results = registry.select_best_solutions()

        assert pivot in results
        assert len(results[pivot]) == 1

        # sol_a (bitmask 2) should come before sol_b (bitmask 4)
        assert results[pivot][0].bitmask == 2

    def test_select_best_solutions_combines_best_solution_from_each_visit(self):
        """Repeated visits to one pivot are cumulative residual conflicts."""
        registry = SolutionRegistry()
        encoding = {"A": 0, "B": 1, "C": 2}
        pivot = Partition((0, 1, 2), encoding=encoding)

        visit_1_worse = PartitionSet(
            {
                Partition((0,), encoding=encoding),
                Partition((1,), encoding=encoding),
            },
            encoding=encoding,
        )
        visit_1_best = PartitionSet(
            {Partition((0, 1), encoding=encoding)}, encoding=encoding
        )
        visit_2_best = PartitionSet(
            {Partition((2,), encoding=encoding)}, encoding=encoding
        )

        registry.add_solutions(pivot, [visit_1_worse, visit_1_best], "test", visit=1)
        registry.add_solutions(pivot, [visit_2_best], "test", visit=2)

        results = registry.select_best_solutions()

        assert pivot in results
        assert {p.bitmask for p in results[pivot]} == {
            Partition((0, 1), encoding=encoding).bitmask,
            Partition((2,), encoding=encoding).bitmask,
        }
