from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.registry import (
    compute_solution_rank_key,
)


class TestSolutionRankingChange:
    def test_ranking_prioritizes_fewer_partitions_over_fewer_taxa(self):
        """
        Test that a solution with fewer partitions is preferred over a solution
        with fewer total taxa.

        Sol A: {A, B, C} (1 partition, 3 taxa)
        Sol B: {D}, {E} (2 partitions, 2 taxa)

        New Logic: Sol A should be ranked better (lower key) than Sol B.
        """
        encoding = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

        # Solution A: 1 partition, 3 taxa
        sol_a = PartitionSet(
            {Partition((1, 2, 3), encoding=encoding)}, encoding=encoding
        )

        # Solution B: 2 partitions, 2 taxa
        sol_b = PartitionSet(
            {Partition((4,), encoding=encoding), Partition((5,), encoding=encoding)},
            encoding=encoding,
        )

        rank_a = compute_solution_rank_key(sol_a)
        rank_b = compute_solution_rank_key(sol_b)

        # rank_a should be "smaller" (better) than rank_b
        # rank structure: (num_partitions, total_taxa, ...)

        assert rank_a < rank_b

        # Verify specific values
        # A: (1, 3, ...)
        assert rank_a[0] == 1
        assert rank_a[1] == 3

        # B: (2, 2, ...)
        assert rank_b[0] == 2
        assert rank_b[1] == 2
