import unittest
from unittest.mock import MagicMock
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.mapping.solution_mapping import (
    _find_covering_common_splits,
)


class TestLatticeSolverCovering(unittest.TestCase):
    def test_covering_common_splits_returns_minimal_supersets(self):
        """
        Verify that _find_covering_common_splits returns minimal supersets
        of the solution partition within the common splits.
        """
        # Setup simple encoding
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Mock tree1 for encoding access
        tree1 = MagicMock(spec=Node)
        tree1.taxa_encoding = encoding
        tree2 = MagicMock(spec=Node)
        tree2.taxa_encoding = encoding

        # Scenario:
        # Target partition is {A, B}
        # Common splits include {A,B,C} and {A,B,D}; pivot edge is {A,B,C,D} (excluded).
        # Minimal supersets should be {A,B,C} and {A,B,D}.
        target_partition = Partition(frozenset({0, 1}), encoding)
        pivot_edge = Partition(frozenset({0, 1, 2, 3}), encoding)

        part_abc = Partition(frozenset({0, 1, 2}), encoding)
        part_abd = Partition(frozenset({0, 1, 3}), encoding)
        common_splits = PartitionSet({part_abc, part_abd, pivot_edge}, encoding=encoding)

        # Mock tree methods to return the mock_common_splits via intersection
        mock_node = MagicMock(spec=Node)
        mock_node.to_splits.return_value = common_splits
        tree1.find_node_by_split.return_value = mock_node
        tree2.find_node_by_split.return_value = mock_node

        # Run covering lookup
        result = _find_covering_common_splits(
            target_partition, pivot_edge, tree1, tree2
        )

        result_indices = [set(p.indices) for p in result]
        expected_indices = [{0, 1, 2}, {0, 1, 3}]

        # Sort for comparison
        result_indices.sort(key=lambda x: sorted(list(x)))
        expected_indices.sort(key=lambda x: sorted(list(x)))

        self.assertEqual(result_indices, expected_indices)
        self.assertEqual(len(result), 2)
