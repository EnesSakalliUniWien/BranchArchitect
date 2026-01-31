"""
Complex test cases for _find_covering_common_splits.

Tests the MRCA (Most Recent Common Ancestor) covering logic:
1. Single MRCA case: solution maps to one common subtree
2. Multiple MRCA case: solution spans multiple subtrees, needs multiple covers
3. Fallback case: no common subtree exists, falls back to leaves
"""

import unittest
from unittest.mock import MagicMock
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.mapping.solution_mapping import (
    _find_covering_common_splits,
)


class TestCoveringComplexCases(unittest.TestCase):
    def test_single_mrca_exact_match(self):
        """
        Case: Solution partition exists exactly in both trees.
        Expected: Return that exact partition as the single MRCA.

        Tree structure (conceptual):
                     ABCDEF (pivot)
                    /      \
                 ABC        DEF
                / | \      / | \
               A  B  C    D  E  F

        Solution: {A,B,C} -> should map to {A,B,C} (exact MRCA)
        """
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

        tree1 = MagicMock(spec=Node)
        tree1.taxa_encoding = encoding
        tree2 = MagicMock(spec=Node)
        tree2.taxa_encoding = encoding

        # Solution and pivot
        solution = Partition(frozenset({0, 1, 2}), encoding)  # ABC
        pivot_edge = Partition(frozenset({0, 1, 2, 3, 4, 5}), encoding)  # ABCDEF

        # Common splits (same in both trees)
        part_abc = Partition(frozenset({0, 1, 2}), encoding)
        part_def = Partition(frozenset({3, 4, 5}), encoding)
        part_a = Partition(frozenset({0}), encoding)
        part_b = Partition(frozenset({1}), encoding)
        part_c = Partition(frozenset({2}), encoding)
        part_d = Partition(frozenset({3}), encoding)
        part_e = Partition(frozenset({4}), encoding)
        part_f = Partition(frozenset({5}), encoding)

        common_splits = PartitionSet(
            {
                pivot_edge,
                part_abc,
                part_def,
                part_a,
                part_b,
                part_c,
                part_d,
                part_e,
                part_f,
            },
            encoding=encoding,
        )

        mock_node = MagicMock(spec=Node)
        mock_node.to_splits.return_value = common_splits
        tree1.find_node_by_split.return_value = mock_node
        tree2.find_node_by_split.return_value = mock_node

        result = _find_covering_common_splits(solution, pivot_edge, tree1, tree2)

        print(f"\n[Test: single_mrca_exact_match]")
        print(f"  Solution: {list(solution.indices)}")
        print(f"  Result: {[list(p.indices) for p in result]}")

        # Should return exactly {A,B,C}
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0].indices), {0, 1, 2})

    def test_multiple_mrca_spanning_subtrees(self):
        """
        Case: Solution partition spans multiple subtrees with different topology.

        Tree1:          ABCDEF
                       /      \
                    ABCD       EF
                   /    \     / \
                  AB    CD   E   F

        Tree2:          ABCDEF
                       /      \
                     AB       CDEF
                    / \      /    \
                   A   B   CD      EF

        Common splits: {AB}, {CD}, {EF}, leaves, and root (ABCDEF)

        Solution: {A,B,C,D}
        -> In Tree1: this is ABCD (exists)
        -> In Tree2: this does NOT exist as a single clade

        Common minimal covers: {AB} and {CD} (since ABCD doesn't exist in both)
        """
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

        tree1 = MagicMock(spec=Node)
        tree1.taxa_encoding = encoding
        tree2 = MagicMock(spec=Node)
        tree2.taxa_encoding = encoding

        solution = Partition(frozenset({0, 1, 2, 3}), encoding)  # ABCD
        pivot_edge = Partition(frozenset({0, 1, 2, 3, 4, 5}), encoding)  # ABCDEF

        # Tree1 splits
        t1_abcd = Partition(frozenset({0, 1, 2, 3}), encoding)
        t1_ab = Partition(frozenset({0, 1}), encoding)
        t1_cd = Partition(frozenset({2, 3}), encoding)
        t1_ef = Partition(frozenset({4, 5}), encoding)

        # Tree2 splits
        t2_ab = Partition(frozenset({0, 1}), encoding)
        t2_cdef = Partition(frozenset({2, 3, 4, 5}), encoding)
        t2_cd = Partition(frozenset({2, 3}), encoding)
        t2_ef = Partition(frozenset({4, 5}), encoding)

        # Leaves (common to both)
        leaves = [Partition(frozenset({i}), encoding) for i in range(6)]

        tree1_splits = PartitionSet(
            {pivot_edge, t1_abcd, t1_ab, t1_cd, t1_ef} | set(leaves), encoding=encoding
        )
        tree2_splits = PartitionSet(
            {pivot_edge, t2_ab, t2_cdef, t2_cd, t2_ef} | set(leaves), encoding=encoding
        )

        mock_node1 = MagicMock(spec=Node)
        mock_node1.to_splits.return_value = tree1_splits
        mock_node2 = MagicMock(spec=Node)
        mock_node2.to_splits.return_value = tree2_splits

        tree1.find_node_by_split.return_value = mock_node1
        tree2.find_node_by_split.return_value = mock_node2

        result = _find_covering_common_splits(solution, pivot_edge, tree1, tree2)

        print(f"\n[Test: multiple_mrca_spanning_subtrees]")
        print(f"  Solution: {list(solution.indices)} (ABCD)")
        print(f"  Result: {[list(p.indices) for p in result]}")

        # ABCD doesn't exist in common splits
        # Minimal covers should be {AB} and {CD}
        result_sets = [set(p.indices) for p in result]
        self.assertIn({0, 1}, result_sets)  # AB
        self.assertIn({2, 3}, result_sets)  # CD

    def test_fallback_to_leaves(self):
        """
        Case: Solution partition has no common superset except leaves.

        Tree1:          ABCD
                       /    \
                     AC      BD

        Tree2:          ABCD
                       /    \
                     AB      CD

        Common splits: only {ABCD} (pivot, excluded) and leaves {A}, {B}, {C}, {D}

        Solution: {A,B}
        -> No common internal node covers {A,B}
        -> Fallback to leaves: {A}, {B}
        """
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        tree1 = MagicMock(spec=Node)
        tree1.taxa_encoding = encoding
        tree2 = MagicMock(spec=Node)
        tree2.taxa_encoding = encoding

        solution = Partition(frozenset({0, 1}), encoding)  # AB
        pivot_edge = Partition(frozenset({0, 1, 2, 3}), encoding)  # ABCD

        # Tree1: AC, BD
        t1_ac = Partition(frozenset({0, 2}), encoding)
        t1_bd = Partition(frozenset({1, 3}), encoding)

        # Tree2: AB, CD
        t2_ab = Partition(frozenset({0, 1}), encoding)
        t2_cd = Partition(frozenset({2, 3}), encoding)

        leaves = [Partition(frozenset({i}), encoding) for i in range(4)]

        tree1_splits = PartitionSet(
            {pivot_edge, t1_ac, t1_bd} | set(leaves), encoding=encoding
        )
        tree2_splits = PartitionSet(
            {pivot_edge, t2_ab, t2_cd} | set(leaves), encoding=encoding
        )

        mock_node1 = MagicMock(spec=Node)
        mock_node1.to_splits.return_value = tree1_splits
        mock_node2 = MagicMock(spec=Node)
        mock_node2.to_splits.return_value = tree2_splits

        tree1.find_node_by_split.return_value = mock_node1
        tree2.find_node_by_split.return_value = mock_node2

        result = _find_covering_common_splits(solution, pivot_edge, tree1, tree2)

        print(f"\n[Test: fallback_to_leaves]")
        print(f"  Solution: {list(solution.indices)} (AB)")
        print(f"  Result: {[list(p.indices) for p in result]}")

        # Should fall back to leaves {A} and {B}
        result_sets = [set(p.indices) for p in result]
        self.assertEqual(len(result), 2)
        self.assertIn({0}, result_sets)  # A
        self.assertIn({1}, result_sets)  # B


if __name__ == "__main__":
    unittest.main()
