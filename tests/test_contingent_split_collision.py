import sys
from unittest.mock import MagicMock

sys.modules["tabulate"] = MagicMock()

import unittest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    build_edge_plan,
)
from brancharchitect.tree import Node


class TestContingentSplitCollision(unittest.TestCase):
    def setUp(self):
        # Encoding: A, B, C, D
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.pivot = Partition((0, 1, 2, 3), self.encoding)

        # Scenario:
        # Subtree A (Primary) expands Split S1.
        # Split S2 is unassigned (Contingent).
        # S2 conflicts with Split C1 (present in Collapse Tree).
        # S1 does NOT conflict with C1.

        # If S2 is treated as contingent-only, it will check "Is C1 collapsed?". No.
        # Then S2 will be skipped (Missing Branch).
        # If S2 is promoted to Primary (via Completeness), it effectively says "I need S2".
        # Logic finds conflict S2 vs C1. Adds C1 to Collapse Path.
        # Collapse Path becomes {C1}. S2 is valid.

        self.subtree_A = Partition((0,), self.encoding)

        # Split S1: A+B (Compatible with C1)
        self.s1 = Partition((0, 1), self.encoding)

        # Split S2: A+C (Incompatible with C1 if C1 is A+B... wait, A+B and A+C are incompatible)
        # Let's make C1 = A+B. S1 = A+B (Same). No conflict.
        # S2 = A+C. Conflict!

        # Collapse Tree has C1 (A+B)
        self.c1 = Partition((0, 1), self.encoding)
        self.collapse_tree = Node(split_indices=self.pivot, taxa_encoding=self.encoding)
        c1_node = Node(split_indices=self.c1, taxa_encoding=self.encoding)
        self.collapse_tree.append_child(c1_node)
        # Leafs
        c1_node.append_child(
            Node(
                split_indices=Partition((0,), self.encoding),
                taxa_encoding=self.encoding,
            )
        )
        c1_node.append_child(
            Node(
                split_indices=Partition((1,), self.encoding),
                taxa_encoding=self.encoding,
            )
        )

        # Expand Tree needs to contain S2 (A+C)
        self.expand_tree = Node(split_indices=self.pivot, taxa_encoding=self.encoding)
        # S2 = A+C
        # Child 1: A+C
        s2_node = Node(split_indices=self.s2_unassigned, taxa_encoding=self.encoding)
        self.expand_tree.append_child(s2_node)
        # Leafs for S2
        s2_node.append_child(
            Node(
                split_indices=Partition((0,), self.encoding),
                taxa_encoding=self.encoding,
            )
        )  # A
        s2_node.append_child(
            Node(
                split_indices=Partition((2,), self.encoding),
                taxa_encoding=self.encoding,
            )
        )  # C

        self.all_expand_splits = PartitionSet(
            [self.s1, self.s2_unassigned], encoding=self.encoding
        )

        # Subtree A assigned S1 only
        self.expand_splits_by_subtree = {
            self.subtree_A: PartitionSet([self.s1], encoding=self.encoding)
        }

    @property
    def s2_unassigned(self):
        return Partition((0, 2), self.encoding)

    def test_unassigned_splits_force_collapse(self):
        """Verify that unassigned split S2 creates a conflict that forces collapse of C1."""

        # Pre-check: s2 is incompatible with c1?
        # A+C vs A+B. Intersect=A. Diff1=C. Diff2=B.
        # Compatible if one contains other or disjoint.
        # Disjoint? No (A).
        # Subset? No.
        # Incompatible!

        plans = build_edge_plan(
            self.expand_splits_by_subtree,
            {},  # collapse_splits_by_subtree empty
            self.collapse_tree,
            self.expand_tree,
            self.pivot,
        )

        plan_A = plans[self.subtree_A]
        collapse_path = plan_A["collapse"]["path_segment"]
        expand_path = plan_A["expand"]["path_segment"]

        print(f"Collapse Path: {collapse_path}")
        print(f"Expand Path: {expand_path}")

        # 1. Verify S2 is in Expand Path (Completeness)
        self.assertIn(
            self.s2_unassigned,
            expand_path,
            "Unassigned Split S2 should be promoted to expand path",
        )

        # 2. Verify C1 is in Collapse Path (Incompatibility Resolution)
        # If logic failed, C1 would be missing because S1 (A+B) is compatible with C1 (A+B).
        # Only S2 (A+C) forces the collapse of C1.
        self.assertIn(
            self.c1, collapse_path, "Collapse of C1 should be forced by promoted S2"
        )


if __name__ == "__main__":
    unittest.main()
