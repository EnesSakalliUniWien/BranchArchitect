import unittest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning import (
    PivotTransitionState,
)


class TestPivotTransitionStateSiblingLogic(unittest.TestCase):
    def setUp(self):
        # Encoding: 4 taxa
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Subtrees: A, B moving. C, D context.
        self.subtree_A = Partition((0,), self.encoding)
        self.subtree_B = Partition((1,), self.encoding)

        # Parent Split P (A+B)
        self.split_P = Partition((0, 1), self.encoding)

        # Grandparent Split GP (A+B+C)
        self.split_GP = Partition((0, 1, 2), self.encoding)

        # Setup Inputs
        self.all_expand_splits = PartitionSet(
            [self.split_P, self.split_GP], encoding=self.encoding
        )

        # Initial Assignments
        # A is inside P and GP.
        self.expand_by_subtree = {
            self.subtree_A: PartitionSet([self.split_P], encoding=self.encoding),
            self.subtree_B: PartitionSet([], encoding=self.encoding),
        }

        self.dummy_pivot = Partition((0, 1, 2, 3), self.encoding)

    def test_parent_claim_logic(self):
        """Verify that subtrees claim their structural parents correctly."""

        state = PivotTransitionState(
            all_collapse_splits=PartitionSet(encoding=self.encoding),
            all_expand_splits=self.all_expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree=self.expand_by_subtree,
            pivot_edge=self.dummy_pivot,
            use_path_grouping=True,
        )

        # 1. Verify A claims P and GP (Ancestors)
        owners_P = state.expand_tracker.get_owners(self.split_P)
        owners_GP = state.expand_tracker.get_owners(self.split_GP)

        print(f"Owners of P: {owners_P}")
        print(f"Owners of GP: {owners_GP}")

        self.assertIn(self.subtree_A, owners_P)
        self.assertIn(self.subtree_A, owners_GP)

        # 2. Verify B claims P and GP (Parent Claims)
        self.assertIn(
            self.subtree_B, owners_P, "B failed to claim Parent P via ancestry!"
        )
        self.assertIn(
            self.subtree_B, owners_GP, "B failed to claim Grandparent GP via ancestry!"
        )

        # 3. Verify Grouping
        group_A = state._path_group_manager.get_group(self.subtree_A)
        group_B = state._path_group_manager.get_group(self.subtree_B)

        self.assertEqual(
            group_A, group_B, "A and B should be grouped together via Shared Parent P"
        )


if __name__ == "__main__":
    unittest.main()
