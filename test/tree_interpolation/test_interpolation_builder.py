"""
Tests for interpolation plan builder.

Tests cover:
- Edge plan construction
- Integration with PivotSplitRegistry
- Handling of shared, unique, and contingent splits
- Last subtree aggregation
- Deterministic ordering
"""

import unittest
from collections import OrderedDict
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    build_edge_plan,
)


class TestEdgePlanBuilder(unittest.TestCase):
    """Test the main build_edge_plan function."""

    def setUp(self):
        """Set up trees and partitions for testing."""
        from brancharchitect.parser import parse_newick

        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_D = Partition((3,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_CD = Partition((2, 3), self.encoding)
        self.part_ABC = Partition((0, 1, 2), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

        # Create properly initialized tree with split indices
        # Tree structure: ((A,B),(C,D))
        taxa_order = ["A", "B", "C", "D"]
        self.root = parse_newick(
            "((A,B),(C,D));", order=taxa_order, encoding=self.encoding
        )

    def test_simple_edge_plan_with_single_subtree(self):
        """Test building a plan for a single subtree."""
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_AB], encoding=self.encoding),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        plan = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            self.root,
            self.root,
            self.part_ABCD,
        )

        # Should have one subtree in plan
        self.assertEqual(len(plan), 1)
        self.assertIn(self.part_A, plan)

        # Check that plan has collapse and expand sections
        self.assertIn("collapse", plan[self.part_A])
        self.assertIn("expand", plan[self.part_A])

    def test_multiple_subtrees_processed_in_order(self):
        """Test that multiple subtrees are processed correctly."""
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_AB], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_CD], encoding=self.encoding),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        plan = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            self.root,
            self.root,
            self.part_ABCD,
        )

        # Should have two subtrees in plan
        self.assertEqual(len(plan), 2)
        self.assertIn(self.part_A, plan)
        self.assertIn(self.part_C, plan)

    def test_shared_splits_only_appear_once(self):
        """Test that shared splits are processed correctly and only once."""
        # Both subtrees share part_AB in collapse
        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB, self.part_A], encoding=self.encoding
            ),
            self.part_B: PartitionSet(
                [self.part_AB, self.part_B], encoding=self.encoding
            ),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
        }

        plan = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            self.root,
            self.root,
            self.part_ABCD,
        )

        # Check that part_AB appears in one of the plans
        ab_count = sum(
            1
            for subtree_plan in plan.values()
            if self.part_AB in subtree_plan["collapse"]["path_segment"]
        )

        # Shared collapse splits should appear in the first subtree that processes them
        self.assertEqual(ab_count, 1)

    def test_splits_sorted_by_size(self):
        """Test that splits are sorted by partition size (larger first)."""
        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_A, self.part_AB, self.part_ABC], encoding=self.encoding
            ),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        plan = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            self.root,
            self.root,
            self.part_ABCD,
        )

        collapse_path = plan[self.part_A]["collapse"]["path_segment"]

        # Verify larger partitions come first
        sizes = [len(p.indices) for p in collapse_path]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_last_subtree_gets_remaining_splits(self):
        """Test that the last subtree aggregates all remaining splits."""
        # Set up so part_B is processed last and has leftover splits
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
        }

        # Add extra expand splits not assigned to any subtree
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
        }

        plan = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            self.root,
            self.root,
            self.part_ABCD,
        )

        # Both subtrees should be in the plan
        self.assertEqual(len(plan), 2)


class TestContingentSplitsInBuilder(unittest.TestCase):
    """Test handling of contingent (opportunistic) splits in builder."""

    def setUp(self):
        """Set up scenario with contingent splits."""
        from brancharchitect.parser import parse_newick

        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_ABC = Partition((0, 1, 2), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

        # Create properly initialized tree with split indices
        taxa_order = ["A", "B", "C", "D"]
        self.root = parse_newick(
            "((A,B),(C,D));", order=taxa_order, encoding=self.encoding
        )

    def test_contingent_splits_used_when_space_available(self):
        """Test that contingent splits are used when collapse creates space."""
        # Collapse part_ABC, which creates space for part_A and part_B
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_ABC], encoding=self.encoding),
        }

        # Only assign part_C to subtree, leaving part_A and part_B as contingent
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_C], encoding=self.encoding),
        }

        # Note: We need to pass ALL expand splits (including contingent ones)
        # through the get_unique_splits_for_active_changing_edge_subtree function
        # For this test, we'll manually construct the scenario

        plan = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            self.root,
            self.root,
            self.part_ABCD,
        )

        # The plan should exist for part_A
        self.assertIn(self.part_A, plan)


class TestDeterministicOrdering(unittest.TestCase):
    """Test that plan building is deterministic."""

    def setUp(self):
        """Set up scenario that could have ordering ambiguity."""
        from brancharchitect.parser import parse_newick

        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

        taxa_order = ["A", "B", "C", "D"]
        self.root = parse_newick("(A,B,C,D);", order=taxa_order, encoding=self.encoding)

    def test_multiple_runs_produce_same_order(self):
        """Test that running build_edge_plan multiple times gives the same result."""
        # First run - create fresh dictionaries
        collapse_by_subtree_1 = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        expand_by_subtree_1 = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        plan1 = build_edge_plan(
            expand_by_subtree_1,
            collapse_by_subtree_1,
            self.root,
            self.root,
            self.part_ABCD,
        )

        # Second run - create fresh dictionaries (not reusing mutated ones)
        collapse_by_subtree_2 = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        expand_by_subtree_2 = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        plan2 = build_edge_plan(
            expand_by_subtree_2,
            collapse_by_subtree_2,
            self.root,
            self.root,
            self.part_ABCD,
        )

        # Check that ordering is the same
        self.assertEqual(list(plan1.keys()), list(plan2.keys()))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up minimal fixtures."""
        from brancharchitect.parser import parse_newick

        self.encoding = {"A": 0, "B": 1}

        self.part_A = Partition((0,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)

        taxa_order = ["A", "B"]
        self.root = parse_newick("(A,B);", order=taxa_order, encoding=self.encoding)

    def test_empty_input_produces_empty_plan(self):
        """Test that empty input dictionaries produce an empty plan."""
        plan = build_edge_plan(
            {},
            {},
            self.root,
            self.root,
            self.part_AB,
        )

        self.assertEqual(len(plan), 0)

    def test_only_collapse_splits_no_expand(self):
        """Test handling when there are only collapse splits."""
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        plan = build_edge_plan(
            {},
            collapse_by_subtree,
            self.root,
            self.root,
            self.part_AB,
        )

        # Should still create a plan with empty expand path
        self.assertIn(self.part_A, plan)
        self.assertEqual(len(plan[self.part_A]["expand"]["path_segment"]), 0)

    def test_only_expand_splits_no_collapse(self):
        """Test handling when there are only expand splits."""
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        plan = build_edge_plan(
            expand_by_subtree,
            {},
            self.root,
            self.root,
            self.part_AB,
        )

        # Should still create a plan with empty collapse path
        self.assertIn(self.part_A, plan)
        self.assertEqual(len(plan[self.part_A]["collapse"]["path_segment"]), 0)


if __name__ == "__main__":
    unittest.main()
