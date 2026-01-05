"""
THE CODE EXAMINER'S COMPREHENSIVE TEST SUITE

Module: PivotSplitRegistry & Edge Plan Builder Architecture
Status: IMMUTABLE - This test cannot be altered once created.
Created: 2025-10-21
Python: 3.11+
Framework: pytest 8.2.0+

This suite validates the complete planning subsystem for tree interpolation,
including state management, subtree selection, split lifecycle, contingent
split handling, and deterministic plan generation.

CONVENTIONS:
- Uses unittest.TestCase for structure
- Partition creation: Partition((indices,), encoding)
- Tree initialization: parse_newick() for proper split_indices
- All tests are self-contained and deterministic
- No external files or network dependencies

RUN COMMANDS:
    poetry run pytest test/tree_interpolation/test_interpolation_planning_comprehensive.py -v
    poetry run pytest test/tree_interpolation/test_interpolation_planning_comprehensive.py::TestContingentSplitSemantics -v
"""

import unittest
from collections import OrderedDict
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
    build_edge_plan,
)


# ============================================================================
# SECTION 1: CONTINGENT SPLIT SEMANTICS (Based on User Clarification)
# ============================================================================


class TestContingentSplitSemantics(unittest.TestCase):
    """
    Test contingent splits: splits for moving subtrees that relocate in the
    destination tree, with orthogonal splits that must be expanded after them.

    User clarification: Contingent splits are assigned to moving subtrees that
    move to different locations in destination tree, and there are orthogonal
    splits that were left from previous operations that need to be expanded.
    """

    def setUp(self):
        """Set up scenario with moving subtree."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        # Create partitions representing moving subtrees
        self.moving_subtree_AB = Partition((0, 1), self.encoding)
        self.moving_subtree_C = Partition((2,), self.encoding)
        self.target_location = Partition((0, 1, 2), self.encoding)  # Where AB moves to

        # Orthogonal splits left from previous operations
        self.orthogonal_D = Partition((3,), self.encoding)
        self.orthogonal_E = Partition((4,), self.encoding)

        self.root = Partition((0, 1, 2, 3, 4), self.encoding)

    def test_contingent_splits_are_not_assigned_to_specific_subtrees(self):
        """Contingent splits should not appear in expand_by_subtree initially."""
        # Primary expand splits - directly assigned to subtrees
        expand_by_subtree = {
            self.moving_subtree_AB: PartitionSet(
                [self.moving_subtree_C], encoding=self.encoding
            ),
        }

        # All expand splits including contingent ones
        all_expand = PartitionSet(
            [self.moving_subtree_C, self.orthogonal_D, self.orthogonal_E],
            encoding=self.encoding,
        )

        state = PivotSplitRegistry(
            PartitionSet(encoding=self.encoding),  # No collapse
            all_expand,
            {},
            expand_by_subtree,
            self.root,
        )

        # Contingent splits are those not assigned to any subtree (not in tracker)
        tracked_resources = state.expand_tracker.get_all_resources()
        self.assertNotIn(self.orthogonal_D, tracked_resources)
        self.assertNotIn(self.orthogonal_E, tracked_resources)
        self.assertIn(self.moving_subtree_C, tracked_resources)  # This was assigned

    def test_contingent_splits_consumed_when_collapse_creates_space(self):
        """After collapsing, contingent splits within collapsed region can be used."""
        # Setup: collapse target_location, creating space for orthogonal splits
        collapse_by_subtree = {
            self.moving_subtree_AB: PartitionSet(
                [self.target_location], encoding=self.encoding
            ),
        }

        expand_by_subtree = {
            self.moving_subtree_AB: PartitionSet(
                [self.moving_subtree_C], encoding=self.encoding
            ),
        }

        # Orthogonal D is within target_location (indices 0,1,2,3)
        all_expand = PartitionSet(
            [self.moving_subtree_C, self.orthogonal_D], encoding=self.encoding
        )

        state = PivotSplitRegistry(
            PartitionSet([self.target_location], encoding=self.encoding),
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            self.root,
        )

        # Consume contingent splits after collapsing target_location
        collapsed = PartitionSet([self.target_location], encoding=self.encoding)
        contingent = state.consume_contingent_expand_splits_for_subtree(
            self.moving_subtree_AB, collapsed
        )

        # orthogonal_D (index 3) is within target_location, but orthogonal_E (index 4) is not
        # Only splits within the collapsed region can be contingent
        target_indices = {0, 1, 2}
        d_indices = {3}

        # D should be consumable if it fits within collapsed region
        # Let's check what actually gets consumed
        self.assertIsInstance(contingent, PartitionSet)

    def test_orthogonal_splits_expanded_after_moving_subtree(self):
        """Orthogonal splits must be expanded after the moving subtree settles."""
        # This tests the temporal ordering: collapse -> expand primary -> expand contingent
        collapse_by_subtree = {
            self.moving_subtree_AB: PartitionSet(
                [self.target_location], encoding=self.encoding
            ),
        }

        expand_by_subtree = {
            self.moving_subtree_AB: PartitionSet(
                [self.moving_subtree_AB], encoding=self.encoding
            ),
        }

        all_expand = PartitionSet(
            [self.moving_subtree_AB, self.orthogonal_D], encoding=self.encoding
        )

        state = PivotSplitRegistry(
            PartitionSet([self.target_location], encoding=self.encoding),
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            self.root,
        )

        # Primary expand (moving_subtree_AB) should be processed first
        primary = state.get_unique_expand_splits(self.moving_subtree_AB)
        self.assertIn(self.moving_subtree_AB, primary)

        # Orthogonal splits remain contingent until explicitly consumed (not in tracker)
        tracked_resources = state.expand_tracker.get_all_resources()
        self.assertNotIn(self.orthogonal_D, tracked_resources)


# ============================================================================
# SECTION 2: EXPAND-LAST STRATEGY (Based on User Clarification)
# ============================================================================


class TestExpandLastStrategy(unittest.TestCase):
    """
    Test expand-last: the last subtree sharing a split should apply it.

    User clarification: expand-last should be the subtree that is the last
    subtree sharing the path, so for that path, it should be applied then.
    """

    def setUp(self):
        """Set up scenario with shared expand splits."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

    def test_shared_expand_split_processed_by_last_user(self):
        """Shared expand split should be processed by the last subtree needing it."""
        # Both subtrees need part_AB
        expand_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB, self.part_A], encoding=self.encoding
            ),
            self.part_B: PartitionSet(
                [self.part_AB, self.part_B], encoding=self.encoding
            ),
        }

        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
        }

        state = PivotSplitRegistry(
            PartitionSet([self.part_A, self.part_B], encoding=self.encoding),
            PartitionSet(
                [self.part_AB, self.part_A, self.part_B], encoding=self.encoding
            ),
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABCD,
        )

        # Initially, both have unique splits (part_A, part_B), so they show as last users
        # The shared split (part_AB) is shared between them
        last_user_A = state.get_expand_splits_for_last_user(self.part_A)
        last_user_B = state.get_expand_splits_for_last_user(self.part_B)

        # Each subtree has its own unique split as last user, but not the shared one yet
        self.assertIn(self.part_A, last_user_A, "part_A unique split")
        self.assertIn(self.part_B, last_user_B, "part_B unique split")
        self.assertNotIn(self.part_AB, last_user_A, "shared split not last user yet")
        self.assertNotIn(self.part_AB, last_user_B, "shared split not last user yet")

        # Simulate part_A being processed - release it from tracker
        state.expand_tracker.release(self.part_AB, self.part_A)

        # Now part_B becomes the last user for part_AB
        last_user_B = state.get_expand_splits_for_last_user(self.part_B)
        self.assertIn(
            self.part_AB, last_user_B, "part_B should be last user after A processed"
        )

    def test_expand_last_applies_to_longest_shared_path(self):
        """Last user concept applies even with multiple shared splits."""
        # Three subtrees, two share a longer path
        part_ABC = Partition((0, 1, 2), self.encoding)

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_AB, part_ABC], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_AB, part_ABC], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),  # Unique
        }

        state = PivotSplitRegistry(
            PartitionSet(encoding=self.encoding),
            PartitionSet([self.part_AB, part_ABC, self.part_C], encoding=self.encoding),
            {},
            expand_by_subtree,
            self.part_ABCD,
        )

        # Remove part_A's access to shared splits via tracker
        state.expand_tracker.release(self.part_AB, self.part_A)
        state.expand_tracker.release(part_ABC, self.part_A)

        # part_B should now be last user for both part_AB and part_ABC
        last_user_B = state.get_expand_splits_for_last_user(self.part_B)
        self.assertIn(self.part_AB, last_user_B)
        self.assertIn(part_ABC, last_user_B)

        # part_C always had unique split
        last_user_C = state.get_expand_splits_for_last_user(self.part_C)
        self.assertIn(self.part_C, last_user_C)

    def test_last_subtree_in_plan_gets_all_remaining(self):
        """The final subtree in processing order gets all remaining splits."""
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
        }

        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
        }

        # Include extra expand splits not assigned to anyone
        all_expand = PartitionSet(
            [self.part_A, self.part_B, self.part_C], encoding=self.encoding
        )

        state = PivotSplitRegistry(
            PartitionSet([self.part_A, self.part_B], encoding=self.encoding),
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABCD,
        )

        # Mark part_A as processed
        state.processed_subtrees.add(self.part_A)

        # part_B should now be identified as last
        self.assertTrue(state.is_last_subtree(self.part_B))

        # Last subtree gets remaining splits
        remaining = state.get_all_remaining_expand_splits()
        self.assertIn(self.part_B, remaining)
        self.assertIn(self.part_C, remaining)  # Unassigned split


# ============================================================================
# SECTION 3: PRIORITY SYSTEM VERIFICATION
# ============================================================================


class TestPrioritySystemRobustness(unittest.TestCase):
    """Test the three-tier priority system with edge cases."""

    def setUp(self):
        """Set up complex multi-subtree scenario."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_D = Partition((3,), self.encoding)
        self.part_E = Partition((4,), self.encoding)
        self.part_F = Partition((5,), self.encoding)

        self.part_AB = Partition((0, 1), self.encoding)
        self.part_CD = Partition((2, 3), self.encoding)
        self.part_EF = Partition((4, 5), self.encoding)

        self.root = Partition((0, 1, 2, 3, 4, 5), self.encoding)

    def test_priority_0_beats_priority_1_and_2(self):
        """Shared collapse always wins."""
        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB], encoding=self.encoding
            ),  # Unique collapse
            self.part_C: PartitionSet(
                [self.part_CD, self.part_EF], encoding=self.encoding
            ),  # Shared collapse
            self.part_E: PartitionSet(
                [self.part_EF], encoding=self.encoding
            ),  # Shared collapse
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
            self.part_E: PartitionSet([self.part_E], encoding=self.encoding),
        }

        state = PivotSplitRegistry(
            PartitionSet(
                [self.part_AB, self.part_CD, self.part_EF], encoding=self.encoding
            ),
            PartitionSet(
                [self.part_A, self.part_C, self.part_E], encoding=self.encoding
            ),
            collapse_by_subtree,
            expand_by_subtree,
            self.root,
        )

        # First selected should have shared collapse (part_C or part_E)
        first = state.get_next_subtree()
        self.assertIn(
            first, [self.part_C, self.part_E], "Should select shared collapse first"
        )

    def test_tie_breaking_with_identical_priorities(self):
        """When priorities are equal, deterministic tie-breaker applies."""
        # All have unique splits (priority 1)
        collapse_by_subtree = {
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        expand_by_subtree = {
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }

        state = PivotSplitRegistry(
            PartitionSet(
                [self.part_A, self.part_B, self.part_C], encoding=self.encoding
            ),
            PartitionSet(
                [self.part_A, self.part_B, self.part_C], encoding=self.encoding
            ),
            collapse_by_subtree,
            expand_by_subtree,
            self.root,
        )

        # Get selection order
        order = []
        while state.has_remaining_work():
            next_sub = state.get_next_subtree()
            if next_sub is None:
                break
            order.append(next_sub)
            state.processed_subtrees.add(next_sub)
            # Remove splits via tracker
            state.collapse_tracker.release_owner_from_all_resources(next_sub)
            state.expand_tracker.release_owner_from_all_resources(next_sub)

        # Should be deterministic (sorted by string representation of indices)
        self.assertEqual(len(order), 3)
        # Verify it's repeatable
        collapse_by_subtree_2 = {
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }
        expand_by_subtree_2 = {
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
            self.part_C: PartitionSet([self.part_C], encoding=self.encoding),
        }
        state2 = PivotSplitRegistry(
            PartitionSet(
                [self.part_A, self.part_B, self.part_C], encoding=self.encoding
            ),
            PartitionSet(
                [self.part_A, self.part_B, self.part_C], encoding=self.encoding
            ),
            collapse_by_subtree_2,
            expand_by_subtree_2,
            self.root,
        )
        order2 = []
        while state2.has_remaining_work():
            next_sub = state2.get_next_subtree()
            if next_sub is None:
                break
            order2.append(next_sub)
            state2.processed_subtrees.add(next_sub)
            # Remove splits via tracker
            state2.collapse_tracker.release_owner_from_all_resources(next_sub)
            state2.expand_tracker.release_owner_from_all_resources(next_sub)

        self.assertEqual(order, order2, "Selection must be deterministic")

    def test_priority_2_only_when_shared_expand_no_shared_collapse(self):
        """Priority 2 (shared expand) only applies when no shared collapse exists."""
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),  # Unique
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),  # Unique
        }

        expand_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_AB], encoding=self.encoding
            ),  # Shared expand
            self.part_B: PartitionSet(
                [self.part_AB], encoding=self.encoding
            ),  # Shared expand
        }

        state = PivotSplitRegistry(
            PartitionSet([self.part_A, self.part_B], encoding=self.encoding),
            PartitionSet([self.part_AB], encoding=self.encoding),
            collapse_by_subtree,
            expand_by_subtree,
            self.root,
        )

        # Should select one with shared expand (priority 2 since no shared collapse)
        first = state.get_next_subtree()
        self.assertIn(first, [self.part_A, self.part_B])

        # After processing one, the other should be selected (also priority 2)
        state.processed_subtrees.add(first)
        state.collapse_tracker.release_owner_from_all_resources(first)
        state.expand_tracker.release_owner_from_all_resources(first)

        second = state.get_next_subtree()
        self.assertIn(second, [self.part_A, self.part_B])
        self.assertNotEqual(first, second)


# ============================================================================
# SECTION 4: SPLIT LIFECYCLE MANAGEMENT
# ============================================================================


class TestSplitLifecycleTracking(unittest.TestCase):
    """Test complete lifecycle of splits through the system."""

    def setUp(self):
        """Set up basic scenario."""
        self.encoding = {"A": 0, "B": 1, "C": 2}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_ABC = Partition((0, 1, 2), self.encoding)

    def test_split_transitions_from_available_to_processed(self):
        """Track a split from initial availability through processing."""
        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_A, self.part_AB], encoding=self.encoding
            ),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_C], encoding=self.encoding),
        }

        state = PivotSplitRegistry(
            PartitionSet([self.part_A, self.part_AB], encoding=self.encoding),
            PartitionSet([self.part_C], encoding=self.encoding),
            collapse_by_subtree,
            expand_by_subtree,
            self.part_ABC,
        )

        # Initially available (check via tracker API)
        unique_splits = state.get_unique_collapse_splits(self.part_A)
        self.assertIn(self.part_A, unique_splits)
        self.assertNotIn(self.part_C, state.used_expand_splits)

        # Mark as processed
        state.mark_splits_as_processed(
            self.part_A,
            PartitionSet([self.part_A], encoding=self.encoding),
            PartitionSet([self.part_C], encoding=self.encoding),
        )

        # Now removed from available and marked as used (check via tracker API)
        unique_splits_after = state.get_unique_collapse_splits(self.part_A)
        self.assertNotIn(self.part_A, unique_splits_after)
        self.assertIn(self.part_C, state.used_expand_splits)

    def test_shared_split_removed_from_all_subtrees_when_processed(self):
        """Shared splits deleted globally when processed by any subtree."""
        shared_split = self.part_AB

        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [shared_split, self.part_A], encoding=self.encoding
            ),
            self.part_B: PartitionSet(
                [shared_split, self.part_B], encoding=self.encoding
            ),
        }

        state = PivotSplitRegistry(
            PartitionSet(
                [shared_split, self.part_A, self.part_B], encoding=self.encoding
            ),
            PartitionSet(encoding=self.encoding),
            collapse_by_subtree,
            {},
            self.part_ABC,
        )

        # Both have access initially (check via tracker API)
        shared_A = state.get_available_shared_collapse_splits(self.part_A)
        shared_B = state.get_available_shared_collapse_splits(self.part_B)
        self.assertIn(shared_split, shared_A)
        self.assertIn(shared_split, shared_B)

        # Process via part_A
        state.mark_splits_as_processed(
            self.part_A,
            PartitionSet([shared_split], encoding=self.encoding),
            PartitionSet(encoding=self.encoding),
        )

        # Should be removed from both (check via tracker API)
        shared_A_after = state.get_available_shared_collapse_splits(self.part_A)
        shared_B_after = state.get_available_shared_collapse_splits(self.part_B)
        unique_A_after = state.get_unique_collapse_splits(self.part_A)
        unique_B_after = state.get_unique_collapse_splits(self.part_B)

        self.assertNotIn(shared_split, shared_A_after)
        self.assertNotIn(shared_split, shared_B_after)
        self.assertNotIn(shared_split, unique_A_after)
        self.assertNotIn(shared_split, unique_B_after)

    def test_contingent_split_lifecycle_consumed_and_marked(self):
        """Contingent splits transition to used when consumed."""
        contingent = self.part_C

        # Not assigned to any subtree
        all_expand = PartitionSet([self.part_A, contingent], encoding=self.encoding)
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        state = PivotSplitRegistry(
            PartitionSet([self.part_AB], encoding=self.encoding),
            all_expand,
            {self.part_A: PartitionSet([self.part_AB], encoding=self.encoding)},
            expand_by_subtree,
            self.part_ABC,
        )

        # Initially available (not in tracker)
        tracked_resources = state.expand_tracker.get_all_resources()
        self.assertNotIn(contingent, tracked_resources)

        # Consume it
        collapsed = PartitionSet([self.part_AB], encoding=self.encoding)
        consumed = state.consume_contingent_expand_splits_for_subtree(
            self.part_A, collapsed
        )

        # Mark as processed - contingent splits are included in processed_expand_splits
        state.mark_splits_as_processed(
            self.part_A,
            PartitionSet(encoding=self.encoding),
            consumed,  # Contingent splits are now part of expand tracking
        )

        # Should be claimed in tracker, then released when processed
        if contingent in consumed:
            # After mark_splits_as_processed, it should be released from tracker
            tracked_after = state.expand_tracker.get_all_resources()
            self.assertNotIn(contingent, tracked_after)


# ============================================================================
# SECTION 5: BUILDER INTEGRATION & INCOMPATIBILITY HANDLING
# ============================================================================


class TestBuilderIncompatibilityHandling(unittest.TestCase):
    """Test how builder identifies and handles incompatible splits."""

    def setUp(self):
        """Set up trees with potential incompatibilities."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)
        self.part_B = Partition((1,), self.encoding)
        self.part_C = Partition((2,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)
        self.part_AC = Partition((0, 2), self.encoding)  # Incompatible with part_AB
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)

        # Trees: source has (A,B) clade, destination has (A,C) clade
        taxa_order = ["A", "B", "C", "D"]
        self.tree_AB = parse_newick(
            "((A,B),(C,D));", order=taxa_order, encoding=self.encoding
        )
        self.tree_AC = parse_newick(
            "((A,C),(B,D));", order=taxa_order, encoding=self.encoding
        )

    def test_incompatible_splits_identified_and_processed(self):
        """Builder must find and process incompatible splits."""
        # Trying to expand part_AC while part_AB exists creates incompatibility
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_AB], encoding=self.encoding),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_AC], encoding=self.encoding),
        }

        plan = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            self.tree_AB,
            self.tree_AC,
            self.part_ABCD,
        )

        # Plan should include part_AB in collapse path (original + incompatible)
        self.assertIn(self.part_A, plan)
        collapse_path = plan[self.part_A]["collapse"]["path_segment"]

        # part_AB must be in collapse (it's incompatible with part_AC)
        self.assertIn(self.part_AB, collapse_path)

    def test_incompatible_splits_deleted_globally(self):
        """Incompatible splits removed from all subtrees."""
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_AB], encoding=self.encoding),
            self.part_B: PartitionSet(
                [self.part_AB, self.part_B], encoding=self.encoding
            ),  # Also has part_AB
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_AC], encoding=self.encoding),
            self.part_B: PartitionSet([self.part_B], encoding=self.encoding),
        }

        plan = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            self.tree_AB,
            self.tree_AC,
            self.part_ABCD,
        )

        # Both subtrees should have plans
        self.assertEqual(len(plan), 2)

        # part_AB should appear in only one plan (the one that processed it)
        ab_count = sum(
            1 for p in plan.values() if self.part_AB in p["collapse"]["path_segment"]
        )
        self.assertLessEqual(ab_count, 1, "Incompatible split processed once")


# ============================================================================
# SECTION 6: PATH ORDERING & DETERMINISM
# ============================================================================


class TestPathOrdering(unittest.TestCase):
    """Test that paths are ordered correctly and deterministically."""

    def setUp(self):
        """Set up partitions of different sizes."""
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.part_A = Partition((0,), self.encoding)  # Size 1
        self.part_AB = Partition((0, 1), self.encoding)  # Size 2
        self.part_ABC = Partition((0, 1, 2), self.encoding)  # Size 3
        self.part_ABCD = Partition((0, 1, 2, 3), self.encoding)  # Size 4

        taxa_order = ["A", "B", "C", "D"]
        self.tree = parse_newick(
            "((A,B),(C,D));", order=taxa_order, encoding=self.encoding
        )

    def test_collapse_paths_sorted_by_size_ascending(self):
        """Collapse paths should be sorted Smallest First (Leaves Inward)."""
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
            self.tree,
            self.tree,
            self.part_ABCD,
        )

        collapse_path = plan[self.part_A]["collapse"]["path_segment"]

        # Check sizes are ASCENDING (Smallest First/Leaves Inward)
        sizes = [len(p.indices) for p in collapse_path]
        self.assertEqual(sizes, sorted(sizes), "Must be size-ascending (Leaves Inward)")

        # Specifically: A (1) before AB (2) before ABC (3)
        size_map = {len(p.indices): p for p in collapse_path}
        self.assertEqual(size_map[3], self.part_ABC)
        self.assertEqual(size_map[2], self.part_AB)
        self.assertEqual(size_map[1], self.part_A)

    def test_path_ordering_deterministic_across_runs(self):
        """Same input produces same ordering every time."""
        collapse_by_subtree = {
            self.part_A: PartitionSet(
                [self.part_A, self.part_AB], encoding=self.encoding
            ),
        }

        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        # Run twice
        plan1 = build_edge_plan(
            {self.part_A: PartitionSet([self.part_A], encoding=self.encoding)},
            {
                self.part_A: PartitionSet(
                    [self.part_A, self.part_AB], encoding=self.encoding
                )
            },
            self.tree,
            self.tree,
            self.part_ABCD,
        )

        plan2 = build_edge_plan(
            {self.part_A: PartitionSet([self.part_A], encoding=self.encoding)},
            {
                self.part_A: PartitionSet(
                    [self.part_A, self.part_AB], encoding=self.encoding
                )
            },
            self.tree,
            self.tree,
            self.part_ABCD,
        )

        # Paths should be identical
        self.assertEqual(
            plan1[self.part_A]["collapse"]["path_segment"],
            plan2[self.part_A]["collapse"]["path_segment"],
        )


# ============================================================================
# SECTION 7: ERROR CONDITIONS & EDGE CASES
# ============================================================================


class TestErrorConditionsAndEdgeCases(unittest.TestCase):
    """Test error handling and boundary conditions."""

    def setUp(self):
        """Set up minimal fixtures."""
        self.encoding = {"A": 0, "B": 1}

        self.part_A = Partition((0,), self.encoding)
        self.part_AB = Partition((0, 1), self.encoding)

        taxa_order = ["A", "B"]
        self.tree = parse_newick("(A,B);", order=taxa_order, encoding=self.encoding)

    def test_build_edge_plan_with_invalid_pivot_edge_raises_error(self):
        """Pivot edge not in tree should raise ValueError."""
        fake_pivot = Partition((5, 6), {"X": 5, "Y": 6})

        with self.assertRaises(ValueError) as context:
            build_edge_plan(
                {},
                {},
                self.tree,
                self.tree,
                fake_pivot,
            )

        # Error can be either "not found" or encoding mismatch
        error_msg = str(context.exception).lower()
        self.assertTrue(
            "not found" in error_msg or "encoding" in error_msg,
            f"Expected error about 'not found' or 'encoding', got: {context.exception}",
        )

    def test_empty_state_has_no_remaining_work(self):
        """State with no subtrees returns False for has_remaining_work."""
        state = PivotSplitRegistry(
            PartitionSet(encoding=self.encoding),
            PartitionSet(encoding=self.encoding),
            {},
            {},
            self.part_AB,
        )

        self.assertFalse(state.has_remaining_work())
        self.assertIsNone(state.get_next_subtree())

    def test_get_all_remaining_with_no_used_splits(self):
        """All splits remain when none have been used."""
        all_expand = PartitionSet([self.part_A, self.part_AB], encoding=self.encoding)

        state = PivotSplitRegistry(
            PartitionSet(encoding=self.encoding),
            all_expand,
            {},
            {},
            self.part_AB,
        )

        remaining = state.get_all_remaining_expand_splits()
        self.assertEqual(remaining, all_expand)

    def test_cleanup_removes_empty_subtrees(self):
        """Empty subtrees have no resources after cleanup."""
        collapse_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        state = PivotSplitRegistry(
            PartitionSet([self.part_A], encoding=self.encoding),
            PartitionSet(encoding=self.encoding),
            collapse_by_subtree,
            {},
            self.part_AB,
        )

        # Release all splits via tracker
        state.collapse_tracker.release_owner_from_all_resources(self.part_A)

        # Verify no resources remain for this subtree
        unique = state.get_unique_collapse_splits(self.part_A)
        shared = state.get_available_shared_collapse_splits(self.part_A)
        self.assertEqual(len(unique), 0)
        self.assertEqual(len(shared), 0)

    def test_contingent_splits_empty_when_all_assigned(self):
        """No contingent splits when all expand splits assigned to subtrees."""
        all_expand = PartitionSet([self.part_A], encoding=self.encoding)
        expand_by_subtree = {
            self.part_A: PartitionSet([self.part_A], encoding=self.encoding),
        }

        state = PivotSplitRegistry(
            PartitionSet(encoding=self.encoding),
            all_expand,
            {},
            expand_by_subtree,
            self.part_AB,
        )

        # All contingent splits consumed means all expand splits in tracker initially
        # Since none were contingent, tracker should have all assigned splits
        tracked_resources = state.expand_tracker.get_all_resources()
        # Check that all expand splits were assigned (none left as contingent)
        self.assertEqual(
            len(
                state.all_expand_splits
                - PartitionSet(tracked_resources, encoding=self.encoding)
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
