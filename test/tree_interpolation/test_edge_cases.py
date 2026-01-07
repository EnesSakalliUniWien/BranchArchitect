import unittest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
)


class TestEdgeCases(unittest.TestCase):
    """
    Test suite for edge cases in tree interpolation planning, including:
    1. Shared Split Double-Counting (Regression Test)
    2. Identity Interpolation (T1 -> T1)
    3. Multifurcation Handling
    4. Extreme Topology Transitions
    """

    def setUp(self):
        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        self.root = Partition((0, 1, 2, 3, 4), self.encoding)

    def test_shared_split_regression(self):
        """
        Regression Test: A split present in BOTH Source and Destination trees
        must NOT be claimed as a collapse split or an expand split.
        It is 'shared' and should remain untouched.

        Scenario mimicking the bug:
        T1: ((A,B),C,D,E) -> has split (A,B)
        T2: ((A,B),C,E,D) -> has split (A,B) [structure preserved, maybe just label swap elsewhere]

        The bug was that (A,B) was being added to 'all_expand' and 'all_collapse'
        derived from subtree traversal, even though it wasn't unique to either.
        """
        # (A,B) is the shared split
        split_AB = Partition((0, 1), self.encoding)

        # In the bug scenario, the input to PivotSplitRegistry had lists that
        # might contain shared splits if not filtered.
        # The fix was in PivotSplitRegistry.__init__ filtering them out.

        # Simulating the inputs derived from tree traversal:
        # traverse_subtree might return AB as part of the path for subtree A.
        collapse_by_subtree_input = {
            Partition((0,), self.encoding): PartitionSet(
                [split_AB], encoding=self.encoding
            )
        }
        expand_by_subtree_input = {
            Partition((0,), self.encoding): PartitionSet(
                [split_AB], encoding=self.encoding
            )
        }

        # The key correctness condition: 'all_collapse' and 'all_expand' passed to
        # PivotSplitRegistry MUST ONLY contain splits unique to T1 and T2 respectively.
        # If the pre-calculation is correct, 'split_AB' should NOT be in all_collapse or all_expand.

        # However, to test the ROBUSTNESS of PivotSplitRegistry (where the fix was applied),
        # we will pass 'split_AB' in the subtree dictionaries but NOT in the global unique sets.
        # The registry should filter it out.

        all_unique_collapse = PartitionSet(encoding=self.encoding)  # Empty
        all_unique_expand = PartitionSet(encoding=self.encoding)  # Empty

        registry = PivotSplitRegistry(
            all_unique_collapse,
            all_unique_expand,
            collapse_by_subtree_input,
            expand_by_subtree_input,
            self.root,
        )

        # Assertions
        # 1. AB should NOT be tracked as a collapse split for A
        unique_collapse_for_A = registry.get_unique_collapse_splits(
            Partition((0,), self.encoding)
        )
        self.assertNotIn(
            split_AB,
            unique_collapse_for_A,
            "Shared split AB should not be claimed as unique collapse split",
        )

        # 2. AB should NOT be tracked as an expand split for A
        unique_expand_for_A = registry.get_unique_expand_splits(
            Partition((0,), self.encoding)
        )
        self.assertNotIn(
            split_AB,
            unique_expand_for_A,
            "Shared split AB should not be claimed as unique expand split",
        )

    def test_identity_interpolation(self):
        """
        Interpolating a tree to itself should result in an empty plan.
        """
        tree_str = "((A,B),(C,(D,E)));"
        parse_newick(tree_str, order=["A", "B", "C", "D", "E"], encoding=self.encoding)

        # If T1 == T2, then:
        # all_collapse_splits = empty
        # all_expand_splits = empty
        # collapse_by_subtree = {leaf: empty...} but might contain shared splits if traversal is naive
        # The registry should handle this gracefully.

        # Let's say traversal finds the path up to root.
        # For leaf A: path is A -> AB -> Root. AB is shared.

        split_AB = Partition((0, 1), self.encoding)

        collapse_by_subtree = {
            Partition((0,), self.encoding): PartitionSet(
                [split_AB], encoding=self.encoding
            )
        }
        expand_by_subtree = {
            Partition((0,), self.encoding): PartitionSet(
                [split_AB], encoding=self.encoding
            )
        }

        registry = PivotSplitRegistry(
            PartitionSet(encoding=self.encoding),  # None unique
            PartitionSet(encoding=self.encoding),  # None unique
            collapse_by_subtree,
            expand_by_subtree,
            self.root,
        )

        self.assertFalse(
            registry.has_remaining_work(), "Identity interpolation should have no work"
        )

        subtree = registry.get_next_subtree()
        self.assertIsNone(
            subtree, "No subtree should be selected for identity interpolation"
        )

    def test_star_topology_handling(self):
        """
        Transition from fully resolved to star topology.
        T1: ((A,B),(C,D))
        T2: (A,B,C,D) - Star

        All internal splits of T1 must be collapsed.
        No new splits need to be expanded.
        """
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
        root = Partition((0, 1, 2, 3), encoding)

        split_AB = Partition((0, 1), encoding)
        split_CD = Partition((2, 3), encoding)

        # all_collapse = {AB, CD}
        # all_expand = {}

        all_collapse = PartitionSet([split_AB, split_CD], encoding)
        all_expand = PartitionSet(encoding=encoding)

        collapse_by_subtree = {
            Partition((0,), encoding): PartitionSet([split_AB], encoding),  # A uses AB
            Partition((1,), encoding): PartitionSet([split_AB], encoding),  # B uses AB
            Partition((2,), encoding): PartitionSet([split_CD], encoding),  # C uses CD
            Partition((3,), encoding): PartitionSet([split_CD], encoding),  # D uses CD
        }
        expand_by_subtree = {
            Partition((0,), encoding): PartitionSet(encoding=encoding),
            Partition((1,), encoding): PartitionSet(encoding=encoding),
            Partition((2,), encoding): PartitionSet(encoding=encoding),
            Partition((3,), encoding): PartitionSet(encoding=encoding),
        }

        registry = PivotSplitRegistry(
            all_collapse, all_expand, collapse_by_subtree, expand_by_subtree, root
        )

        self.assertTrue(registry.has_remaining_work())

        # We expect to process until empty
        processed_count = 0
        while registry.has_remaining_work():
            sub = registry.get_next_subtree()
            if not sub:
                break
            registry.processed_subtrees.add(sub)
            registry.collapse_tracker.release_owner_from_all_resources(sub)
            registry.expand_tracker.release_owner_from_all_resources(sub)
            processed_count += 1

        # Should process enough subtrees to cover all splits
        self.assertGreater(processed_count, 0)

        # Verify all splits were "used" (in this mock, released)
        # Check by asserting trackers are empty of the original resources
        self.assertEqual(len(registry.collapse_tracker.get_all_resources()), 0)

    def test_multifurcation_transition(self):
        """
        Transition between two different multifurcating trees.
        T1: (A,B,C,D) (Star)
        T2: ((A,B),(C,D)) (Resolved)

        This is the reverse of the star topology test. It requires pure expansion.
        """
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
        root = Partition((0, 1, 2, 3), encoding)

        split_AB = Partition((0, 1), encoding)
        split_CD = Partition((2, 3), encoding)

        # T1 has no internal splits (star). T2 has AB and CD.
        # all_collapse = {}
        # all_expand = {AB, CD}

        all_collapse = PartitionSet(encoding=encoding)
        all_expand = PartitionSet([split_AB, split_CD], encoding)

        collapse_by_subtree = {
            Partition((0,), encoding): PartitionSet(encoding=encoding),
            Partition((1,), encoding): PartitionSet(encoding=encoding),
            Partition((2,), encoding): PartitionSet(encoding=encoding),
            Partition((3,), encoding): PartitionSet(encoding=encoding),
        }

        # AB is needed by A and B
        # CD is needed by C and D
        expand_by_subtree = {
            Partition((0,), encoding): PartitionSet([split_AB], encoding),
            Partition((1,), encoding): PartitionSet([split_AB], encoding),
            Partition((2,), encoding): PartitionSet([split_CD], encoding),
            Partition((3,), encoding): PartitionSet([split_CD], encoding),
        }

        registry = PivotSplitRegistry(
            all_collapse, all_expand, collapse_by_subtree, expand_by_subtree, root
        )

        # Verify work is detected
        self.assertTrue(registry.has_remaining_work())

        # Verify expand splits are tracked
        self.assertIn(split_AB, registry.expand_tracker.get_all_resources())
        self.assertIn(split_CD, registry.expand_tracker.get_all_resources())

        # Simulate processing one "side" (e.g., A)
        # Subtree Partition((0,), encoding)
        # Assuming A is processed:
        # It should consume AB (or wait if shared logic requires it)
        # Since AB is shared with B, priority might be lower than unique splits,
        # but here ONLY shared splits exist.

        # Let's ensure we can make progress
        next_sub = registry.get_next_subtree()
        self.assertIsNotNone(next_sub)

    def test_jumping_taxon_scenario(self):
        """
        Scenario: Taxon E moves from ((A,B),C,D,E) to ((A,B),(C,(D,E))).

        T1: (AB, C, D, E)  -> Splits: {AB}
        T2: (AB, (C, (D, E))) -> Splits: {AB, CDE, DE}

        Common: {AB}
        Unique T1 (Collapse): None
        Unique T2 (Expand): {CDE, DE}

        E needs to "pass through" C and D to get to its position?
        Or rather, CDE and DE need to be created.
        E is involved in DE.
        D is involved in DE.
        C is involved in CDE.

        This tests if the registry correctly associates these "nested" expansions.
        """
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        root = Partition((0, 1, 2, 3, 4), encoding)

        # Shared AB
        split_DE = Partition((3, 4), encoding)  # New
        split_CDE = Partition((2, 3, 4), encoding)  # New

        # Only new splits in all_expand
        all_expand = PartitionSet([split_DE, split_CDE], encoding)
        all_collapse = PartitionSet(encoding=encoding)

        # Expand assignments (simplified for test):
        # DE is shared by D and E
        # CDE is shared by C, D, E
        expand_by_subtree = {
            Partition((2,), encoding): PartitionSet(
                [split_CDE], encoding
            ),  # C needs CDE
            Partition((3,), encoding): PartitionSet(
                [split_DE, split_CDE], encoding
            ),  # D needs DE, CDE
            Partition((4,), encoding): PartitionSet(
                [split_DE, split_CDE], encoding
            ),  # E needs DE, CDE
            Partition((0,), encoding): PartitionSet(
                encoding=encoding
            ),  # A needs nothing new
            Partition((1,), encoding): PartitionSet(
                encoding=encoding
            ),  # B needs nothing new
        }

        registry = PivotSplitRegistry(
            all_collapse,
            all_expand,
            {},  # No collapse
            expand_by_subtree,
            root,
        )

        # C, D, E have work. A, B do not.

        # 1. Verify C, D, E are candidates
        # D and E share DE. C, D, E share CDE.
        # Likely candidates are those with more specific splits or by priority.

        sub = registry.get_next_subtree()
        self.assertIn(
            list(sub.indices)[0], [2, 3, 4], "Selected subtree should be C, D, or E"
        )

    def test_reverse_interpolation_symmetry(self):
        """
        Verify that T1->T2 and T2->T1 generate symmetric split sets.
        T1: ((A,B),C)
        T2: (A,(B,C))

        T1->T2: Collapse (A,B), Expand (B,C)
        T2->T1: Collapse (B,C), Expand (A,B)
        """
        encoding = {"A": 0, "B": 1, "C": 2}
        root = Partition((0, 1, 2), encoding)

        split_AB = Partition((0, 1), encoding)
        split_BC = Partition((1, 2), encoding)

        # Case 1: T1 -> T2
        reg1 = PivotSplitRegistry(
            PartitionSet([split_AB], encoding),  # Collapse AB
            PartitionSet([split_BC], encoding),  # Expand BC
            {
                Partition((0,), encoding): PartitionSet([split_AB], encoding),
                Partition((1,), encoding): PartitionSet([split_AB], encoding),
            },
            {
                Partition((1,), encoding): PartitionSet([split_BC], encoding),
                Partition((2,), encoding): PartitionSet([split_BC], encoding),
            },
            root,
        )

        # Case 2: T2 -> T1
        reg2 = PivotSplitRegistry(
            PartitionSet([split_BC], encoding),  # Collapse BC
            PartitionSet([split_AB], encoding),  # Expand AB
            {
                Partition((1,), encoding): PartitionSet([split_BC], encoding),
                Partition((2,), encoding): PartitionSet([split_BC], encoding),
            },
            {
                Partition((0,), encoding): PartitionSet([split_AB], encoding),
                Partition((1,), encoding): PartitionSet([split_AB], encoding),
            },
            root,
        )

        # Verify counts are symmetric
        self.assertEqual(
            len(reg1.get_all_remaining_collapse_splits()),
            len(reg2.get_all_remaining_expand_splits()),
        )
        self.assertEqual(
            len(reg1.get_all_remaining_expand_splits()),
            len(reg2.get_all_remaining_collapse_splits()),
        )

        # Verify specific splits
        self.assertIn(split_AB, reg1.get_all_remaining_collapse_splits())
        self.assertIn(split_AB, reg2.get_all_remaining_expand_splits())

    def test_grouped_movers_reordering(self):
        """
        Verify that when multiple movers are in the same moving partition,
        they are reordered according to the Destination Tree, NOT the Source Tree.

        Scenario:
        T1: (A, M1, M2, B)
        T2: (A, M2, M1, B)

        Anchors: A, B (Stable)
        Movers: M1, M2 (Moving Group)

        If we collapse (A,M1,M2,B) -> (A,B), then expand, it's trivial.
        But 'reorder_tree_toward_destination' is used during the Reorder phase
        where we might have a group {M1, M2} floating relative to {A, B}.

        The function should return a tree ordered as A, M2, M1, B.
        """
        from brancharchitect.tree_interpolation.subtree_paths.execution.reordering import (
            reorder_tree_toward_destination,
        )

        encoding = {"A": 0, "B": 1, "M1": 2, "M2": 3}
        # T1: (A, M1, M2, B)
        # T2: (A, M2, M1, B)
        # Note: M1, M2 are swapped in T2 relative to T1.

        t1 = parse_newick("(A,B,M1,M2);", encoding=encoding)
        t1.reorder_taxa(["A", "M1", "M2", "B"])

        t2 = parse_newick("(A,B,M1,M2);", encoding=encoding)
        t2.reorder_taxa(["A", "M2", "M1", "B"])

        # Current pivot edge is the root in this flat case, or any edge containing them all.
        # Let's assume we are reordering under the root for simplicity.
        root_partition = Partition((0, 1, 2, 3), encoding)

        # M1 and M2 are the moving group
        movers_partition = Partition((2, 3), encoding)

        # Act
        # We expect M1 and M2 to be reordered to match T2
        result_tree = reorder_tree_toward_destination(
            t1, t2, root_partition, movers_partition, unstable_taxa=set(), copy=True
        )

        # Assert
        final_order = list(result_tree.get_current_order())
        expected_order = ["A", "M2", "M1", "B"]

        self.assertEqual(
            final_order,
            expected_order,
            f"Expected {expected_order}, but got {final_order}. Movers should follow destination order.",
        )
