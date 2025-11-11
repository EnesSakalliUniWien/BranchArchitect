"""
Test suite for incompatibility detection in state_v2.py

This module tests the enhanced incompatibility detection that uses
Partition.is_compatible_with() to identify splits that must be collapsed
before expand operations can proceed.

Key test scenarios:
1. Overlapping-but-not-nested splits (INCOMPATIBLE)
2. Nested splits (COMPATIBLE)
3. Disjoint splits (COMPATIBLE)
4. Multiple incompatibilities
5. Real tree scenarios from small_example.newick
"""

import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.state_v2 import (
    InterpolationState,
)


class TestIncompatibilityDetection:
    """Test incompatibility detection using Partition.is_compatible_with()"""

    @pytest.fixture
    def encoding(self):
        """Standard encoding for test partitions"""
        return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    @pytest.fixture
    def all_indices(self, encoding):
        """Set of all indices for compatibility checks"""
        return set(encoding.values())

    def test_overlapping_but_not_nested_incompatible(self, encoding, all_indices):
        """
        Test that overlapping-but-not-nested splits are detected as INCOMPATIBLE.

        Example:
            Split A: (A, B, C)    indices: {0, 1, 2}
            Split B: (C, D, E)    indices: {2, 3, 4}

        These splits overlap at C (index 2) but neither is a subset of the other.
        They are INCOMPATIBLE and cannot coexist in the same tree.
        """
        split_a = Partition((0, 1, 2), encoding)  # (A, B, C)
        split_b = Partition((2, 3, 4), encoding)  # (C, D, E)

        # Verify they overlap
        assert len(set(split_a.indices) & set(split_b.indices)) > 0

        # Verify neither is subset of other
        set_a = set(split_a.indices)
        set_b = set(split_b.indices)
        assert not set_a.issubset(set_b)
        assert not set_b.issubset(set_a)

        # They should be INCOMPATIBLE
        assert not split_a.is_compatible_with(split_b, all_indices)
        assert not split_b.is_compatible_with(split_a, all_indices)

    def test_nested_splits_compatible(self, encoding, all_indices):
        """
        Test that nested splits are detected as COMPATIBLE.

        Example:
            Split A: (A, B, C, D)    indices: {0, 1, 2, 3}
            Split B: (B, C)          indices: {1, 2}

        Split B is a subset of Split A. They are COMPATIBLE and can coexist.
        """
        split_a = Partition((0, 1, 2, 3), encoding)  # (A, B, C, D)
        split_b = Partition((1, 2), encoding)  # (B, C)

        # Verify subset relationship
        assert set(split_b.indices).issubset(set(split_a.indices))

        # They should be COMPATIBLE
        assert split_a.is_compatible_with(split_b, all_indices)
        assert split_b.is_compatible_with(split_a, all_indices)

    def test_disjoint_splits_compatible(self, encoding, all_indices):
        """
        Test that disjoint splits are detected as COMPATIBLE.

        Example:
            Split A: (A, B)       indices: {0, 1}
            Split B: (D, E, F)    indices: {3, 4, 5}

        These splits have no overlap. They are COMPATIBLE.
        """
        split_a = Partition((0, 1), encoding)  # (A, B)
        split_b = Partition((3, 4, 5), encoding)  # (D, E, F)

        # Verify disjoint
        assert len(set(split_a.indices) & set(split_b.indices)) == 0

        # They should be COMPATIBLE
        assert split_a.is_compatible_with(split_b, all_indices)
        assert split_b.is_compatible_with(split_a, all_indices)

    def test_find_single_incompatible_split(self, encoding):
        """
        Test finding a single incompatible split in state_v2.

        Setup:
            - Expand split: (A, B, C)
            - Collapse splits: (C, D, E), (E, F)

        Expected:
            - (C, D, E) is INCOMPATIBLE with (A, B, C)
            - (E, F) is COMPATIBLE (disjoint from A, B)
        """
        # Create minimal state
        expand_split = Partition((0, 1, 2), encoding)  # (A, B, C)
        collapse_split_1 = Partition((2, 3, 4), encoding)  # (C, D, E) - INCOMPATIBLE
        collapse_split_2 = Partition((4, 5), encoding)  # (E, F) - COMPATIBLE

        expand_splits = PartitionSet([expand_split], encoding=encoding)
        collapse_splits = PartitionSet(
            [collapse_split_1, collapse_split_2], encoding=encoding
        )

        # Create state (minimal setup)
        state = InterpolationState(
            all_collapse_splits=collapse_splits,
            all_expand_splits=expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        # Find incompatible splits
        incompatible = state.find_all_incompatible_splits_for_expand(
            expand_splits, collapse_splits
        )

        # Should find exactly one incompatible split
        assert len(incompatible) == 1
        assert collapse_split_1 in incompatible
        assert collapse_split_2 not in incompatible

    def test_find_multiple_incompatible_splits(self, encoding):
        """
        Test finding multiple incompatible splits.

        Setup:
            - Expand splits: (A, B), (C, D)
            - Collapse splits: (B, C), (D, E), (E, F)

        Expected:
            - (B, C) is INCOMPATIBLE with (A, B) - overlaps at B
            - (B, C) is INCOMPATIBLE with (C, D) - overlaps at C
            - (D, E) is INCOMPATIBLE with (C, D) - overlaps at D
            - (E, F) is COMPATIBLE with both (disjoint)
        """
        expand_split_1 = Partition((0, 1), encoding)  # (A, B)
        expand_split_2 = Partition((2, 3), encoding)  # (C, D)

        collapse_split_1 = Partition((1, 2), encoding)  # (B, C) - INCOMPATIBLE
        collapse_split_2 = Partition((3, 4), encoding)  # (D, E) - INCOMPATIBLE
        collapse_split_3 = Partition((4, 5), encoding)  # (E, F) - COMPATIBLE

        expand_splits = PartitionSet(
            [expand_split_1, expand_split_2], encoding=encoding
        )
        collapse_splits = PartitionSet(
            [collapse_split_1, collapse_split_2, collapse_split_3], encoding=encoding
        )

        state = InterpolationState(
            all_collapse_splits=collapse_splits,
            all_expand_splits=expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        incompatible = state.find_all_incompatible_splits_for_expand(
            expand_splits, collapse_splits
        )

        # Should find two incompatible splits
        assert len(incompatible) == 2
        assert collapse_split_1 in incompatible  # (B, C) overlaps both expand splits
        assert collapse_split_2 in incompatible  # (D, E) overlaps (C, D)
        assert collapse_split_3 not in incompatible  # (E, F) is disjoint

    def test_empty_sets(self, encoding):
        """Test that empty sets return empty results"""
        state = InterpolationState(
            all_collapse_splits=PartitionSet(encoding=encoding),
            all_expand_splits=PartitionSet(encoding=encoding),
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        # Empty expand splits
        result = state.find_all_incompatible_splits_for_expand(
            PartitionSet(encoding=encoding),
            PartitionSet([Partition((0, 1), encoding)], encoding=encoding),
        )
        assert len(result) == 0

        # Empty collapse splits
        result = state.find_all_incompatible_splits_for_expand(
            PartitionSet([Partition((0, 1), encoding)], encoding=encoding),
            PartitionSet(encoding=encoding),
        )
        assert len(result) == 0

    def test_identical_splits_not_incompatible(self, encoding):
        """
        Test that identical splits are not marked as incompatible.

        A split cannot be incompatible with itself.
        """
        split = Partition((0, 1, 2), encoding)  # (A, B, C)

        expand_splits = PartitionSet([split], encoding=encoding)
        collapse_splits = PartitionSet([split], encoding=encoding)

        state = InterpolationState(
            all_collapse_splits=collapse_splits,
            all_expand_splits=expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        incompatible = state.find_all_incompatible_splits_for_expand(
            expand_splits, collapse_splits
        )

        # Should be empty - a split is not incompatible with itself
        assert len(incompatible) == 0


class TestRealTreeScenarios:
    """Test incompatibility detection with real tree scenarios"""

    @pytest.fixture
    def simple_encoding(self):
        """Encoding from small_example.newick"""
        return {
            "A1": 0,
            "A2": 1,
            "B": 2,
            "C1": 3,
            "C2": 4,
            "D1": 5,
            "D2": 6,
        }

    def test_scenario_overlapping_clades(self, simple_encoding):
        """
        Test realistic scenario: overlapping clades that must be collapsed.

        Tree 1 has: ((A1, A2), (B, C1))
        Tree 2 has: ((A2, B), (C1, C2))

        To expand (A2, B), we must collapse (A1, A2) and (B, C1) first.
        """
        encoding = simple_encoding

        # Expand split from Tree 2
        expand_a2_b = Partition((1, 2), encoding)  # (A2, B)

        # Collapse splits from Tree 1
        collapse_a1_a2 = Partition((0, 1), encoding)  # (A1, A2) - overlaps at A2
        collapse_b_c1 = Partition((2, 3), encoding)  # (B, C1) - overlaps at B
        collapse_c1_c2 = Partition((3, 4), encoding)  # (C1, C2) - disjoint

        expand_splits = PartitionSet([expand_a2_b], encoding=encoding)
        collapse_splits = PartitionSet(
            [collapse_a1_a2, collapse_b_c1, collapse_c1_c2], encoding=encoding
        )

        state = InterpolationState(
            all_collapse_splits=collapse_splits,
            all_expand_splits=expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        incompatible = state.find_all_incompatible_splits_for_expand(
            expand_splits, collapse_splits
        )

        # Should find two incompatible splits
        assert len(incompatible) == 2
        assert collapse_a1_a2 in incompatible  # Overlaps at A2
        assert collapse_b_c1 in incompatible  # Overlaps at B
        assert collapse_c1_c2 not in incompatible  # Disjoint

    def test_scenario_nested_clades_compatible(self, simple_encoding):
        """
        Test that nested clades are compatible.

        If Tree 1 has ((A1, A2, B)) and we want to expand (A1, A2),
        the nested structure should be compatible.
        """
        encoding = simple_encoding

        expand_a1_a2 = Partition((0, 1), encoding)  # (A1, A2)
        collapse_a1_a2_b = Partition(
            (0, 1, 2), encoding
        )  # (A1, A2, B) - contains (A1, A2)

        expand_splits = PartitionSet([expand_a1_a2], encoding=encoding)
        collapse_splits = PartitionSet([collapse_a1_a2_b], encoding=encoding)

        state = InterpolationState(
            all_collapse_splits=collapse_splits,
            all_expand_splits=expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        incompatible = state.find_all_incompatible_splits_for_expand(
            expand_splits, collapse_splits
        )

        # Should be empty - nested splits are compatible
        assert len(incompatible) == 0

    def test_scenario_chain_of_incompatibilities(self, simple_encoding):
        """
        Test chain reaction: expanding one split may require collapsing multiple others.

        Expand: (A2, B, C1)
        Collapse candidates: (A1, A2), (B, C1, C2), (C1, D1)

        All three overlap with the expand split in different ways.
        """
        encoding = simple_encoding

        expand_split = Partition((1, 2, 3), encoding)  # (A2, B, C1)

        collapse_1 = Partition((0, 1), encoding)  # (A1, A2) - overlaps at A2
        collapse_2 = Partition((2, 3, 4), encoding)  # (B, C1, C2) - overlaps at B, C1
        collapse_3 = Partition((3, 5), encoding)  # (C1, D1) - overlaps at C1

        expand_splits = PartitionSet([expand_split], encoding=encoding)
        collapse_splits = PartitionSet(
            [collapse_1, collapse_2, collapse_3], encoding=encoding
        )

        state = InterpolationState(
            all_collapse_splits=collapse_splits,
            all_expand_splits=expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        incompatible = state.find_all_incompatible_splits_for_expand(
            expand_splits, collapse_splits
        )

        # All three should be incompatible
        assert len(incompatible) == 3
        assert collapse_1 in incompatible
        assert collapse_2 in incompatible
        assert collapse_3 in incompatible


class TestCollapseBeforeExpand:
    """Test that incompatible splits are collapsed before expansion"""

    @pytest.fixture
    def encoding(self):
        return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def test_collapse_first_strategy(self, encoding):
        """
        Test the "collapse-first" strategy.

        When we have incompatible splits, they should be identified and
        added to the collapse path BEFORE any expand operations.
        """
        # Setup: want to expand (A, B), but (B, C) exists in tree
        expand_split = Partition((0, 1), encoding)  # (A, B)
        incompatible_split = Partition((1, 2), encoding)  # (B, C) - must collapse first

        # Additional collapse splits that are compatible
        other_collapse = Partition((3, 4), encoding)  # (D, E) - disjoint

        expand_splits = PartitionSet([expand_split], encoding=encoding)
        collapse_splits = PartitionSet(
            [incompatible_split, other_collapse], encoding=encoding
        )

        state = InterpolationState(
            all_collapse_splits=collapse_splits,
            all_expand_splits=expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        # Find what must be collapsed
        must_collapse = state.find_all_incompatible_splits_for_expand(
            expand_splits, collapse_splits
        )

        # Verify only the incompatible split is identified
        assert len(must_collapse) == 1
        assert incompatible_split in must_collapse
        assert other_collapse not in must_collapse

        # In the actual execution, this would be added to collapse path
        # BEFORE the expand operations

    def test_multiple_expands_with_dependencies(self, encoding):
        """
        Test that when multiple expands are planned, all their incompatibilities
        are found upfront.

        This ensures we collapse everything needed BEFORE starting any expansions.
        """
        # Want to expand: (A, B) and (C, D)
        expand_1 = Partition((0, 1), encoding)  # (A, B)
        expand_2 = Partition((2, 3), encoding)  # (C, D)

        # Incompatible with expand_1
        collapse_1 = Partition((1, 2), encoding)  # (B, C) - overlaps with (A, B)

        # Incompatible with expand_2
        collapse_2 = Partition((3, 4), encoding)  # (D, E) - overlaps with (C, D)

        # Compatible with both
        collapse_3 = Partition((4,), encoding)  # (E) - singleton, disjoint

        expand_splits = PartitionSet([expand_1, expand_2], encoding=encoding)
        collapse_splits = PartitionSet(
            [collapse_1, collapse_2, collapse_3], encoding=encoding
        )

        state = InterpolationState(
            all_collapse_splits=collapse_splits,
            all_expand_splits=expand_splits,
            collapse_splits_by_subtree={},
            expand_splits_by_subtree={},
            active_changing_edge=Partition((0,), encoding),
        )

        # Find all incompatibilities
        must_collapse = state.find_all_incompatible_splits_for_expand(
            expand_splits, collapse_splits
        )

        # Should find both incompatible splits
        assert len(must_collapse) == 2
        assert collapse_1 in must_collapse
        assert collapse_2 in must_collapse
        assert collapse_3 not in must_collapse


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
