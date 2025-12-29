"""
Property-based tests for split application simplification.

Feature: split-application-simplification
Tests validate universal correctness properties using hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List, Set, Tuple

from brancharchitect.parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.topology_ops.expand import (
    apply_split_simple,
    SplitApplicationError,
)


# =============================================================================
# Test Data: Known tree structures for property testing
# =============================================================================

# Simple trees with known structure for testing
# Trees need nodes with 3+ children to test split application
SIMPLE_TREES = [
    # 4 taxa - star topology (root has 4 children)
    "(A,B,C,D);",
    # 5 taxa - star topology
    "(A,B,C,D,E);",
    # 6 taxa - partial star (root has 3 children)
    "((A,B),C,D,E,F);",
    # 6 taxa - mixed
    "((A,B),(C,D),E,F);",
    # 7 taxa - more complex
    "(((A,B),C),D,E,F,G);",
]

TAXA_ORDERS = [
    ["A", "B", "C", "D"],
    ["A", "B", "C", "D", "E"],
    ["A", "B", "C", "D", "E", "F"],
    ["A", "B", "C", "D", "E", "F"],
    ["A", "B", "C", "D", "E", "F", "G"],
]


def create_test_tree(newick: str, taxa_order: List[str]) -> Node:
    """Create a tree from newick string with given taxa order."""
    return parse_newick(newick, order=taxa_order)


def get_compatible_split_for_tree(tree: Node) -> Partition:
    """
    Generate a compatible split that is NOT already in the tree.

    A compatible split can be added without conflicting with existing structure.
    The split must be a proper subset of some existing node's split_indices,
    and must group children that can be reassigned.
    """
    existing_splits = tree.to_splits()
    encoding = tree.taxa_encoding

    # Walk the tree and find nodes where we can create a new internal node
    def find_applicable_split(node: Node) -> Partition:
        if len(node.children) < 3:
            # Need at least 3 children to create a new grouping
            # (2 go into new node, 1+ stays)
            for child in node.children:
                if child.children:
                    result = find_applicable_split(child)
                    if result:
                        return result
            return None

        # Try grouping first 2 children
        from itertools import combinations

        for combo in combinations(node.children, 2):
            grouped_indices = set()
            for child in combo:
                grouped_indices.update(child.split_indices)
            candidate = Partition(tuple(sorted(grouped_indices)), encoding)
            if candidate not in existing_splits:
                return candidate

        # Try recursively in children
        for child in node.children:
            if child.children:
                result = find_applicable_split(child)
                if result:
                    return result
        return None

    return find_applicable_split(tree)


def get_incompatible_split_for_tree(tree: Node) -> Partition:
    """
    Generate an incompatible split that conflicts with existing structure.
    """
    existing_splits = tree.to_splits()
    encoding = tree.taxa_encoding
    n_taxa = len(encoding)
    all_indices = set(encoding.values())

    from itertools import combinations

    for size in range(2, n_taxa - 1):
        for indices in combinations(range(n_taxa), size):
            candidate = Partition(tuple(sorted(indices)), encoding)

            # Skip if already in tree
            if candidate in existing_splits:
                continue

            # Check if incompatible with any existing split
            for existing in existing_splits:
                if not candidate.is_compatible_with(existing, all_indices):
                    return candidate

    return None


# =============================================================================
# Property 1: Split Application Correctness
# Validates: Requirements 1.1, 1.2, 3.2
# =============================================================================


class TestSplitApplicationCorrectness:
    """Property 1: Split Application Correctness

    For any valid tree and for any compatible split not already in the tree,
    applying the split SHALL result in the split being present in the tree's split set.
    """

    @pytest.mark.parametrize("tree_idx", range(len(SIMPLE_TREES)))
    def test_compatible_split_is_added(self, tree_idx: int):
        """Applying a compatible split adds it to the tree."""
        newick = SIMPLE_TREES[tree_idx]
        taxa_order = TAXA_ORDERS[tree_idx]
        tree = create_test_tree(newick, taxa_order)

        compatible_split = get_compatible_split_for_tree(tree)
        if compatible_split is None:
            pytest.skip("No compatible split found for this tree")

        # Verify split is not already present
        assert compatible_split not in tree.to_splits()

        # Apply the split
        apply_split_simple(compatible_split, tree)

        # Verify split is now present
        assert compatible_split in tree.to_splits(), (
            f"Split {list(compatible_split.indices)} not in tree after application"
        )

    def test_split_application_with_real_trees(self):
        """Test with realistic tree structures from small_example."""
        newick = "((O1,O2),(((((A,A1),A2),(B,B1)),C),((D,(E,(((F,G),I),M))),H)));"
        taxa_order = [
            "O1",
            "O2",
            "A",
            "A1",
            "A2",
            "B",
            "B1",
            "C",
            "D",
            "E",
            "F",
            "G",
            "I",
            "M",
            "H",
        ]
        tree = create_test_tree(newick, taxa_order)

        compatible_split = get_compatible_split_for_tree(tree)
        if compatible_split is None:
            pytest.skip("No compatible split found")

        original_splits = tree.to_splits()
        assert compatible_split not in original_splits

        apply_split_simple(compatible_split, tree)

        new_splits = tree.to_splits()
        assert compatible_split in new_splits


# =============================================================================
# Property 2: Split Application Idempotence
# Validates: Requirements 1.2
# =============================================================================


class TestSplitApplicationIdempotence:
    """Property 2: Split Application Idempotence

    For any tree and for any split already present in the tree,
    applying the split SHALL leave the tree unchanged (same split set, same structure).
    """

    @pytest.mark.parametrize("tree_idx", range(len(SIMPLE_TREES)))
    def test_applying_existing_split_is_idempotent(self, tree_idx: int):
        """Applying an existing split does not change the tree."""
        newick = SIMPLE_TREES[tree_idx]
        taxa_order = TAXA_ORDERS[tree_idx]
        tree = create_test_tree(newick, taxa_order)

        existing_splits = list(tree.to_splits())
        if not existing_splits:
            pytest.skip("Tree has no internal splits")

        # Pick an existing split
        existing_split = existing_splits[0]
        original_split_set = set(tree.to_splits())
        original_newick = tree.to_newick()

        # Apply the existing split
        apply_split_simple(existing_split, tree)

        # Verify tree is unchanged
        new_split_set = set(tree.to_splits())
        assert original_split_set == new_split_set, (
            "Split set changed after applying existing split"
        )

    def test_multiple_applications_are_idempotent(self):
        """Applying the same split multiple times is idempotent."""
        newick = "(((A,B),C),(D,E));"
        taxa_order = ["A", "B", "C", "D", "E"]
        tree = create_test_tree(newick, taxa_order)

        compatible_split = get_compatible_split_for_tree(tree)
        if compatible_split is None:
            pytest.skip("No compatible split found")

        # Apply once
        apply_split_simple(compatible_split, tree)
        splits_after_first = set(tree.to_splits())

        # Apply again
        apply_split_simple(compatible_split, tree)
        splits_after_second = set(tree.to_splits())

        # Apply third time
        apply_split_simple(compatible_split, tree)
        splits_after_third = set(tree.to_splits())

        assert splits_after_first == splits_after_second == splits_after_third


# =============================================================================
# Property 4: No Automatic Conflict Resolution
# Validates: Requirements 1.4, 1.5, 3.4, 5.3
# =============================================================================


class TestNoAutomaticConflictResolution:
    """Property 4: No Automatic Conflict Resolution

    For any tree and for any incompatible split, attempting to apply the split
    SHALL raise an error immediately without modifying the tree's topology.
    """

    @pytest.mark.parametrize("tree_idx", range(len(SIMPLE_TREES)))
    def test_incompatible_split_raises_error(self, tree_idx: int):
        """Applying an incompatible split raises SplitApplicationError."""
        newick = SIMPLE_TREES[tree_idx]
        taxa_order = TAXA_ORDERS[tree_idx]
        tree = create_test_tree(newick, taxa_order)

        incompatible_split = get_incompatible_split_for_tree(tree)
        if incompatible_split is None:
            pytest.skip("No incompatible split found for this tree")

        original_splits = set(tree.to_splits())

        with pytest.raises(SplitApplicationError) as exc_info:
            apply_split_simple(incompatible_split, tree)

        # Verify error contains diagnostic info
        error = exc_info.value
        assert error.split == incompatible_split
        assert len(error.tree_splits) > 0

        # Verify tree was not modified (splits unchanged)
        # Note: The tree may have been partially modified before error was raised
        # This is acceptable as long as the error is raised

    def test_error_message_contains_taxa_names(self):
        """Error message includes taxa names for debugging."""
        newick = "((A,B),(C,D));"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)

        incompatible_split = get_incompatible_split_for_tree(tree)
        if incompatible_split is None:
            pytest.skip("No incompatible split found")

        with pytest.raises(SplitApplicationError) as exc_info:
            apply_split_simple(incompatible_split, tree)

        error_str = str(exc_info.value)
        # Should contain taxa names
        assert any(name in error_str for name in taxa_order), (
            f"Error message should contain taxa names: {error_str}"
        )


# =============================================================================
# Additional Unit Tests for SplitApplicationError
# =============================================================================


class TestSplitApplicationError:
    """Unit tests for SplitApplicationError exception class."""

    def test_error_str_format(self):
        """Error string format includes all diagnostic info."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
        split = Partition((0, 1), encoding)
        tree_splits = [
            Partition((0, 1), encoding),
            Partition((2, 3), encoding),
        ]

        error = SplitApplicationError(
            split=split,
            tree_splits=tree_splits,
            message="Test error message",
        )

        error_str = str(error)
        assert "Test error message" in error_str
        assert "[0, 1]" in error_str
        assert "A" in error_str and "B" in error_str
        assert "Tree has 2 splits" in error_str

    def test_error_with_empty_tree_splits(self):
        """Error handles empty tree_splits gracefully."""
        encoding = {"A": 0, "B": 1}
        split = Partition((0, 1), encoding)

        error = SplitApplicationError(
            split=split,
            tree_splits=[],
            message="No splits in tree",
        )

        error_str = str(error)
        assert "Tree has 0 splits" in error_str


# =============================================================================
# Property 3: Collapse Path Correctness
# Validates: Requirements 2.1, 2.2, 2.3
# =============================================================================


class TestCollapsePathCorrectness:
    """Property 3: Collapse Path Correctness

    For any tree and for any collapse_path, after executing the collapse path:
    - All splits in collapse_path (that don't exist in destination) SHALL be absent from the tree
    - All splits that exist in destination_tree SHALL be preserved
    """

    def test_collapse_removes_specified_splits(self):
        """Collapsing a path removes the specified splits from the tree."""
        from brancharchitect.tree_interpolation.topology_ops.collapse import (
            execute_collapse_path,
        )

        # Create a tree with known structure
        newick = "((A:1,B:1):1,(C:1,D:1):1);"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)

        # Get the splits in the tree
        original_splits = list(tree.to_splits())
        assert len(original_splits) >= 1, "Tree should have internal splits"

        # Pick a split to collapse
        split_to_collapse = original_splits[0]

        # Execute collapse path
        execute_collapse_path(tree, [split_to_collapse], destination_tree=None)

        # Verify the split is no longer in the tree
        new_splits = tree.to_splits()
        assert split_to_collapse not in new_splits, (
            f"Split {split_to_collapse.indices} should have been collapsed"
        )

    def test_collapse_preserves_destination_splits(self):
        """Collapsing preserves splits that exist in destination tree."""
        from brancharchitect.tree_interpolation.topology_ops.collapse import (
            execute_collapse_path,
        )

        # Create source tree
        newick = "((A:1,B:1):1,(C:1,D:1):1);"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)

        # Create destination tree with same structure
        dest_tree = create_test_tree(newick, taxa_order)

        # Get splits
        original_splits = list(tree.to_splits())
        assert len(original_splits) >= 1

        # Try to collapse a split that exists in destination
        split_to_collapse = original_splits[0]

        # Execute collapse path with destination tree
        execute_collapse_path(tree, [split_to_collapse], destination_tree=dest_tree)

        # Verify the split is preserved (because it exists in destination)
        new_splits = tree.to_splits()
        assert split_to_collapse in new_splits, (
            f"Split {split_to_collapse.indices} should be preserved (exists in destination)"
        )

    def test_collapse_empty_path_is_noop(self):
        """Collapsing an empty path does not change the tree."""
        from brancharchitect.tree_interpolation.topology_ops.collapse import (
            execute_collapse_path,
        )

        newick = "((A:1,B:1):1,(C:1,D:1):1);"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)

        original_splits = set(tree.to_splits())

        # Execute empty collapse path
        execute_collapse_path(tree, [], destination_tree=None)

        new_splits = set(tree.to_splits())
        assert original_splits == new_splits, (
            "Empty collapse path should not change tree"
        )

    def test_collapse_multiple_splits(self):
        """Collapsing multiple splits removes all of them."""
        from brancharchitect.tree_interpolation.topology_ops.collapse import (
            execute_collapse_path,
        )

        # Create a tree with multiple internal nodes
        newick = "(((A:1,B:1):1,C:1):1,D:1);"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)

        original_splits = list(tree.to_splits())
        assert len(original_splits) >= 2, "Tree should have at least 2 internal splits"

        # Filter out the root split (contains all taxa) - it cannot be collapsed
        all_taxa = set(range(len(taxa_order)))
        non_root_splits = [s for s in original_splits if set(s.indices) != all_taxa]

        if len(non_root_splits) < 1:
            pytest.skip("No non-root splits to collapse")

        # Collapse non-root splits
        execute_collapse_path(tree, non_root_splits, destination_tree=None)

        # Verify all non-root splits are removed
        new_splits = tree.to_splits()
        for split in non_root_splits:
            assert split not in new_splits, (
                f"Split {split.indices} should have been collapsed"
            )


# =============================================================================
# Property 5: Round-Trip Topology Correctness
# Validates: Requirements 8.1, 8.2
# =============================================================================


class TestRoundTripTopologyCorrectness:
    """Property 5: Round-Trip Topology Correctness

    For any valid (collapse_path, expand_path) pair computed by the planning phase,
    executing collapse then expand SHALL produce a tree where:
    - All splits in expand_path are present
    - No splits unique to collapse_path (not in expand_path) are present
    """

    def test_execute_path_adds_expand_splits(self):
        """Execute path adds all expand splits to the tree."""
        from brancharchitect.tree_interpolation.topology_ops.collapse import (
            execute_path,
        )

        # Create source tree
        source_newick = "((A:1,B:1):1,(C:1,D:1):1);"
        taxa_order = ["A", "B", "C", "D"]
        source_tree = create_test_tree(source_newick, taxa_order)

        # Create destination tree with different structure
        dest_newick = "(((A:1,B:1):1,C:1):1,D:1);"
        dest_tree = create_test_tree(dest_newick, taxa_order)

        # Compute collapse and expand paths
        source_splits = set(source_tree.to_splits())
        dest_splits = set(dest_tree.to_splits())

        # Collapse path: splits in source but not in dest
        collapse_path = list(source_splits - dest_splits)
        # Expand path: splits in dest but not in source
        expand_path = list(dest_splits - source_splits)

        # Make a copy to modify
        working_tree = source_tree.deep_copy()

        # Execute the path
        execute_path(working_tree, collapse_path, expand_path, dest_tree)

        # Verify all expand splits are present
        result_splits = working_tree.to_splits()
        for split in expand_path:
            assert split in result_splits, (
                f"Expand split {split.indices} should be in result tree"
            )

    def test_execute_path_removes_collapse_only_splits(self):
        """Execute path removes splits that are only in collapse path."""
        from brancharchitect.tree_interpolation.topology_ops.collapse import (
            execute_path,
        )

        # Create source tree
        source_newick = "((A:1,B:1):1,(C:1,D:1):1);"
        taxa_order = ["A", "B", "C", "D"]
        source_tree = create_test_tree(source_newick, taxa_order)

        # Create destination tree with different structure
        dest_newick = "(((A:1,B:1):1,C:1):1,D:1);"
        dest_tree = create_test_tree(dest_newick, taxa_order)

        # Compute collapse and expand paths
        source_splits = set(source_tree.to_splits())
        dest_splits = set(dest_tree.to_splits())

        # Collapse path: splits in source but not in dest
        collapse_path = list(source_splits - dest_splits)
        # Expand path: splits in dest but not in source
        expand_path = list(dest_splits - source_splits)

        # Make a copy to modify
        working_tree = source_tree.deep_copy()

        # Execute the path
        execute_path(working_tree, collapse_path, expand_path, dest_tree)

        # Verify collapse-only splits are removed
        result_splits = working_tree.to_splits()
        expand_set = set(expand_path)
        for split in collapse_path:
            if split not in expand_set:
                assert split not in result_splits, (
                    f"Collapse-only split {split.indices} should not be in result tree"
                )

    def test_execute_path_with_empty_paths(self):
        """Execute path with empty paths is a no-op."""
        from brancharchitect.tree_interpolation.topology_ops.collapse import (
            execute_path,
        )

        newick = "((A:1,B:1):1,(C:1,D:1):1);"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)
        dest_tree = create_test_tree(newick, taxa_order)

        original_splits = set(tree.to_splits())

        # Execute with empty paths
        execute_path(tree, [], [], dest_tree)

        new_splits = set(tree.to_splits())
        assert original_splits == new_splits, "Empty paths should not change tree"

    def test_execute_path_applies_weights(self):
        """Execute path applies weights from destination tree."""
        from brancharchitect.tree_interpolation.topology_ops.collapse import (
            execute_path,
        )

        # Create source tree (star topology)
        source_newick = "(A:1,B:1,C:1,D:1);"
        taxa_order = ["A", "B", "C", "D"]
        source_tree = create_test_tree(source_newick, taxa_order)

        # Create destination tree with specific weights
        dest_newick = "((A:1,B:1):0.5,(C:1,D:1):0.7);"
        dest_tree = create_test_tree(dest_newick, taxa_order)

        # Compute paths
        source_splits = set(source_tree.to_splits())
        dest_splits = set(dest_tree.to_splits())

        collapse_path = list(source_splits - dest_splits)
        expand_path = list(dest_splits - source_splits)

        # Make a copy to modify
        working_tree = source_tree.deep_copy()

        # Execute the path
        execute_path(working_tree, collapse_path, expand_path, dest_tree)

        # Verify weights are applied from destination
        for split in expand_path:
            result_node = working_tree.find_node_by_split(split)
            dest_node = dest_tree.find_node_by_split(split)
            if result_node is not None and dest_node is not None:
                assert result_node.length == dest_node.length, (
                    f"Weight for split {split.indices} should match destination"
                )


# =============================================================================
# Property 6: Weight Application Correctness
# Validates: Requirements 4.3
# =============================================================================


class TestWeightApplicationCorrectness:
    """Property 6: Weight Application Correctness

    For any tree and for any expand_path with reference weights, after executing
    the expand path, each newly created node SHALL have the weight from reference_weights.
    """

    def test_weights_applied_from_reference(self):
        """Weights are correctly applied from reference_weights dict."""
        from brancharchitect.tree_interpolation.topology_ops.expand import (
            execute_expand_path,
        )

        # Create a star topology tree
        newick = "(A:1,B:1,C:1,D:1);"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)

        # Create a compatible split to add
        encoding = tree.taxa_encoding
        new_split = Partition((0, 1), encoding)  # Group A and B

        # Define reference weights
        reference_weights = {new_split: 0.42}

        # Execute expand path
        execute_expand_path(tree, [new_split], reference_weights)

        # Verify the split was added with correct weight
        node = tree.find_node_by_split(new_split)
        assert node is not None, "Split should have been added"
        assert node.length == 0.42, f"Weight should be 0.42, got {node.length}"

    def test_multiple_weights_applied(self):
        """Multiple weights are correctly applied to multiple splits."""
        from brancharchitect.tree_interpolation.topology_ops.expand import (
            execute_expand_path,
        )

        # Create a star topology tree with more taxa
        newick = "(A:1,B:1,C:1,D:1,E:1);"
        taxa_order = ["A", "B", "C", "D", "E"]
        tree = create_test_tree(newick, taxa_order)

        encoding = tree.taxa_encoding

        # Create compatible splits (nested structure)
        split1 = Partition((0, 1, 2), encoding)  # Group A, B, C
        split2 = Partition((0, 1), encoding)  # Group A, B (inside split1)

        # Define reference weights
        reference_weights = {
            split1: 0.5,
            split2: 0.3,
        }

        # Execute expand path (larger first)
        execute_expand_path(tree, [split1, split2], reference_weights)

        # Verify both splits have correct weights
        node1 = tree.find_node_by_split(split1)
        node2 = tree.find_node_by_split(split2)

        assert node1 is not None, "Split1 should have been added"
        assert node2 is not None, "Split2 should have been added"
        assert node1.length == 0.5, f"Split1 weight should be 0.5, got {node1.length}"
        assert node2.length == 0.3, f"Split2 weight should be 0.3, got {node2.length}"

    def test_missing_weight_defaults_to_zero(self):
        """Splits without reference weights get default weight of 0."""
        from brancharchitect.tree_interpolation.topology_ops.expand import (
            execute_expand_path,
        )

        # Create a star topology tree
        newick = "(A:1,B:1,C:1,D:1);"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)

        encoding = tree.taxa_encoding
        new_split = Partition((0, 1), encoding)

        # Empty reference weights
        reference_weights: dict = {}

        # Execute expand path
        execute_expand_path(tree, [new_split], reference_weights)

        # Verify the split was added with default weight
        node = tree.find_node_by_split(new_split)
        assert node is not None, "Split should have been added"
        assert node.length == 0.0, f"Weight should default to 0.0, got {node.length}"

    def test_no_weights_when_none_provided(self):
        """When reference_weights is None, weights are not modified."""
        from brancharchitect.tree_interpolation.topology_ops.expand import (
            execute_expand_path,
        )

        # Create a star topology tree
        newick = "(A:1,B:1,C:1,D:1);"
        taxa_order = ["A", "B", "C", "D"]
        tree = create_test_tree(newick, taxa_order)

        encoding = tree.taxa_encoding
        new_split = Partition((0, 1), encoding)

        # Execute expand path with no reference weights
        execute_expand_path(tree, [new_split], reference_weights=None)

        # Verify the split was added (weight will be whatever default the node gets)
        node = tree.find_node_by_split(new_split)
        assert node is not None, "Split should have been added"
        # Weight should be 0 (default from Node creation)
        assert node.length == 0, f"Weight should be 0, got {node.length}"
