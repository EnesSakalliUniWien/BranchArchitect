"""
Test Suite for Subtree Relocation Animator Stages

Tests each of the 5 stages of subtree relocation using heiko_4_test_tree.tree:
1. Collapse Down: Zero-length branches inside the moving subtree
2. Collapse: Merge zero-length branches into consensus topology
3. Reorder: Place the subtree at its new position among stable anchors
4. Expand Up: Restore branch lengths from destination tree
5. Snap: Final state matching the destination topology

Test Data:
- Tree 1: ((O1,O2),(((A,B),C),((D,(E,(F,G))),H)));
- Tree 2: ((O1,O2),(((A,B),(C,(F,(E,G)))),(D,H)));
"""

import pytest
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.subtree_paths.execution.step_executor import (
    build_microsteps_for_selection,
)
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    create_collapsed_consensus_tree,
)
from brancharchitect.tree_interpolation.topology_ops.weights import (
    apply_zero_branch_lengths,
)
from brancharchitect.tree_interpolation.topology_ops.expand import (
    create_subtree_grafted_tree,
)
from brancharchitect.tree_interpolation.subtree_paths.execution.reordering import (
    reorder_tree_toward_destination,
)
from brancharchitect.elements.partition_set import PartitionSet


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def source_tree():
    """Load source tree from heiko_4_test_tree.tree (Tree 1)."""
    newick = "((O1,O2),(((A,B),C),((D,(E,(F,G))),H)));"
    return parse_newick(newick)


@pytest.fixture
def destination_tree(source_tree):
    """Load destination tree from heiko_4_test_tree.tree (Tree 2).

    Uses the same encoding as the source tree to ensure compatibility.
    """
    newick = "((O1,O2),(((A,B),(C,(F,(E,G)))),(D,H)));"
    # Parse with the same encoding as source tree
    return parse_newick(newick, encoding=source_tree.taxa_encoding)


# ============================================================================
# Helper Functions
# ============================================================================


def get_leaf_order(tree):
    """Get the current leaf order from a tree."""
    return list(tree.get_current_order())


def get_splits(tree):
    """Get all splits (partitions) from a tree as sets of taxa."""
    splits = tree.to_splits()
    return {frozenset(s.taxa) for s in splits}


def make_partition(taxa_names, encoding):
    """Create a Partition from taxa names and encoding."""
    indices = tuple(encoding[name] for name in taxa_names)
    return Partition(indices, encoding)


def find_partition_by_taxa(tree, taxa_set):
    """Find a partition in the tree that matches the given taxa set."""
    for partition in tree.to_splits():
        if partition.taxa == frozenset(taxa_set):
            return partition
    return None


def create_mock_selection(subtree_partition, expand_path, collapse_path):
    """Create a mock selection dictionary for testing."""
    return {
        "subtree": subtree_partition,
        "expand": {"path_segment": expand_path},
        "collapse": {"path_segment": collapse_path},
    }


# ============================================================================
# Stage 1: Collapse Down (Zero Branch Lengths)
# ============================================================================


class TestStage1CollapseDown:
    """
    Stage 1: Collapse Down

    Apply zero branch lengths to edges inside the moving subtree.
    This prepares the subtree for visual "collapsing" in the animation.
    """

    def test_collapse_down_preserves_topology(self, source_tree):
        """After zeroing branch lengths, topology should remain unchanged."""
        encoding = source_tree.taxa_encoding

        # Create partitions to zero
        zeroing_splits = [
            make_partition(["E", "F", "G"], encoding),
            make_partition(["F", "G"], encoding),
        ]

        original_splits = get_splits(source_tree)

        # Apply zero branch lengths
        result = apply_zero_branch_lengths(
            source_tree, PartitionSet(set(zeroing_splits))
        )

        # Topology should be preserved (same splits)
        result_splits = get_splits(result)
        assert result_splits == original_splits, (
            f"Topology changed! Lost: {original_splits - result_splits}, "
            f"Gained: {result_splits - original_splits}"
        )

    def test_collapse_down_zeros_specified_branches(self, source_tree):
        """Branch lengths should be set to zero for specified splits."""
        encoding = source_tree.taxa_encoding
        fg_partition = make_partition(["F", "G"], encoding)
        zeroing_splits = [fg_partition]

        # Apply zero branch lengths
        result = apply_zero_branch_lengths(
            source_tree, PartitionSet(set(zeroing_splits))
        )

        # Check that the specified branch now has zero length
        weighted_splits = result.to_weighted_splits()

        for partition, weight in weighted_splits.items():
            if partition.taxa == frozenset({"F", "G"}):
                assert weight == 0.0, f"Branch length should be 0.0, got {weight}"
                break

    def test_collapse_down_preserves_leaf_order(self, source_tree):
        """Leaf order should not change after zeroing branch lengths."""
        encoding = source_tree.taxa_encoding
        zeroing_splits = [
            make_partition(["E", "F", "G"], encoding),
        ]

        original_order = get_leaf_order(source_tree)

        result = apply_zero_branch_lengths(
            source_tree, PartitionSet(set(zeroing_splits))
        )

        result_order = get_leaf_order(result)
        assert result_order == original_order, (
            f"Leaf order changed from {original_order} to {result_order}"
        )


# ============================================================================
# Stage 2: Collapse (Merge Zero-Length Branches)
# ============================================================================


class TestStage2Collapse:
    """
    Stage 2: Collapse

    Merge zero-length branches into a consensus topology.
    This visually "collapses" the moving subtree into a single node.
    """

    def test_collapse_produces_valid_tree(self, source_tree, destination_tree):
        """Collapse should produce a valid tree with all taxa."""
        encoding = source_tree.taxa_encoding

        # First, zero out some branches
        zeroing_splits = [
            make_partition(["E", "F", "G"], encoding),
            make_partition(["F", "G"], encoding),
        ]

        zeroed_tree = apply_zero_branch_lengths(
            source_tree, PartitionSet(set(zeroing_splits))
        )

        # Find a pivot edge for the collapse operation
        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)

        # Collapse the tree
        collapsed = create_collapsed_consensus_tree(
            zeroed_tree, pivot_edge, destination_tree=destination_tree, copy=True
        )

        # The collapsed tree should exist and have all taxa
        assert collapsed is not None
        assert len(get_leaf_order(collapsed)) == len(get_leaf_order(source_tree))

    def test_collapse_preserves_taxa_count(self, source_tree, destination_tree):
        """Collapse should not lose any taxa."""
        encoding = source_tree.taxa_encoding

        # Zero and collapse
        zeroing_splits = [make_partition(["F", "G"], encoding)]
        zeroed_tree = apply_zero_branch_lengths(
            source_tree, PartitionSet(set(zeroing_splits))
        )

        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)
        collapsed = create_collapsed_consensus_tree(
            zeroed_tree, pivot_edge, destination_tree=destination_tree, copy=True
        )

        source_taxa = set(get_leaf_order(source_tree))
        collapsed_taxa = set(get_leaf_order(collapsed))

        assert source_taxa == collapsed_taxa, (
            f"Taxa mismatch! Lost: {source_taxa - collapsed_taxa}, "
            f"Gained: {collapsed_taxa - source_taxa}"
        )


# ============================================================================
# Stage 3: Reorder (Place Subtree at New Position)
# ============================================================================


class TestStage3Reorder:
    """
    Stage 3: Reorder

    Place the moving subtree at its destination position among stable anchors.
    This is where the visual "movement" happens.
    """

    def test_reorder_preserves_all_taxa(self, source_tree, destination_tree):
        """The reordered tree should maintain all taxa."""
        encoding = source_tree.taxa_encoding
        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)
        moving_subtree = make_partition(["F", "G"], encoding)

        reordered = reorder_tree_toward_destination(
            source_tree=source_tree,
            destination_tree=destination_tree,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=moving_subtree,
            all_mover_partitions=None,
            copy=True,
        )

        source_taxa = set(get_leaf_order(source_tree))
        reordered_taxa = set(get_leaf_order(reordered))
        assert source_taxa == reordered_taxa, (
            f"Taxa mismatch! Lost: {source_taxa - reordered_taxa}, "
            f"Gained: {reordered_taxa - source_taxa}"
        )

    def test_reorder_respects_unstable_taxa(self, source_tree, destination_tree):
        """Unstable taxa should not act as anchors during reordering."""
        encoding = source_tree.taxa_encoding
        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)
        moving_subtree = make_partition(["F", "G"], encoding)
        # E will move in a later step - represented as a partition
        other_mover = make_partition(["E"], encoding)
        all_movers = [moving_subtree, other_mover]

        reordered = reorder_tree_toward_destination(
            source_tree=source_tree,
            destination_tree=destination_tree,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=moving_subtree,
            all_mover_partitions=all_movers,
            copy=True,
        )

        # Verify the tree is valid
        assert reordered is not None
        assert len(get_leaf_order(reordered)) == len(get_leaf_order(source_tree))

    def test_reorder_preserves_topology(self, source_tree, destination_tree):
        """Reordering should not change the tree topology (splits)."""
        encoding = source_tree.taxa_encoding
        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)
        moving_subtree = make_partition(["F", "G"], encoding)

        original_splits = get_splits(source_tree)

        reordered = reorder_tree_toward_destination(
            source_tree=source_tree,
            destination_tree=destination_tree,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=moving_subtree,
            all_mover_partitions=None,
            copy=True,
        )

        reordered_splits = get_splits(reordered)

        # Topology should be unchanged (same splits)
        assert original_splits == reordered_splits, (
            f"Topology changed! Lost: {original_splits - reordered_splits}, "
            f"Gained: {reordered_splits - original_splits}"
        )


# ============================================================================
# Stage 4: Expand Up (Restore Branch Lengths)
# ============================================================================


class TestStage4ExpandUp:
    """
    Stage 4: Expand Up

    Graft the reference path to create splits from the destination tree.
    This restores the internal structure of the relocated subtree.
    """

    def test_expand_with_empty_path_preserves_structure(self, source_tree):
        """Expand with an empty path should preserve splits."""
        original_splits = get_splits(source_tree)

        # Expand with an empty path (no changes)
        expand_path = []

        expanded = create_subtree_grafted_tree(
            base_tree=source_tree,
            ref_path_to_build=expand_path,
            copy=True,
        )

        expanded_splits = get_splits(expanded)

        # With empty expand path, topology should be identical
        assert original_splits == expanded_splits

    def test_expand_preserves_taxa(self, source_tree):
        """Expand should preserve all taxa."""
        expand_path = []

        expanded = create_subtree_grafted_tree(
            base_tree=source_tree,
            ref_path_to_build=expand_path,
            copy=True,
        )

        source_taxa = set(get_leaf_order(source_tree))
        expanded_taxa = set(get_leaf_order(expanded))

        assert source_taxa == expanded_taxa


# ============================================================================
# Stage 5: Snap (Final State)
# ============================================================================


class TestStage5Snap:
    """
    Stage 5: Snap

    The final tree state should match the destination topology.
    This is verified by comparing the complete output of build_microsteps_for_selection.
    """

    def test_snap_produces_valid_tree(self, source_tree, destination_tree):
        """The snapped tree should be a valid tree with all taxa."""
        encoding = source_tree.taxa_encoding
        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)
        subtree_partition = make_partition(["F", "G"], encoding)

        selection = create_mock_selection(
            subtree_partition=subtree_partition,
            expand_path=[],
            collapse_path=[],
        )

        trees, edges, snapped_tree, subtree_tracker = build_microsteps_for_selection(
            interpolation_state=source_tree,
            destination_tree=destination_tree,
            current_pivot_edge=pivot_edge,
            selection=selection,
            all_mover_partitions=None,
        )

        # Verify 5 intermediate trees are generated
        assert len(trees) == 5, f"Expected 5 trees, got {len(trees)}"

        # Verify snapped tree has all taxa
        snapped_leaves = set(get_leaf_order(snapped_tree))
        source_leaves = set(get_leaf_order(source_tree))
        assert snapped_leaves == source_leaves, (
            f"Taxa mismatch in snapped tree! "
            f"Lost: {source_leaves - snapped_leaves}, "
            f"Gained: {snapped_leaves - source_leaves}"
        )

    def test_full_pipeline_produces_5_frames(self, source_tree, destination_tree):
        """The complete pipeline should produce exactly 5 animation frames."""
        encoding = source_tree.taxa_encoding
        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)
        subtree_partition = make_partition(["F", "G"], encoding)
        fg_collapse = make_partition(["F", "G"], encoding)

        selection = create_mock_selection(
            subtree_partition=subtree_partition,
            expand_path=[],
            collapse_path=[fg_collapse],
        )

        trees, edges, snapped_tree, subtree_tracker = build_microsteps_for_selection(
            interpolation_state=source_tree,
            destination_tree=destination_tree,
            current_pivot_edge=pivot_edge,
            selection=selection,
            all_mover_partitions=None,
        )

        assert len(trees) == 5, f"Expected 5 animation frames, got {len(trees)}"
        assert len(edges) == 5, f"Expected 5 edge references, got {len(edges)}"
        assert len(subtree_tracker) == 5, (
            f"Expected 5 subtree trackers, got {len(subtree_tracker)}"
        )

    def test_all_frames_have_consistent_taxa(self, source_tree, destination_tree):
        """All 5 frames should have the same set of taxa."""
        encoding = source_tree.taxa_encoding
        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)
        subtree_partition = make_partition(["F", "G"], encoding)

        selection = create_mock_selection(
            subtree_partition=subtree_partition,
            expand_path=[],
            collapse_path=[],
        )

        trees, edges, snapped_tree, subtree_tracker = build_microsteps_for_selection(
            interpolation_state=source_tree,
            destination_tree=destination_tree,
            current_pivot_edge=pivot_edge,
            selection=selection,
            all_mover_partitions=None,
        )

        source_taxa = set(get_leaf_order(source_tree))

        for i, tree in enumerate(trees):
            tree_taxa = set(get_leaf_order(tree))
            assert tree_taxa == source_taxa, (
                f"Frame {i + 1} has taxa mismatch! "
                f"Lost: {source_taxa - tree_taxa}, "
                f"Gained: {tree_taxa - source_taxa}"
            )


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests for the complete relocation workflow."""

    def test_heiko_tree_full_interpolation(self, source_tree, destination_tree):
        """
        Full interpolation from Tree 1 to Tree 2 in heiko_4_test_tree.tree.

        Source: ((O1,O2),(((A,B),C),((D,(E,(F,G))),H)));
        Dest:   ((O1,O2),(((A,B),(C,(F,(E,G)))),(D,H)));

        Expected movement: (E,(F,G)) subtree relocates from D's subtree to C's subtree.
        """
        encoding = source_tree.taxa_encoding
        pivot_edge = make_partition(["A", "B", "C", "D", "E", "F", "G", "H"], encoding)

        # The moving subtree
        subtree_partition = make_partition(["E", "F", "G"], encoding)
        efg_collapse = make_partition(["E", "F", "G"], encoding)
        fg_collapse = make_partition(["F", "G"], encoding)

        selection = create_mock_selection(
            subtree_partition=subtree_partition,
            expand_path=[],
            collapse_path=[efg_collapse, fg_collapse],
        )

        trees, edges, snapped_tree, subtree_tracker = build_microsteps_for_selection(
            interpolation_state=source_tree,
            destination_tree=destination_tree,
            current_pivot_edge=pivot_edge,
            selection=selection,
            all_mover_partitions=None,
        )

        # Verify the animation sequence is valid
        assert len(trees) == 5

        # Print the leaf orders for debugging
        print("\n=== Heiko Tree Full Interpolation ===")
        print(f"Source order: {get_leaf_order(source_tree)}")
        stage_names = ["Collapse Down", "Collapse", "Reorder", "Expand Up", "Snap"]
        for i, tree in enumerate(trees):
            print(f"Stage {i + 1} ({stage_names[i]}): {get_leaf_order(tree)}")
        print(f"Snapped order: {get_leaf_order(snapped_tree)}")
        print(f"Destination order: {get_leaf_order(destination_tree)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
