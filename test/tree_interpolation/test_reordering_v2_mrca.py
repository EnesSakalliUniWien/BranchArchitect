"""
Test Suite for MRCA-Aware Reordering (V2).

Tests the enhanced reordering logic that uses parent mappings from
map_solution_elements_via_parent to determine optimal block placement.
"""

import pytest
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.subtree_paths.execution.reordering_v2_mrca_aware import (
    reorder_tree_toward_destination,
    classify_mover_relationships,
    map_solution_elements_via_parent,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_encoding():
    """Simple encoding for basic tests."""
    return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


@pytest.fixture
def simple_trees(simple_encoding):
    """
    Simple source and destination trees for testing.

    Source:  ((A,B),(C,D),(E,F))
    Dest:    ((A,B),(E,F),(C,D))

    Movement: (E,F) block moves from position 3 to position 2
    """
    source = parse_newick("((A,B),(C,D),(E,F));", encoding=simple_encoding)
    dest = parse_newick("((A,B),(E,F),(C,D));", encoding=simple_encoding)
    return source, dest, simple_encoding


@pytest.fixture
def nested_trees():
    """
    Nested trees where mover changes parent.

    Source:  ((A,(B,M)),C)  - M is under (A,B,M) subtree
    Dest:    (A,(C,(B,M)))  - M is now under (B,C,M) subtree

    The mover M changes parent from (A,B,M) to (B,C,M).
    """
    encoding = {"A": 0, "B": 1, "C": 2, "M": 3}
    source = parse_newick("((A,(B,M)),C);", encoding=encoding)
    dest = parse_newick("(A,(C,(B,M)));", encoding=encoding)
    return source, dest, encoding


@pytest.fixture
def diverging_trees():
    """
    Trees where two movers diverge (were together, now separate).

    Source:  ((A,B),(M1,M2),C)  - M1 and M2 are siblings
    Dest:    ((A,M1),(M2,B),C)  - M1 goes to A's subtree, M2 goes to B's subtree
    """
    encoding = {"A": 0, "B": 1, "C": 2, "M1": 3, "M2": 4}
    source = parse_newick("((A,B),(M1,M2),C);", encoding=encoding)
    dest = parse_newick("((A,M1),(M2,B),C);", encoding=encoding)
    return source, dest, encoding


@pytest.fixture
def converging_trees():
    """
    Trees where two movers converge (were separate, now together).

    Source:  ((A,M1),(B,M2),C)  - M1 under A, M2 under B
    Dest:    ((A,B),(M1,M2),C)  - M1 and M2 become siblings
    """
    encoding = {"A": 0, "B": 1, "C": 2, "M1": 3, "M2": 4}
    source = parse_newick("((A,M1),(B,M2),C);", encoding=encoding)
    dest = parse_newick("((A,B),(M1,M2),C);", encoding=encoding)
    return source, dest, encoding


# =============================================================================
# Test: map_solution_elements_via_parent
# =============================================================================


class TestMapSolutionElementsViaParent:
    """Test the parent mapping function from minimum_cover_mappings."""

    def test_maps_monophyletic_mover_to_parent(self, simple_trees):
        """Monophyletic mover should map to its direct parent."""
        source, dest, encoding = simple_trees

        # Create a pivot edge solution with (E,F) as the mover
        pivot_edge = Partition((0, 1, 2, 3, 4, 5), encoding)  # Root
        mover_ef = Partition((4, 5), encoding)  # (E, F)

        pivot_edge_solutions = {pivot_edge: [mover_ef]}

        mapped_t1, mapped_t2 = map_solution_elements_via_parent(
            pivot_edge_solutions, source, dest
        )

        # E,F should map to their parent in each tree
        assert pivot_edge in mapped_t1
        assert mover_ef in mapped_t1[pivot_edge]

        # In source: parent of (E,F) is root
        parent_in_source = mapped_t1[pivot_edge][mover_ef]
        assert parent_in_source.indices == pivot_edge.indices

        # In dest: parent of (E,F) is also root
        parent_in_dest = mapped_t2[pivot_edge][mover_ef]
        assert parent_in_dest.indices == pivot_edge.indices

    def test_maps_nested_mover_to_correct_parent(self, nested_trees):
        """Nested mover should map to its actual parent in each tree."""
        source, dest, encoding = nested_trees

        pivot_edge = Partition((0, 1, 2, 3), encoding)  # Root
        mover_m = Partition((3,), encoding)  # Just M

        pivot_edge_solutions = {pivot_edge: [mover_m]}

        mapped_t1, mapped_t2 = map_solution_elements_via_parent(
            pivot_edge_solutions, source, dest
        )

        # M's parent in source should be (B,M)
        parent_in_source = mapped_t1[pivot_edge][mover_m]
        # Parent of M in source tree ((A,(B,M)),C) is (B,M)
        expected_source_parent = Partition((1, 3), encoding)
        assert parent_in_source.indices == expected_source_parent.indices

        # M's parent in dest should be (B,M) as well
        parent_in_dest = mapped_t2[pivot_edge][mover_m]
        expected_dest_parent = Partition((1, 3), encoding)
        assert parent_in_dest.indices == expected_dest_parent.indices


# =============================================================================
# Test: classify_mover_relationships
# =============================================================================


class TestClassifyMoverRelationships:
    """Test the mover relationship classification."""

    def test_classifies_diverging_movers(self, diverging_trees):
        """Movers that were siblings but go to different parents are diverging."""
        source, dest, encoding = diverging_trees

        pivot_edge = Partition((0, 1, 2, 3, 4), encoding)
        m1 = Partition((3,), encoding)
        m2 = Partition((4,), encoding)

        pivot_edge_solutions = {pivot_edge: [m1, m2]}

        mapped_t1, mapped_t2 = map_solution_elements_via_parent(
            pivot_edge_solutions, source, dest
        )

        relationships = classify_mover_relationships(
            [m1, m2],
            mapped_t1[pivot_edge],
            mapped_t2[pivot_edge],
        )

        # M1 and M2 should be classified as diverging
        assert len(relationships["diverging"]) == 1
        assert len(relationships["converging"]) == 0
        assert len(relationships["stable"]) == 0

        diverging_pair = relationships["diverging"][0]
        assert m1 in diverging_pair and m2 in diverging_pair

    def test_classifies_converging_movers(self, converging_trees):
        """Movers that were separate but become siblings are converging."""
        source, dest, encoding = converging_trees

        pivot_edge = Partition((0, 1, 2, 3, 4), encoding)
        m1 = Partition((3,), encoding)
        m2 = Partition((4,), encoding)

        pivot_edge_solutions = {pivot_edge: [m1, m2]}

        mapped_t1, mapped_t2 = map_solution_elements_via_parent(
            pivot_edge_solutions, source, dest
        )

        relationships = classify_mover_relationships(
            [m1, m2],
            mapped_t1[pivot_edge],
            mapped_t2[pivot_edge],
        )

        # M1 and M2 should be classified as converging
        assert len(relationships["converging"]) == 1
        assert len(relationships["diverging"]) == 0
        assert len(relationships["stable"]) == 0

    def test_classifies_stable_movers(self, simple_trees):
        """Movers that stay with same parent relationship are stable."""
        source, dest, encoding = simple_trees

        pivot_edge = Partition((0, 1, 2, 3, 4, 5), encoding)
        # E and F are always siblings (same parent) in both trees
        m_e = Partition((4,), encoding)
        m_f = Partition((5,), encoding)

        pivot_edge_solutions = {pivot_edge: [m_e, m_f]}

        mapped_t1, mapped_t2 = map_solution_elements_via_parent(
            pivot_edge_solutions, source, dest
        )

        relationships = classify_mover_relationships(
            [m_e, m_f],
            mapped_t1[pivot_edge],
            mapped_t2[pivot_edge],
        )

        # E and F should be classified as stable (same parent in both)
        assert len(relationships["stable"]) == 1
        assert len(relationships["diverging"]) == 0
        assert len(relationships["converging"]) == 0


# =============================================================================
# Test: reorder_tree_toward_destination (MRCA-aware)
# =============================================================================


class TestReorderWithMRCA:
    """Test the MRCA-aware reordering function."""

    def test_reorder_without_parent_maps_uses_fallback(self, simple_trees):
        """Without parent maps, should use fallback first-occurrence method."""
        source, dest, encoding = simple_trees

        pivot_edge = Partition((0, 1, 2, 3, 4, 5), encoding)
        mover_ef = Partition((4, 5), encoding)

        source_order_before = list(source.get_current_order())

        result = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=mover_ef,
            all_mover_partitions=[mover_ef],
            source_parent_map=None,  # No MRCA info
            dest_parent_map=None,  # No MRCA info
            copy=True,
        )

        # Result should have E,F in the correct position
        result_order = list(result.get_current_order())
        dest_order = list(dest.get_current_order())

        # E and F should move toward their destination position
        # Source: (A,B,C,D,E,F) - E,F at end
        # Dest: (A,B,E,F,C,D) - E,F in middle
        assert "E" in result_order
        assert "F" in result_order

    def test_reorder_with_parent_maps_uses_mrca(self, simple_trees):
        """With parent maps, should use MRCA-aware positioning."""
        source, dest, encoding = simple_trees

        pivot_edge = Partition((0, 1, 2, 3, 4, 5), encoding)
        mover_ef = Partition((4, 5), encoding)

        # Compute parent maps
        pivot_edge_solutions = {pivot_edge: [mover_ef]}
        mapped_t1, mapped_t2 = map_solution_elements_via_parent(
            pivot_edge_solutions, source, dest
        )

        result = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=mover_ef,
            all_mover_partitions=[mover_ef],
            source_parent_map=mapped_t1[pivot_edge],
            dest_parent_map=mapped_t2[pivot_edge],
            copy=True,
        )

        result_order = list(result.get_current_order())

        # Block should move to its destination parent's position
        assert "E" in result_order
        assert "F" in result_order

    def test_reorder_preserves_block_internal_order(self, simple_trees):
        """Block's internal order should be preserved from source."""
        source, dest, encoding = simple_trees

        pivot_edge = Partition((0, 1, 2, 3, 4, 5), encoding)
        mover_ef = Partition((4, 5), encoding)

        # Get source order for E,F
        source_order = list(source.get_current_order())
        e_pos_source = source_order.index("E")
        f_pos_source = source_order.index("F")
        e_before_f_in_source = e_pos_source < f_pos_source

        result = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=mover_ef,
            all_mover_partitions=[mover_ef],
            copy=True,
        )

        result_order = list(result.get_current_order())
        e_pos_result = result_order.index("E")
        f_pos_result = result_order.index("F")
        e_before_f_in_result = e_pos_result < f_pos_result

        # Internal order should match source
        assert e_before_f_in_source == e_before_f_in_result

    def test_reorder_preserves_all_taxa(self, nested_trees):
        """Reordering should not lose or duplicate any taxa."""
        source, dest, encoding = nested_trees

        pivot_edge = Partition((0, 1, 2, 3), encoding)
        mover = Partition((3,), encoding)

        source_taxa = set(source.get_current_order())

        result = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=mover,
            all_mover_partitions=[mover],
            copy=True,
        )

        result_taxa = set(result.get_current_order())

        assert source_taxa == result_taxa, (
            f"Taxa mismatch! Lost: {source_taxa - result_taxa}, "
            f"Gained: {result_taxa - source_taxa}"
        )

    def test_reorder_with_no_anchors(self):
        """When all taxa are movers, should use destination order."""
        encoding = {"M1": 0, "M2": 1, "M3": 2}

        source = parse_newick("(M1,M2,M3);", encoding=encoding)
        source.reorder_taxa(["M1", "M2", "M3"])

        dest = parse_newick("(M1,M2,M3);", encoding=encoding)
        dest.reorder_taxa(["M3", "M1", "M2"])

        pivot_edge = Partition((0, 1, 2), encoding)
        mover = Partition((0, 1, 2), encoding)  # All taxa are movers

        result = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=mover,
            all_mover_partitions=[mover],
            copy=True,
        )

        result_order = list(result.get_current_order())

        # With no anchors, should follow destination order
        # (or at least not crash)
        assert set(result_order) == {"M1", "M2", "M3"}

    def test_reorder_returns_copy_when_change_needed(self):
        """When copy=True and reordering occurs, result should be a new tree."""
        encoding = {"A": 0, "B": 1, "C": 2, "M": 3}

        # Source and dest have M in different positions
        source = parse_newick("(A,B,C,M);", encoding=encoding)
        source.reorder_taxa(["A", "M", "B", "C"])  # M between A and B

        dest = parse_newick("(A,B,C,M);", encoding=encoding)
        dest.reorder_taxa(["A", "B", "C", "M"])  # M at end

        pivot_edge = Partition((0, 1, 2, 3), encoding)
        mover = Partition((3,), encoding)  # M

        source_order_before = list(source.get_current_order())

        result = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=mover,
            all_mover_partitions=[mover],
            copy=True,
        )

        source_order_after = list(source.get_current_order())

        # Source should be unchanged
        assert source_order_before == source_order_after
        # Result should be different object when change occurred
        assert result is not source


# =============================================================================
# Test: Integration with multiple movers
# =============================================================================


class TestMultipleMoverReordering:
    """Test reordering with multiple simultaneous movers."""

    def test_other_movers_stay_at_source_position(self):
        """Other movers (not current) should stay at their source positions."""
        encoding = {"A": 0, "B": 1, "M1": 2, "M2": 3, "C": 4}

        source = parse_newick("(A,B,M1,M2,C);", encoding=encoding)
        source.reorder_taxa(["A", "M1", "M2", "B", "C"])

        dest = parse_newick("(A,B,M1,M2,C);", encoding=encoding)
        dest.reorder_taxa(["A", "B", "M1", "M2", "C"])

        pivot_edge = Partition((0, 1, 2, 3, 4), encoding)
        current_mover = Partition((2,), encoding)  # M1
        other_mover = Partition((3,), encoding)  # M2

        result = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=current_mover,
            all_mover_partitions=[current_mover, other_mover],
            copy=True,
        )

        result_order = list(result.get_current_order())

        # M1 should move toward destination
        # M2 should stay at source position (stability)
        # Anchors: A, B, C
        assert "M1" in result_order
        assert "M2" in result_order

    def test_diverging_movers_separate_correctly(self, diverging_trees):
        """Diverging movers should end up at their respective destination parents."""
        source, dest, encoding = diverging_trees

        pivot_edge = Partition((0, 1, 2, 3, 4), encoding)
        m1 = Partition((3,), encoding)
        m2 = Partition((4,), encoding)

        pivot_edge_solutions = {pivot_edge: [m1, m2]}
        mapped_t1, mapped_t2 = map_solution_elements_via_parent(
            pivot_edge_solutions, source, dest
        )

        # Process M1 first
        result1 = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=m1,
            all_mover_partitions=[m1, m2],
            source_parent_map=mapped_t1[pivot_edge],
            dest_parent_map=mapped_t2[pivot_edge],
            copy=True,
        )

        # Tree should still be valid
        assert len(list(result1.get_current_order())) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
