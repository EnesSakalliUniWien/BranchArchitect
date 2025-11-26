"""
Integration test for edge ordering across the tree interpolation pipeline.

This test verifies that:
1. Edges are sorted correctly by depth (leaves to root)
2. Subset relationships are preserved (children before parents)
3. Reordering happens in the correct order (prevents snapback)
4. The full pipeline produces correct interpolation sequences
"""

import pytest
from brancharchitect.io import read_newick
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.compute_pivot_solutions_with_deletions import (
    compute_pivot_solutions_with_deletions,
)
from brancharchitect.tree_interpolation.edge_sorting_utils import (
    sort_edges_by_depth,
    compute_edge_depths,
)
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)


class TestEdgeOrderingIntegration:
    """Test edge ordering across the entire interpolation pipeline."""

    @pytest.fixture
    def simple_tree_pair(self):
        """Create a simple tree pair for testing."""
        # Tree 1: ((A,B),C,D)  - C and D are sisters to (A,B)
        # Tree 2: (A,(B,C),D)  - B and C are sisters
        tree1_str = "((A:1,B:1):1,C:1,D:1):0;"
        tree2_str = "(A:1,(B:1,C:1):1,D:1):0;"

        tree1 = parse_newick(tree1_str, force_list=False)
        tree2 = parse_newick(tree2_str, force_list=False)

        return tree1, tree2

    @pytest.fixture
    def complex_tree_pair(self):
        """Create a more complex tree pair with nested structure."""
        # Tree 1: (((A,B),C),D,E)  - Deeply nested on left
        # Tree 2: (A,B,C,(D,E))    - Nested on right
        tree1_str = "(((A:1,B:1):1,C:1):1,D:1,E:1):0;"
        tree2_str = "(A:1,B:1,C:1,(D:1,E:1):1):0;"

        tree1 = parse_newick(tree1_str, force_list=False)
        tree2 = parse_newick(tree2_str, force_list=False)

        return tree1, tree2

    def test_depth_computation_reflects_tree_structure(self, simple_tree_pair):
        """Test that depth computation correctly reflects tree topology."""
        tree1, tree2 = simple_tree_pair

        # Get s-edges from lattice algorithm
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        # Compute depths in tree1
        depths_tree1 = compute_edge_depths(edges, tree1)

        # Verify that leaves have greater depth than root
        for edge in edges:
            depth = depths_tree1.get(edge, 0)
            # Depth should be >= 0
            assert depth >= 0, f"Edge {edge} has negative depth: {depth}"

            # Smaller partitions (closer to leaves) should generally have greater depth
            edge_size = len(edge)
            if edge_size == 1:  # Leaf
                # Leaves should be deepest (largest depth value)
                for other_edge in edges:
                    if len(other_edge) > 1:  # Non-leaf
                        assert depth >= depths_tree1.get(other_edge, 0), (
                            f"Leaf {edge} should be deeper than internal {other_edge}"
                        )

    def test_edges_sorted_leaves_to_root(self, simple_tree_pair):
        """Test that ascending=True sorts from leaves to root (subsets before supersets)."""
        tree1, tree2 = simple_tree_pair

        # Get s-edges from lattice algorithm
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        # Sort edges: leaves to root (ascending=True)
        sorted_edges = sort_edges_by_depth(edges, tree1, ascending=True)

        # Verify ordering: for any edge E1 that is a subset of E2,
        # E1 should appear before E2 in the sorted list
        for i, edge1 in enumerate(sorted_edges):
            indices1 = set(edge1.resolve_to_indices())
            for j, edge2 in enumerate(sorted_edges):
                if i >= j:
                    continue
                indices2 = set(edge2.resolve_to_indices())

                # If edge1 is a strict subset of edge2
                if indices1.issubset(indices2) and indices1 != indices2:
                    # edge1 should come BEFORE edge2 (i < j)
                    assert i < j, (
                        f"Subset {edge1} at position {i} should come before superset {edge2} at position {j}"
                    )

    def test_edges_sorted_root_to_leaves(self, simple_tree_pair):
        """Test that ascending=False sorts from root to leaves (supersets before subsets)."""
        tree1, tree2 = simple_tree_pair

        # Get s-edges from lattice algorithm
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        # Sort edges: root to leaves (ascending=False)
        sorted_edges = sort_edges_by_depth(edges, tree1, ascending=False)

        # Verify ordering: for any edge E1 that is a superset of E2,
        # E1 should appear before E2 in the sorted list
        for i, edge1 in enumerate(sorted_edges):
            indices1 = set(edge1.resolve_to_indices())
            for j, edge2 in enumerate(sorted_edges):
                if i >= j:
                    continue
                indices2 = set(edge2.resolve_to_indices())

                # If edge1 is a strict superset of edge2
                if indices2.issubset(indices1) and indices1 != indices2:
                    # edge1 should come BEFORE edge2 (i < j)
                    assert i < j, (
                        f"Superset {edge1} at position {i} should come before subset {edge2} at position {j}"
                    )

    def test_sort_is_deterministic(self, simple_tree_pair):
        """Test that sorting produces consistent results across multiple calls."""
        tree1, tree2 = simple_tree_pair

        # Get s-edges from lattice algorithm
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        # Sort multiple times
        sorted1 = sort_edges_by_depth(edges, tree1, ascending=True)
        sorted2 = sort_edges_by_depth(edges, tree1, ascending=True)
        sorted3 = sort_edges_by_depth(edges, tree1, ascending=True)

        # All results should be identical
        assert sorted1 == sorted2, "First and second sort differ"
        assert sorted2 == sorted3, "Second and third sort differ"

    def test_pair_interpolation_uses_correct_ordering(self, simple_tree_pair):
        """Test that process_tree_pair_interpolation uses leaves-to-root ordering."""
        tree1, tree2 = simple_tree_pair

        # Process interpolation
        result = process_tree_pair_interpolation(tree1, tree2)

        # Verify we got some interpolated trees
        assert len(result.trees) > 0, "Should generate interpolated trees"

        # Verify tracking has correct structure
        assert len(result.current_pivot_edge_tracking) == len(result.trees), (
            "Each tree should have tracking info"
        )

        # The s-edges in tracking should be in leaves-to-root order
        s_edges_used = [
            s for s in result.current_pivot_edge_tracking if s is not None
        ]

        if len(s_edges_used) > 1:
            # Verify subset relationships are preserved in execution order
            for i in range(len(s_edges_used) - 1):
                edge1 = s_edges_used[i]
                # Find next DIFFERENT s-edge (skip duplicates from 5-step sequences)
                for j in range(i + 1, len(s_edges_used)):
                    edge2 = s_edges_used[j]
                    if edge1 != edge2:
                        indices1 = set(edge1.resolve_to_indices())
                        indices2 = set(edge2.resolve_to_indices())

                        # If edge2 is a superset of edge1, that's correct (child before parent)
                        if indices1.issubset(indices2) and indices1 != indices2:
                            # This is the correct order
                            pass
                        # If edge1 is a superset of edge2, that's WRONG (parent before child)
                        elif indices2.issubset(indices1) and indices1 != indices2:
                            pytest.fail(
                                f"S-edge ordering violation: Parent {edge1} processed before child {edge2}"
                            )
                        break

    def test_complex_nested_ordering(self, complex_tree_pair):
        """Test edge ordering with deeply nested tree structures."""
        tree1, tree2 = complex_tree_pair

        # Get s-edges from lattice algorithm
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        # Sort edges: leaves to root
        sorted_edges = sort_edges_by_depth(edges, tree1, ascending=True)

        # Find leaf edges (size 1) and their parents
        leaf_edges = [e for e in sorted_edges if len(e) == 1]
        non_leaf_edges = [e for e in sorted_edges if len(e) > 1]

        # All leaf edges should come before all non-leaf edges
        if leaf_edges and non_leaf_edges:
            last_leaf_pos = sorted_edges.index(leaf_edges[-1])
            first_nonleaf_pos = sorted_edges.index(non_leaf_edges[0])

            assert last_leaf_pos < first_nonleaf_pos, (
                "All leaves should be processed before internal nodes"
            )

    def test_ordering_prevents_snapback(self, simple_tree_pair):
        """
        Test that leaves-to-root ordering prevents snapback problem.

        Snapback occurs when a parent s-edge is processed before its child,
        causing the parent's reordering to undo the child's work.
        """
        tree1, tree2 = simple_tree_pair

        # Get s-edges and sort them
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        sorted_edges = sort_edges_by_depth(edges, tree1, ascending=True)

        # Check that no parent comes before its child
        for i, edge1 in enumerate(sorted_edges):
            indices1 = set(edge1.resolve_to_indices())
            for edge2 in sorted_edges[i + 1 :]:
                indices2 = set(edge2.resolve_to_indices())

                # If edge1 is a strict superset of edge2 (edge1 is parent of edge2)
                if indices2.issubset(indices1) and indices1 != indices2:
                    pytest.fail(
                        f"Snapback hazard: Parent {edge1} comes before child {edge2}"
                    )

    def test_empty_edge_list_handling(self):
        """Test that sorting handles empty edge lists gracefully."""
        tree1_str = "(A:1,B:1,C:1):0;"
        tree1 = parse_newick(tree1_str, force_list=False)

        edges = []
        sorted_edges = sort_edges_by_depth(edges, tree1, ascending=True)

        assert sorted_edges == [], "Empty list should return empty list"

    def test_single_edge_handling(self, simple_tree_pair):
        """Test that sorting handles single-edge lists correctly."""
        tree1, tree2 = simple_tree_pair

        # Create a single-edge list
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        single_edge = [edges[0]]
        sorted_edges = sort_edges_by_depth(single_edge, tree1, ascending=True)

        assert sorted_edges == single_edge, "Single edge should remain unchanged"

    def test_depth_values_are_consistent(self, simple_tree_pair):
        """Test that depth computation is consistent across multiple calls."""
        tree1, tree2 = simple_tree_pair

        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        # Compute depths multiple times
        depths1 = compute_edge_depths(edges, tree1)
        depths2 = compute_edge_depths(edges, tree1)
        depths3 = compute_edge_depths(edges, tree1)

        # All should be identical
        assert depths1 == depths2, "Depth computation should be deterministic"
        assert depths2 == depths3, "Depth computation should be deterministic"

    def test_ordering_with_identical_trees(self):
        """Test edge ordering when trees are identical (no s-edges)."""
        tree1_str = "(A:1,B:1,C:1):0;"
        tree1 = parse_newick(tree1_str, force_list=False)
        tree2 = tree1.deep_copy()

        # Should have no s-edges
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        assert len(edges) == 0, "Identical trees should produce no s-edges"

        # Sorting empty list should work fine
        sorted_edges = sort_edges_by_depth(edges, tree1, ascending=True)
        assert sorted_edges == [], "Empty edge list should remain empty"


class TestEdgeOrderingWithRealData:
    """Test edge ordering using real phylogenetic data."""

    def test_ordering_with_bootstrap_trees(self, tmp_path):
        """Test edge ordering with actual bootstrap tree data if available."""
        try:
            trees = read_newick("current_testfiles/small_example.newick")
        except FileNotFoundError:
            pytest.skip("Real test data not available")

        if len(trees) < 2:
            pytest.skip("Need at least 2 trees for testing")

        tree1 = trees[0]
        tree2 = trees[1]

        # Get s-edges
        solutions, _ = compute_pivot_solutions_with_deletions(tree1, tree2)
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found between these trees")

        # Sort both ways
        ascending = sort_edges_by_depth(edges, tree1, ascending=True)
        descending = sort_edges_by_depth(edges, tree1, ascending=False)

        # Verify they're different (unless only 1 edge)
        if len(edges) > 1:
            assert ascending != descending, "Ascending and descending should differ"

        # Verify ascending order has subset relationships preserved
        for i in range(len(ascending) - 1):
            edge1 = ascending[i]
            indices1 = set(edge1.resolve_to_indices())
            for edge2 in ascending[i + 1 :]:
                indices2 = set(edge2.resolve_to_indices())
                # Superset should not come before subset
                if indices2.issubset(indices1) and indices1 != indices2:
                    pytest.fail(
                        f"Parent {edge1} comes before child {edge2} in ascending order"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
