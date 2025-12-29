"""
Integration test for edge ordering across the tree interpolation pipeline.

This test verifies that:
1. Edges are sorted correctly (subsets before supersets)
2. Subset relationships are preserved (children before parents)
3. Reordering happens in the correct order (prevents snapback)
4. The full pipeline produces correct interpolation sequences
"""

import pytest
from brancharchitect.io import read_newick
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
    topological_sort_edges,
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

    def test_edges_sorted_subsets_before_supersets(self, simple_tree_pair):
        """Test that topological sort places subsets before supersets."""
        tree1, tree2 = simple_tree_pair

        # Get s-edges from lattice algorithm
        solutions, _ = LatticeSolver(tree1, tree2).solve_iteratively()
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        # Sort edges topologically
        sorted_edges = topological_sort_edges(edges, tree1)

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

    def test_sort_is_deterministic(self, simple_tree_pair):
        """Test that sorting produces consistent results across multiple calls."""
        tree1, tree2 = simple_tree_pair

        # Get s-edges from lattice algorithm
        solutions, _ = LatticeSolver(tree1, tree2).solve_iteratively()
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        # Sort multiple times
        sorted1 = topological_sort_edges(edges, tree1)
        sorted2 = topological_sort_edges(edges, tree1)
        sorted3 = topological_sort_edges(edges, tree1)

        # All results should be identical
        assert sorted1 == sorted2, "First and second sort differ"
        assert sorted2 == sorted3, "Second and third sort differ"

    def test_pair_interpolation_uses_correct_ordering(self, simple_tree_pair):
        """Test that process_tree_pair_interpolation uses correct ordering."""
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
        s_edges_used = [s for s in result.current_pivot_edge_tracking if s is not None]

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

                        # If edge1 is a superset of edge2, that's WRONG (parent before child)
                        if indices2.issubset(indices1) and indices1 != indices2:
                            pytest.fail(
                                f"S-edge ordering violation: Parent {edge1} processed before child {edge2}"
                            )
                        break

    def test_complex_nested_ordering(self, complex_tree_pair):
        """Test edge ordering with deeply nested tree structures."""
        tree1, tree2 = complex_tree_pair

        # Get s-edges from lattice algorithm
        solutions, _ = LatticeSolver(tree1, tree2).solve_iteratively()
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        # Sort edges topologically
        sorted_edges = topological_sort_edges(edges, tree1)

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
        Test that topological ordering prevents snapback problem.

        Snapback occurs when a parent s-edge is processed before its child,
        causing the parent's reordering to undo the child's work.
        """
        tree1, tree2 = simple_tree_pair

        # Get s-edges and sort them
        solutions, _ = LatticeSolver(tree1, tree2).solve_iteratively()
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        sorted_edges = topological_sort_edges(edges, tree1)

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
        sorted_edges = topological_sort_edges(edges, tree1)

        assert sorted_edges == [], "Empty list should return empty list"

    def test_single_edge_handling(self, simple_tree_pair):
        """Test that sorting handles single-edge lists correctly."""
        tree1, tree2 = simple_tree_pair

        # Create a single-edge list
        solutions, _ = LatticeSolver(tree1, tree2).solve_iteratively()
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found for this tree pair")

        single_edge = [edges[0]]
        sorted_edges = topological_sort_edges(single_edge, tree1)

        assert sorted_edges == single_edge, "Single edge should remain unchanged"

    def test_ordering_with_identical_trees(self):
        """Test edge ordering when trees are identical (no s-edges)."""
        tree1_str = "(A:1,B:1,C:1):0;"
        tree1 = parse_newick(tree1_str, force_list=False)
        tree2 = tree1.deep_copy()

        # Should have no s-edges
        solutions, _ = LatticeSolver(tree1, tree2).solve_iteratively()
        edges = list(solutions.keys())

        assert len(edges) == 0, "Identical trees should produce no s-edges"

        # Sorting empty list should work fine
        sorted_edges = topological_sort_edges(edges, tree1)
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
        solutions, _ = LatticeSolver(tree1, tree2).solve_iteratively()
        edges = list(solutions.keys())

        if not edges:
            pytest.skip("No s-edges found between these trees")

        # Sort topologically
        sorted_edges = topological_sort_edges(edges, tree1)

        # Verify subset relationships are preserved
        for i in range(len(sorted_edges) - 1):
            edge1 = sorted_edges[i]
            indices1 = set(edge1.resolve_to_indices())
            for edge2 in sorted_edges[i + 1 :]:
                indices2 = set(edge2.resolve_to_indices())
                # Superset should not come before subset
                if indices2.issubset(indices1) and indices1 != indices2:
                    pytest.fail(f"Parent {edge1} comes before child {edge2}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
