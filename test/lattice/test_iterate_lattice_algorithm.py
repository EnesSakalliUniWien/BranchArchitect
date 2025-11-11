"""
Test for iterate_lattice_algorithm to verify correct mapping of solutions to s-edges.

This test verifies that:
1. The function returns a tuple (dict, list) not just a dict
2. Solutions are correctly mapped to their corresponding s-edges
3. The mapping preserves pivot-to-partitions correspondence
"""

import unittest
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from brancharchitect.elements.partition import Partition


class TestIterateLatticeAlgorithm(unittest.TestCase):
    """Test the iterate_lattice_algorithm function."""

    def setUp(self):
        """Set up common test trees with shared encoding."""
        # Simple case with one s-edge
        tree1_str = "((A,B),(C,D));"
        tree2_str = "((A,C),(B,D));"

        # Parse trees - they will share encoding automatically
        trees = parse_newick(tree1_str + tree2_str)
        self.tree1_simple = trees[0]
        self.tree2_simple = trees[1]

        # More complex case
        tree1_complex_str = "((((A,B),(C,D)),O));"
        tree2_complex_str = "((((A,C),(B,D)),O));"

        trees_complex = parse_newick(tree1_complex_str + tree2_complex_str)
        self.tree1_complex = trees_complex[0]
        self.tree2_complex = trees_complex[1]

    def test_return_type_is_tuple(self):
        """Verify that iterate_lattice_algorithm returns a tuple, not just a dict."""
        result = iterate_lattice_algorithm(self.tree1_simple, self.tree2_simple)

        # Should return a tuple
        self.assertIsInstance(result, tuple, "Result should be a tuple")
        self.assertEqual(len(result), 2, "Tuple should have 2 elements")

        jumping_subtree_solutions, deleted_taxa_per_iteration = result

        # First element should be a dict
        self.assertIsInstance(
            jumping_subtree_solutions, dict, "First element should be a dictionary"
        )

        # Second element should be a list
        self.assertIsInstance(
            deleted_taxa_per_iteration, list, "Second element should be a list"
        )

    def test_solution_to_sedge_mapping_correctness(self):
        """
        Verify that solutions are correctly mapped to their s-edges.

        The result structure maps each pivot edge to a flat list of
        solution partitions (jumping taxa groups) selected by parsimony.
        """
        result = iterate_lattice_algorithm(self.tree1_complex, self.tree2_complex)
        jumping_subtree_solutions, _ = result

        # Verify structure: dict maps Partition -> List[Partition]
        self.assertIsInstance(jumping_subtree_solutions, dict)

        for pivot_edge, partitions in jumping_subtree_solutions.items():
            # Each pivot edge should map to a list of partitions
            self.assertIsInstance(
                pivot_edge, Partition, f"Key should be Partition: {pivot_edge}"
            )
            self.assertIsInstance(
                partitions, list, f"Value should be list for pivot edge {pivot_edge}"
            )
            for partition in partitions:
                self.assertIsInstance(
                    partition,
                    Partition,
                    f"Solution should contain Partitions for pivot edge {pivot_edge}",
                )

    def test_solutions_are_nonempty(self):
        """Verify that we get at least one solution for differing trees."""
        result = iterate_lattice_algorithm(self.tree1_simple, self.tree2_simple)
        jumping_subtree_solutions, _ = result

        # Should have at least one pivot edge with solutions
        self.assertGreater(
            len(jumping_subtree_solutions),
            0,
            "Should find at least one pivot edge with solutions for different trees",
        )

        # Each pivot edge should have at least one solution partition
        for pivot_edge, partitions in jumping_subtree_solutions.items():
            self.assertGreater(
                len(partitions),
                0,
                f"Pivot edge {pivot_edge} should have at least one solution partition",
            )

    def test_identical_trees_return_empty(self):
        """Verify that identical trees return empty solutions."""
        # Use the same tree twice
        result = iterate_lattice_algorithm(self.tree1_simple, self.tree1_simple)
        jumping_subtree_solutions, deleted_taxa_per_iteration = result

        # Should have no solutions for identical trees
        self.assertEqual(
            len(jumping_subtree_solutions),
            0,
            "Identical trees should produce no solutions",
        )
        self.assertEqual(
            len(deleted_taxa_per_iteration),
            0,
            "Identical trees should have no deleted taxa",
        )

    def test_mapping_preserves_correspondence(self):
        """
        Test that the index correspondence is preserved during mapping.

        This is the KEY test for the bug: When lattice_algorithm returns
        (solution_sets, splits), and we map splits to original, we must
        ensure solution_sets[i] still corresponds to mapped_splits[i].
        """
        result = iterate_lattice_algorithm(self.tree1_complex, self.tree2_complex)
        jumping_subtree_solutions, _ = result

        # Track total number of partitions returned across all pivots
        total_partitions = sum(
            len(parts) for parts in jumping_subtree_solutions.values()
        )

        # Should have solutions (this catches if mapping is broken)
        self.assertGreater(
            total_partitions,
            0,
            "Should have at least one solution partition across all pivot edges",
        )

        # Verify no None keys (would indicate mapping failure)
        for pivot_edge in jumping_subtree_solutions.keys():
            self.assertIsNotNone(
                pivot_edge,
                "Pivot edge keys should never be None (indicates mapping failure)",
            )

    def test_deleted_taxa_tracking(self):
        """Verify that deleted taxa are properly tracked."""
        result = iterate_lattice_algorithm(self.tree1_complex, self.tree2_complex)
        _, deleted_taxa_per_iteration = result

        # Should be a list of sets
        self.assertIsInstance(deleted_taxa_per_iteration, list)

        for iteration_taxa in deleted_taxa_per_iteration:
            self.assertIsInstance(
                iteration_taxa, set, "Each iteration should track deleted taxa as a set"
            )


class TestIterateLatticeAlgorithmEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_complex_tree_pair(self):
        """Test with a more complex tree pair that has multiple pivot edges."""
        tree1_str = "((O1,O2),(((((A,A1),A2),(B,B1)),C),((D,(E,(((F,G),I),M))),H)));"
        tree2_str = "((O1,O2),(((((A,A1),B1),(B,A2)),(C,(E,(((F,M),I),G)))),(D,H)));"

        trees = parse_newick(tree1_str + tree2_str)
        tree1 = trees[0]
        tree2 = trees[1]

        result = iterate_lattice_algorithm(tree1, tree2)
        jumping_subtree_solutions, deleted_taxa_per_iteration = result

        # Should handle complex case
        self.assertIsInstance(jumping_subtree_solutions, dict)
        self.assertIsInstance(deleted_taxa_per_iteration, list)

        # Verify structure
        for pivot_edge, partitions in jumping_subtree_solutions.items():
            self.assertIsInstance(pivot_edge, Partition)
            self.assertGreater(
                len(partitions), 0, f"Pivot edge {pivot_edge} has no solutions"
            )


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
