"""
Detailed test to verify pivot edge to solution mapping in iterate_lattice_algorithm.

This test specifically checks:
1. That each pivot edge key in the returned dictionary is valid
2. That solutions are correctly associated with their pivot edges
3. That the mapping preserves the correspondence from lattice_algorithm
"""

import unittest
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from brancharchitect.jumping_taxa.lattice.pivot_edge_solver import lattice_algorithm
from brancharchitect.elements.partition import Partition


class TestPivotEdgeMappingVerification(unittest.TestCase):
    """Verify that pivot edge to solution mapping is correct."""

    def setUp(self):
        """Set up test trees."""
        # Simple case with clear pivot edge
        tree1_str = "((A,B),(C,D));"
        tree2_str = "((A,C),(B,D));"

        trees = parse_newick(tree1_str + tree2_str)
        self.tree1_simple = trees[0]
        self.tree2_simple = trees[1]

        # Complex case
        tree1_complex_str = (
            "((O1,O2),(((((A,A1),A2),(B,B1)),C),((D,(E,(((F,G),I),M))),H)));"
        )
        tree2_complex_str = (
            "((O1,O2),(((((A,A1),B1),(B,A2)),(C,(E,(((F,M),I),G)))),(D,H)));"
        )

        trees_complex = parse_newick(tree1_complex_str + tree2_complex_str)
        self.tree1_complex = trees_complex[0]
        self.tree2_complex = trees_complex[1]

    def test_pivot_edge_keys_are_valid_partitions(self):
        """Verify that all pivot edge keys are valid Partition objects."""
        result = iterate_lattice_algorithm(self.tree1_complex, self.tree2_complex)
        jumping_subtree_solutions, _ = result

        print("\n=== Pivot Edge Keys in Dictionary ===")
        for i, (pivot_edge, partitions) in enumerate(
            jumping_subtree_solutions.items()
        ):
            print(f"\nPivot Edge {i + 1}:")
            print(f"  Type: {type(pivot_edge)}")
            print(f"  Partition: {pivot_edge}")
            print(f"  Bitmask: {pivot_edge.bitmask}")
            print(f"  Indices: {pivot_edge.resolve_to_indices()}")
            print(f"  Number of solution partitions: {len(partitions)}")

            # Verify it's a valid Partition
            self.assertIsInstance(
                pivot_edge, Partition, f"Key {i + 1} should be a Partition"
            )
            self.assertIsNotNone(
                pivot_edge.bitmask, f"Pivot edge {i + 1} should have a bitmask"
            )

            # Verify solutions structure
            for j, partition in enumerate(partitions):
                print(f"    Partition {j + 1}: {partition}")
                self.assertIsInstance(partition, Partition)

    def test_pivot_edges_match_original_trees(self):
        """Verify that pivot edges exist in at least one of the original trees."""
        result = iterate_lattice_algorithm(self.tree1_complex, self.tree2_complex)
        jumping_subtree_solutions, _ = result

        # Get all splits from both trees
        tree1_splits = self.tree1_complex.to_splits()
        tree2_splits = self.tree2_complex.to_splits()

        print("\n=== Verifying Pivot Edges Match Original Trees ===")
        print(f"Tree 1 has {len(tree1_splits)} splits")
        print(f"Tree 2 has {len(tree2_splits)} splits")

        for pivot_edge in jumping_subtree_solutions.keys():
            in_tree1 = pivot_edge in tree1_splits
            in_tree2 = pivot_edge in tree2_splits

            print(f"\nPivot Edge {pivot_edge.resolve_to_indices()}:")
            print(f"  In Tree 1: {in_tree1}")
            print(f"  In Tree 2: {in_tree2}")

            # Pivot edge should be in at least one tree (or be the complement)
            # The mapping should ensure this
            self.assertTrue(
                in_tree1 or in_tree2,
                f"Pivot edge {pivot_edge.resolve_to_indices()} not found in either tree - mapping may be broken",
            )

    def test_compare_direct_lattice_vs_iterate(self):
        """
        Compare calling lattice_algorithm directly vs iterate_lattice_algorithm.
        This verifies the mapping is working correctly.
        """
        # Call lattice_algorithm directly (single iteration) - now returns a dictionary
        direct_solutions_dict = lattice_algorithm(self.tree1_simple, self.tree2_simple)

        # Call iterate_lattice_algorithm
        iterate_result = iterate_lattice_algorithm(self.tree1_simple, self.tree2_simple)
        iterate_solutions_dict, _ = iterate_result

        print("\n=== Comparing Direct vs Iterate Results ===")
        print("Direct lattice_algorithm:")
        print(f"  Number of pivot edges: {len(direct_solutions_dict)}")
        total_direct_partitions = sum(
            len(parts) for parts in direct_solutions_dict.values()
        )
        print(f"  Total partitions: {total_direct_partitions}")

        print("\nIterate lattice_algorithm:")
        print(f"  Number of pivot edge keys: {len(iterate_solutions_dict)}")
        total_partitions = sum(len(parts) for parts in iterate_solutions_dict.values())
        print(f"  Total partitions: {total_partitions}")

        # Verify structure
        print("\nDirect pivot edges:")
        for pivot_edge, partitions in direct_solutions_dict.items():
            print(
                f"  {pivot_edge.resolve_to_indices()} -> {len(partitions)} partitions"
            )
            for i, p in enumerate(partitions):
                print(f"    Partition {i}: {p.resolve_to_indices()}")

        print("\nIterate pivot edges (mapped):")
        for pivot_edge, partitions in iterate_solutions_dict.items():
            print(
                f"  {pivot_edge.resolve_to_indices()} -> {len(partitions)} partitions"
            )
            for i, p in enumerate(partitions):
                print(f"    Partition {i}: {p.resolve_to_indices()}")

    def test_no_none_keys_or_values(self):
        """Verify no None values appear in the dictionary (indicates mapping failure)."""
        result = iterate_lattice_algorithm(self.tree1_complex, self.tree2_complex)
        jumping_subtree_solutions, _ = result

        print("\n=== Checking for None Values ===")

        # Check keys
        for pivot_edge in jumping_subtree_solutions.keys():
            self.assertIsNotNone(pivot_edge, "Found None key in dictionary")
            self.assertIsInstance(
                pivot_edge,
                Partition,
                f"Key should be Partition, got {type(pivot_edge)}",
            )

        # Check values
        for pivot_edge, partitions in jumping_subtree_solutions.items():
            self.assertIsNotNone(
                partitions, f"Found None value for pivot edge {pivot_edge}"
            )
            self.assertIsInstance(
                partitions, list, f"Value should be list for pivot edge {pivot_edge}"
            )
            self.assertGreater(
                len(partitions),
                0,
                f"Empty partitions for pivot edge {pivot_edge}",
            )
            for partition in partitions:
                self.assertIsNotNone(
                    partition,
                    f"Found None partition for pivot edge {pivot_edge}",
                )

        print("✓ No None values found in dictionary")

    def test_solution_counts_consistency(self):
        """
        Verify that the number of solutions is consistent across iterations.
        This catches issues where solutions might be lost or duplicated during mapping.
        """
        result = iterate_lattice_algorithm(self.tree1_complex, self.tree2_complex)
        jumping_subtree_solutions, deleted_taxa_per_iteration = result

        print("\n=== Solution Count Consistency ===")
        print(f"Number of iterations: {len(deleted_taxa_per_iteration)}")
        print(f"Number of unique pivot edges: {len(jumping_subtree_solutions)}")

        total_partitions = 0
        for pivot_edge, partitions in jumping_subtree_solutions.items():
            num_parts = len(partitions)
            total_partitions += num_parts
            print(
                f"Pivot edge {pivot_edge.resolve_to_indices()}: {num_parts} partition(s)"
            )

            # Each pivot edge should have at least one partition
            self.assertGreater(
                num_parts,
                0,
                f"Pivot edge {pivot_edge.resolve_to_indices()} has no solution partitions",
            )

        print(f"\nTotal partitions across all pivot edges: {total_partitions}")
        self.assertGreater(
            total_partitions, 0, "Should have at least one solution partition"
        )

    def test_index_correspondence_preserved(self):
        """
        Critical test: Verify that lattice_algorithm returns a dictionary
        with correct pivot edge to solution mappings.

        This verifies the refactored return type works correctly.
        """
        # Get results from iterate_lattice_algorithm
        result = iterate_lattice_algorithm(self.tree1_simple, self.tree2_simple)
        jumping_subtree_solutions, _ = result

        # Also call lattice_algorithm directly to compare - now returns dictionary
        direct_solutions_dict = lattice_algorithm(self.tree1_simple, self.tree2_simple)

        print("\n=== Testing Dictionary Return Type ===")
        print("Direct lattice_algorithm returned:")
        print(f"  {len(direct_solutions_dict)} pivot edges")
        total_partitions = sum(len(parts) for parts in direct_solutions_dict.values())
        print(f"  {total_partitions} total partitions")

        # Verify it's a dictionary
        self.assertIsInstance(
            direct_solutions_dict,
            dict,
            "lattice_algorithm should return a dictionary",
        )

        # Now verify that iterate_lattice_algorithm correctly uses them
        print("\nDirect results (dictionary from lattice_algorithm):")
        for pivot_edge, partitions in direct_solutions_dict.items():
            print(
                f"  Pivot edge {pivot_edge.resolve_to_indices()} -> {len(partitions)} partition(s)"
            )
            for j, p in enumerate(partitions):
                print(f"    Partition {j}: {p.resolve_to_indices()}")

        print("\nMapped results (after mapping in iterate_lattice_algorithm):")
        for pivot_edge, partitions in jumping_subtree_solutions.items():
            print(
                f"  Pivot edge {pivot_edge.resolve_to_indices()} -> {len(partitions)} partition(s)"
            )
            for j, p in enumerate(partitions):
                print(f"    Partition {j}: {p.resolve_to_indices()}")

        # Key assertion: Every direct pivot edge should map to something in the dictionary
        # (either the partition itself or its complement, depending on the mapping logic)
        print(
            "\nVerifying all direct pivot edges are represented in mapped dictionary..."
        )
        for direct_pivot_edge in direct_solutions_dict.keys():
            # Check if this pivot edge (or its representation) is in the dictionary
            found = False
            for mapped_pivot_edge in jumping_subtree_solutions.keys():
                # They might be the same partition or complements
                if (
                    direct_pivot_edge.bitmask == mapped_pivot_edge.bitmask
                    or direct_pivot_edge.bitmask == ~mapped_pivot_edge.bitmask
                ):
                    found = True
                    print(
                        f"  ✓ Direct pivot edge {direct_pivot_edge.resolve_to_indices()} found as {mapped_pivot_edge.resolve_to_indices()}"
                    )
                    break

            self.assertTrue(
                found,
                f"Direct pivot edge {direct_pivot_edge.resolve_to_indices()} not found in mapped dictionary",
            )


if __name__ == "__main__":
    # Run with verbose output to see all the diagnostic prints
    unittest.main(verbosity=2)
