"""
Tests for identify_and_delete_jumping_taxa function.

This function is responsible for:
1. Extracting taxa indices from solution partitions
2. Deleting those taxa from both trees
3. Determining if the iteration loop should break
"""

import pytest
from brancharchitect.jumping_taxa.lattice.orchestration.delete_taxa import (
    identify_and_delete_jumping_taxa,
)
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition


class TestIdentifyAndDeleteJumpingTaxa:
    """Test the identify_and_delete_jumping_taxa helper function."""

    def test_delete_single_solution_set(self):
        """Test deletion with a single solution set containing one partition."""
        # Create simple trees: ((A,B),(C,D))
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"  # Different topology

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Create a solution set with partition containing taxa A and B (indices 0, 1)
        partition = Partition(
            frozenset({0, 1}),  # indices for A, B
            tree1.taxa_encoding,
        )
        partitions = [partition]

        # Initial leaf count
        initial_leaves_t1 = len(tree1.get_leaves())
        initial_leaves_t2 = len(tree2.get_leaves())

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Verify results
        assert should_break is False, "Should not break with enough remaining leaves"
        assert deleted_indices == {0, 1}, "Should delete indices 0 and 1 (A and B)"

        # Verify trees have fewer leaves
        assert len(tree1.get_leaves()) == initial_leaves_t1 - 2
        assert len(tree2.get_leaves()) == initial_leaves_t2 - 2

    def test_delete_multiple_partitions_in_solution(self):
        """Test deletion with multiple partitions in a single solution set."""
        # Create trees with 6 taxa
        tree1_newick = "(((A,B),(C,D)),(E,F));"
        tree2_newick = "(((A,C),(B,D)),(E,F));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Create solution set with multiple partitions
        partition1 = Partition(frozenset({0, 1}), tree1.taxa_encoding)  # A, B
        partition2 = Partition(frozenset({2, 3}), tree1.taxa_encoding)  # C, D
        partitions = [partition1, partition2]

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Verify results
        assert should_break is False
        assert deleted_indices == {0, 1, 2, 3}, (
            "Should delete all indices from both partitions"
        )

        # Trees should now have only 2 leaves left (E and F)
        assert len(tree1.get_leaves()) == 2
        assert len(tree2.get_leaves()) == 2

    def test_delete_multiple_solution_sets(self):
        """Test deletion with multiple solution sets."""
        tree1_newick = "(((A,B),(C,D)),(E,F));"
        tree2_newick = "(((A,C),(B,D)),(E,F));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Create multiple solution sets
        partition1 = Partition(frozenset({0, 1}), tree1.taxa_encoding)  # A, B
        partition2 = Partition(frozenset({4}), tree1.taxa_encoding)  # E
        partitions = [partition1, partition2]

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Verify results
        assert should_break is False
        assert deleted_indices == {0, 1, 4}, (
            "Should delete indices from all solution sets"
        )

        # Trees should now have 3 leaves (C, D, F)
        assert len(tree1.get_leaves()) == 3
        assert len(tree2.get_leaves()) == 3

    def test_empty_solution_sets(self):
        """Test with empty solution sets - should break loop."""
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Empty solution sets
        partitions = []

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Should break because no taxa to delete
        assert should_break is True, "Should break with empty solution sets"
        assert deleted_indices == set(), "Should have empty deleted indices"

        # Trees should be unchanged
        assert len(tree1.get_leaves()) == 4
        assert len(tree2.get_leaves()) == 4

    def test_break_when_tree_too_small(self):
        """Test that loop breaks when tree has < 2 leaves remaining."""
        # Start with 3 taxa
        tree1_newick = "((A,B),C);"
        tree2_newick = "((A,C),B);"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Delete 2 taxa, leaving only 1
        partition = Partition(frozenset({0, 1}), tree1.taxa_encoding)  # A, B
        partitions = [partition]

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Should break because tree has only 1 leaf left
        assert should_break is True, "Should break when tree has < 2 leaves"
        assert deleted_indices == {0, 1}
        assert len(tree1.get_leaves()) == 1
        assert len(tree2.get_leaves()) == 1

    def test_bird_phylogeny_first_iteration(self):
        """Test with real data from actual_state_26_27.json (bird phylogeny)."""
        # Load the bird phylogeny trees
        import json
        from pathlib import Path

        test_file = Path(
            "test/colouring/trees/failing_tree_pair_26_27/actual_state_26_27.json"
        )
        if not test_file.exists():
            pytest.skip("Test data file not found")

        with open(test_file) as f:
            data = json.load(f)

        tree1 = parse_newick(data["tree1"])
        tree2 = parse_newick(
            data["tree2"],
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Initial leaf count should be 24
        assert len(tree1.get_leaves()) == 24
        assert len(tree2.get_leaves()) == 24

        # Create solution sets based on known jumping taxa from this test
        # First iteration should find kiwis as jumping taxa
        kiwi_indices = {
            tree1.taxa_encoding["BrownKiwi"],
            tree1.taxa_encoding["gskiwi"],
            tree1.taxa_encoding["LSKiwi"],
        }

        # Create partition for kiwis
        kiwi_partition = Partition(frozenset(kiwi_indices), tree1.taxa_encoding)
        partitions = [kiwi_partition]

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Verify results
        assert should_break is False, "Should continue - enough leaves remain"
        assert deleted_indices == kiwi_indices

        # Should have 21 leaves remaining (24 - 3)
        assert len(tree1.get_leaves()) == 21
        assert len(tree2.get_leaves()) == 21

    def test_overlapping_partitions(self):
        """Test with overlapping partitions (same taxon in multiple partitions)."""
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Create overlapping partitions (A appears in both)
        partition1 = Partition(frozenset({0, 1}), tree1.taxa_encoding)  # A, B
        partition2 = Partition(frozenset({0, 2}), tree1.taxa_encoding)  # A, C
        partitions = [partition1, partition2]

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Verify: should delete A, B, C (indices 0, 1, 2) - union of both partitions
        # Note: After deleting 3 taxa from a 4-taxa tree, only 1 leaf remains
        # The function breaks when < 2 leaves remain, so this should break
        assert should_break is True, "Should break with only 1 leaf remaining"
        assert deleted_indices == {0, 1, 2}, (
            "Should delete union of overlapping indices"
        )

        # Should have 1 leaf remaining (D)
        assert len(tree1.get_leaves()) == 1
        assert len(tree2.get_leaves()) == 1

    def test_preserves_tree_structure(self):
        """Test that deletion preserves valid tree structure."""
        tree1_newick = "(((A,B),(C,D)),((E,F),(G,H)));"
        tree2_newick = "(((A,C),(B,D)),((E,G),(F,H)));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Delete A and B
        partition = Partition(frozenset({0, 1}), tree1.taxa_encoding)
        partitions = [partition]

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Verify trees are still valid
        assert should_break is False
        assert len(tree1.get_leaves()) == 6
        assert len(tree2.get_leaves()) == 6

        # Trees should still be parseable (valid structure)
        newick1 = tree1.to_newick(lengths=False)
        newick2 = tree2.to_newick(lengths=False)
        assert newick1 is not None
        assert newick2 is not None

        # Should not contain deleted taxa names
        remaining_leaves_t1 = [leaf.name for leaf in tree1.get_leaves()]
        remaining_leaves_t2 = [leaf.name for leaf in tree2.get_leaves()]
        assert "A" not in remaining_leaves_t1
        assert "B" not in remaining_leaves_t1
        assert "A" not in remaining_leaves_t2
        assert "B" not in remaining_leaves_t2

    def test_iteration_count_logging(self):
        """Test that iteration count is properly used in logging."""
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        partition = Partition(frozenset({0}), tree1.taxa_encoding)
        partitions = [partition]

        # Test with different iteration counts
        for iteration in [1, 5, 10]:
            tree1_copy = tree1.deep_copy()
            tree2_copy = tree2.deep_copy()

            should_break, deleted_indices = identify_and_delete_jumping_taxa(
                partitions, tree1_copy, tree2_copy, iteration_count=iteration
            )

            # Function should work regardless of iteration count
            assert deleted_indices == {0}
            assert len(tree1_copy.get_leaves()) == 3
            assert len(tree2_copy.get_leaves()) == 3

    def test_single_taxon_partition(self):
        """Test deletion of single taxon partition."""
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Single taxon partition (just A)
        partition = Partition(frozenset({0}), tree1.taxa_encoding)
        partitions = [partition]

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Verify
        assert should_break is False
        assert deleted_indices == {0}
        assert len(tree1.get_leaves()) == 3
        assert len(tree2.get_leaves()) == 3

    def test_trees_modified_in_place(self):
        """Test that trees are modified in-place (not copied)."""
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Keep references to original tree objects
        tree1_id = id(tree1)
        tree2_id = id(tree2)

        partition = Partition(frozenset({0, 1}), tree1.taxa_encoding)
        partitions = [partition]

        # Call the function
        identify_and_delete_jumping_taxa(partitions, tree1, tree2, iteration_count=1)

        # Verify same objects (not copies)
        assert id(tree1) == tree1_id, "Tree1 should be modified in-place"
        assert id(tree2) == tree2_id, "Tree2 should be modified in-place"

        # And verify they were actually modified
        assert len(tree1.get_leaves()) == 2
        assert len(tree2.get_leaves()) == 2


class TestIdentifyAndDeleteEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_solution_with_empty_partition_list(self):
        """Test solution set containing empty partition list."""
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Solution set with empty partition list
        partitions = []

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Should break because no taxa to delete
        assert should_break is True
        assert deleted_indices == set()

    def test_delete_all_but_two_leaves(self):
        """Test deleting until exactly 2 leaves remain (boundary case)."""
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        # Delete 2 taxa, leaving exactly 2
        partition = Partition(frozenset({0, 1}), tree1.taxa_encoding)
        partitions = [partition]

        # Call the function
        should_break, deleted_indices = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Should NOT break - 2 leaves is still valid
        assert should_break is False, "Should continue with exactly 2 leaves"
        assert len(tree1.get_leaves()) == 2
        assert len(tree2.get_leaves()) == 2

    def test_return_type_consistency(self):
        """Test that return types are always consistent."""
        tree1_newick = "((A,B),(C,D));"
        tree2_newick = "((A,C),(B,D));"

        tree1 = parse_newick(tree1_newick)
        tree2 = parse_newick(
            tree2_newick,
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )

        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        partition = Partition(frozenset({0}), tree1.taxa_encoding)
        partitions = [partition]

        result = identify_and_delete_jumping_taxa(
            partitions, tree1, tree2, iteration_count=1
        )

        # Verify return type
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-tuple"

        should_break, deleted_indices = result
        assert isinstance(should_break, bool), "First element should be bool"
        assert isinstance(deleted_indices, set), "Second element should be set"
