#!/usr/bin/env python3
"""
Tests for s-edge distance calculations in tree interpolation.

This module tests the _calculate_s_edge_distances function to ensure it correctly
computes both topological and weighted distances from jumping taxa to s-edges.
"""

import pytest
from typing import Dict, List
from brancharchitect.parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.interpolation import _calculate_s_edge_distances
from brancharchitect.tree import Node


class TestSEdgeDistances:
    """Test suite for s-edge distance calculations."""

    def test_basic_s_edge_distances(self):
        """Test basic distance calculation with simple trees."""
        # Create two simple trees with known structure
        # Tree 1: ((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;
        target = parse_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;")
        
        # Tree 2: (A:0.2,(B:0.1,(C:0.15,D:0.2):0.1):0.2)E:0.0;
        reference = parse_newick("(A:0.2,(B:0.1,(C:0.15,D:0.2):0.1):0.2)E:0.0;")
        
        # Mock lattice edge solutions with known s-edge and components
        # Using the actual partition indices from the trees
        s_edge = Partition([2, 3])  # (C, D) split
        component1 = Partition([2])  # C
        component2 = Partition([3])  # D
        
        lattice_edge_solutions = {
            s_edge: [[component1, component2]]
        }
        
        # Calculate distances
        distances = _calculate_s_edge_distances(target, reference, lattice_edge_solutions)
        
        # Verify we have distances for our s-edge
        assert s_edge in distances
        s_edge_dist = distances[s_edge]
        
        # Check all required fields exist
        required_fields = [
            "target_topological", "target_weighted",
            "reference_topological", "reference_weighted", 
            "total_topological", "total_weighted",
            "component_count"
        ]
        for field in required_fields:
            assert field in s_edge_dist, f"Missing field: {field}"
        
        # Verify component count
        assert s_edge_dist["component_count"] == 2.0
        
        # Verify topological distances are integers (edge counts)
        assert s_edge_dist["target_topological"] == 1.0
        assert s_edge_dist["reference_topological"] == 1.0
        assert s_edge_dist["total_topological"] == 2.0
        
        # Verify weighted distances are reasonable
        assert s_edge_dist["target_weighted"] >= 0.0
        assert s_edge_dist["reference_weighted"] >= 0.0
        assert s_edge_dist["total_weighted"] == (
            s_edge_dist["target_weighted"] + s_edge_dist["reference_weighted"]
        )

    def test_zero_distance_same_node(self):
        """Test when component and s-edge are the same node."""
        tree = parse_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;")
        
        # Component and s-edge are the same
        s_edge = Partition([0, 1])  # (A, B) split
        component = Partition([0, 1])  # Same split
        
        lattice_edge_solutions = {
            s_edge: [[component]]
        }
        
        distances = _calculate_s_edge_distances(tree, tree, lattice_edge_solutions)
        
        assert s_edge in distances
        s_edge_dist = distances[s_edge]
        
        # Distance from a node to itself should be 0
        assert s_edge_dist["target_topological"] == 0.0
        assert s_edge_dist["reference_topological"] == 0.0
        assert s_edge_dist["target_weighted"] == 0.0
        assert s_edge_dist["reference_weighted"] == 0.0

    def test_multiple_components_averaging(self):
        """Test that distances are properly averaged across multiple components."""
        tree = parse_newick("(((A:0.1,B:0.1):0.2,C:0.3):0.4,D:0.5)E:0.0;")
        
        # S-edge at deeper level
        s_edge = Partition([0, 1, 2])  # ((A,B),C) split
        # Components at different distances
        component1 = Partition([0])  # A - distance 2 edges
        component2 = Partition([1])  # B - distance 2 edges  
        component3 = Partition([2])  # C - distance 1 edge
        
        lattice_edge_solutions = {
            s_edge: [[component1, component2, component3]]
        }
        
        distances = _calculate_s_edge_distances(tree, tree, lattice_edge_solutions)
        
        s_edge_dist = distances[s_edge]
        
        # Average topological distance should be (2 + 2 + 1) / 3 = 1.67
        expected_avg_topo = (2 + 2 + 1) / 3
        assert abs(s_edge_dist["target_topological"] - expected_avg_topo) < 0.01
        assert s_edge_dist["component_count"] == 3.0

    def test_weighted_vs_topological_difference(self):
        """Test that weighted and topological distances can differ significantly."""
        # Tree with very uneven branch lengths
        tree = parse_newick("((A:0.01,B:0.01):1.0,(C:0.5,D:0.5):0.01)E:0.0;")
        
        s_edge = Partition([2, 3])  # (C, D) split
        component1 = Partition([2])  # C
        component2 = Partition([3])  # D
        
        lattice_edge_solutions = {
            s_edge: [[component1, component2]]
        }
        
        distances = _calculate_s_edge_distances(tree, tree, lattice_edge_solutions)
        
        s_edge_dist = distances[s_edge]
        
        # Both components are 1 edge away (topological)
        assert s_edge_dist["target_topological"] == 1.0
        
        # Weighted distance should reflect the actual branch lengths in the path
        # For this tree structure, the path from leaf to internal node has the internal node's branch length
        # The components (C and D) connect to the (C,D) internal node which has length 0.01
        expected_weighted = 0.01  # Branch length of the (C,D) internal node
        assert abs(s_edge_dist["target_weighted"] - expected_weighted) < 0.01
        
        # This demonstrates different insights
        print(f"Topo: {s_edge_dist['target_topological']}, Weighted: {s_edge_dist['target_weighted']}")

    def test_missing_splits_handled_gracefully(self):
        """Test that missing splits in trees are handled with 0 distances."""
        target = parse_newick("((A:0.1,B:0.2):0.15,C:0.3)D:0.0;")
        reference = parse_newick("(A:0.1,(B:0.2,C:0.3):0.15)D:0.0;")
        
        # Create a split that might not exist in both trees
        s_edge = Partition([1, 2])  # (B, C) split
        component = Partition([1])  # B
        
        lattice_edge_solutions = {
            s_edge: [[component]]
        }
        
        distances = _calculate_s_edge_distances(target, reference, lattice_edge_solutions)
        
        # Even if paths aren't found, we should get valid results
        assert s_edge in distances
        s_edge_dist = distances[s_edge]
        
        # All fields should exist even if distances are 0
        assert "target_topological" in s_edge_dist
        assert "reference_topological" in s_edge_dist
        assert s_edge_dist["component_count"] == 1.0

    def test_empty_solution_sets(self):
        """Test handling of empty solution sets."""
        tree = parse_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;")
        
        s_edge = Partition([0, 1])
        
        # Empty solution set
        lattice_edge_solutions = {
            s_edge: [[]]  # Empty component list
        }
        
        distances = _calculate_s_edge_distances(tree, tree, lattice_edge_solutions)
        
        assert s_edge in distances
        s_edge_dist = distances[s_edge]
        
        # With no components, all distances should be 0
        assert s_edge_dist["target_topological"] == 0.0
        assert s_edge_dist["reference_topological"] == 0.0
        assert s_edge_dist["target_weighted"] == 0.0
        assert s_edge_dist["reference_weighted"] == 0.0
        assert s_edge_dist["component_count"] == 0.0

    def test_multiple_solution_sets(self):
        """Test handling of multiple solution sets for same s-edge."""
        tree = parse_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;")
        
        s_edge = Partition([0, 1])  # (A, B) split
        
        # Multiple solution sets
        lattice_edge_solutions = {
            s_edge: [
                [Partition([0])],  # First set: just A
                [Partition([1])]   # Second set: just B
            ]
        }
        
        distances = _calculate_s_edge_distances(tree, tree, lattice_edge_solutions)
        
        assert s_edge in distances
        s_edge_dist = distances[s_edge]
        
        # Should process both components
        assert s_edge_dist["component_count"] == 2.0
        
        # Both A and B are 1 edge from (A,B) split
        assert s_edge_dist["target_topological"] == 1.0

    def test_branch_length_none_handling(self):
        """Test that None branch lengths are treated as 0."""
        # Manually create a tree with None branch lengths
        root = Node(name="E", length=0.0)
        internal1 = Node(name="", length=None)  # None length
        internal2 = Node(name="", length=0.25)
        leaf_a = Node(name="A", length=0.1)
        leaf_b = Node(name="B", length=None)  # None length
        leaf_c = Node(name="C", length=0.3)
        leaf_d = Node(name="D", length=0.1)
        
        root.children = [internal1, internal2]
        internal1.children = [leaf_a, leaf_b]
        internal2.children = [leaf_c, leaf_d]
        
        # Set up parent relationships and indices
        for node in [internal1, internal2]:
            node.parent = root
        for node in [leaf_a, leaf_b]:
            node.parent = internal1
        for node in [leaf_c, leaf_d]:
            node.parent = internal2
            
        # Tree is already properly initialized from parse_newick
        
        s_edge = Partition([0, 1])  # (A, B) split
        component = Partition([0])  # A
        
        lattice_edge_solutions = {
            s_edge: [[component]]
        }
        
        # Should not crash with None lengths
        distances = _calculate_s_edge_distances(root, root, lattice_edge_solutions)
        
        assert s_edge in distances
        s_edge_dist = distances[s_edge]
        
        # Should handle None as 0
        assert s_edge_dist["target_weighted"] >= 0.0

    @pytest.mark.parametrize("tree1_newick,tree2_newick,expected_diff", [
        # Same topology, different branch lengths
        ("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1)E:0.0;",
         "((A:0.5,B:0.5):0.5,(C:0.5,D:0.5):0.5)E:0.0;",
         True),  # Weighted should differ
    ])
    def test_distance_differences_between_trees(self, tree1_newick, tree2_newick, expected_diff):
        """Test that distances differ appropriately between different trees."""
        target = parse_newick(tree1_newick)
        reference = parse_newick(tree2_newick)
        
        s_edge = Partition([2, 3])  # (C, D) split
        component = Partition([2])  # C
        
        lattice_edge_solutions = {
            s_edge: [[component]]
        }
        
        distances = _calculate_s_edge_distances(target, reference, lattice_edge_solutions)
        
        s_edge_dist = distances[s_edge]
        
        if expected_diff:
            # For same topology but different branch lengths, weighted distances should differ
            weighted_diff = abs(s_edge_dist["target_weighted"] - s_edge_dist["reference_weighted"]) > 0.01
            # Print debug info
            print(f"Target weighted: {s_edge_dist['target_weighted']}, Reference weighted: {s_edge_dist['reference_weighted']}")
            assert weighted_diff, f"Expected weighted distances to differ between trees with different branch lengths"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])