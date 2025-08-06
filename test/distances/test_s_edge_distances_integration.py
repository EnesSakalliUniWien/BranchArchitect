#!/usr/bin/env python3
"""
Integration tests for s-edge distances in the tree interpolation pipeline.

This module tests that s-edge distances are correctly calculated and propagated
through the entire tree interpolation pipeline.
"""

import pytest
from brancharchitect.parser import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import TreeInterpolationPipeline
from brancharchitect.movie_pipeline.types import PipelineConfig


class TestSEdgeDistancesIntegration:
    """Integration tests for s-edge distances in the pipeline."""

    def test_pipeline_includes_s_edge_distances(self):
        """Test that the pipeline correctly includes s-edge distances in results."""
        # Create test trees
        trees = [
            parse_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;"),
            parse_newick("(A:0.2,(B:0.1,(C:0.15,D:0.2):0.1):0.2)E:0.0;"),
            parse_newick("(((A:0.05,B:0.1):0.1,C:0.2):0.15,D:0.25)E:0.0;")
        ]
        
        # Process through pipeline
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(
                enable_rooting=True,
                optimization_iterations=0  # Disable optimization for predictable results
            )
        )
        
        result = pipeline.process_trees(trees)
        
        # Check that we have tree pair solutions
        assert len(result['tree_pair_solutions']) == 2  # 2 pairs for 3 trees
        
        # Check each tree pair solution
        for pair_key, solution in result['tree_pair_solutions'].items():
            # Verify s_edge_distances field exists
            assert 's_edge_distances' in solution
            assert isinstance(solution['s_edge_distances'], dict)
            
            # If there are lattice solutions, we should have distances
            if solution['lattice_edge_solutions']:
                # For each s-edge in lattice solutions
                for s_edge in solution['lattice_edge_solutions']:
                    # Should have corresponding distance data
                    if s_edge in solution['s_edge_distances']:
                        distances = solution['s_edge_distances'][s_edge]
                        
                        # Verify all required fields
                        self._verify_distance_fields(distances)
                        
                        # Verify distance relationships
                        self._verify_distance_relationships(distances)

    def test_distance_metrics_consistency(self):
        """Test that distance metrics are consistent and meaningful."""
        # Create trees with known structure for predictable distances
        tree1 = parse_newick("((A:0.1,B:0.1):0.5,(C:0.1,D:0.1):0.5)E:0.0;")
        tree2 = parse_newick("((A:0.2,B:0.2):1.0,(C:0.2,D:0.2):1.0)E:0.0;")
        
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(enable_rooting=False, optimization_iterations=0)
        )
        
        result = pipeline.process_trees([tree1, tree2])
        
        # Get first (and only) tree pair solution
        pair_solution = result['tree_pair_solutions']['pair_0_1']
        
        if pair_solution['s_edge_distances']:
            for s_edge, distances in pair_solution['s_edge_distances'].items():
                # Topological distances should be integers
                assert distances['target_topological'] == int(distances['target_topological'])
                assert distances['reference_topological'] == int(distances['reference_topological'])
                
                # Weighted distances should be >= 0
                assert distances['target_weighted'] >= 0
                assert distances['reference_weighted'] >= 0
                
                # Total should be sum of parts
                assert abs(distances['total_topological'] - 
                          (distances['target_topological'] + distances['reference_topological'])) < 0.001
                assert abs(distances['total_weighted'] - 
                          (distances['target_weighted'] + distances['reference_weighted'])) < 0.001

    def test_s_edge_distances_with_different_topologies(self):
        """Test s-edge distances when trees have different topologies."""
        # Different topologies should result in different distance patterns
        tree1 = parse_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;")
        tree2 = parse_newick("(A:0.2,(B:0.1,(C:0.15,D:0.2):0.1):0.2)E:0.0;")
        
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(enable_rooting=False, optimization_iterations=0)
        )
        
        result = pipeline.process_trees([tree1, tree2])
        
        pair_solution = result['tree_pair_solutions']['pair_0_1']
        
        # Should have s-edge distances since topologies differ
        assert pair_solution['s_edge_distances']
        
        # Check that we have meaningful distances
        total_components = 0
        for s_edge, distances in pair_solution['s_edge_distances'].items():
            total_components += distances['component_count']
            
            # With different topologies, distances often differ between trees
            # (though not always - depends on specific s-edge and component)
            print(f"S-edge {s_edge}:")
            print(f"  Target topo: {distances['target_topological']}, "
                  f"Reference topo: {distances['reference_topological']}")
            print(f"  Target weighted: {distances['target_weighted']:.3f}, "
                  f"Reference weighted: {distances['reference_weighted']:.3f}")
        
        # Should have found jumping taxa
        assert total_components > 0

    def test_empty_s_edge_distances(self):
        """Test handling when no s-edges are found (identical or very similar trees)."""
        # Very similar trees might not produce s-edges
        tree1 = parse_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1)E:0.0;")
        tree2 = parse_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1)E:0.0;")  # Identical
        
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(enable_rooting=False, optimization_iterations=0)
        )
        
        result = pipeline.process_trees([tree1, tree2])
        
        pair_solution = result['tree_pair_solutions']['pair_0_1']
        
        # Should still have s_edge_distances field, even if empty
        assert 's_edge_distances' in pair_solution
        assert isinstance(pair_solution['s_edge_distances'], dict)
        
        # With identical trees, might have no s-edges
        if not pair_solution['lattice_edge_solutions']:
            assert len(pair_solution['s_edge_distances']) == 0

    def test_distance_calculation_performance(self):
        """Test that distance calculations don't significantly impact performance."""
        import time
        
        # Create a moderately complex tree set
        trees = [
            parse_newick("((((A:0.1,B:0.1):0.1,C:0.1):0.1,D:0.1):0.1,E:0.1)F:0.0;"),
            parse_newick("(((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1):0.1,E:0.1)F:0.0;"),
            parse_newick("((A:0.1,(B:0.1,C:0.1):0.1):0.1,(D:0.1,E:0.1):0.1)F:0.0;")
        ]
        
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(enable_rooting=False, optimization_iterations=0)
        )
        
        # Time the processing
        start_time = time.time()
        result = pipeline.process_trees(trees)
        elapsed_time = time.time() - start_time
        
        # Should complete reasonably quickly (< 1 second for small trees)
        assert elapsed_time < 1.0, f"Processing took too long: {elapsed_time:.3f}s"
        
        # Verify distances were calculated
        for pair_solution in result['tree_pair_solutions'].values():
            assert 's_edge_distances' in pair_solution

    def test_weighted_vs_topological_insights(self):
        """Test that weighted and topological distances provide different insights."""
        # Create trees with same topology but very different branch lengths
        tree1 = parse_newick("((A:0.01,B:0.01):0.01,(C:0.01,D:0.01):0.01)E:0.0;")
        tree2 = parse_newick("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0)E:0.0;")
        
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(enable_rooting=False, optimization_iterations=0)
        )
        
        result = pipeline.process_trees([tree1, tree2])
        
        pair_solution = result['tree_pair_solutions']['pair_0_1']
        
        # With same topology but different branch lengths, 
        # topological distances should be similar but weighted should differ
        if pair_solution['s_edge_distances']:
            for s_edge, distances in pair_solution['s_edge_distances'].items():
                # Topological distances should be the same between trees
                assert abs(distances['target_topological'] - 
                          distances['reference_topological']) < 0.1
                
                # But weighted distances should differ significantly
                # (tree2 has 100x longer branches)
                if distances['target_weighted'] > 0 or distances['reference_weighted'] > 0:
                    ratio = max(distances['target_weighted'], distances['reference_weighted']) / \
                           max(min(distances['target_weighted'], distances['reference_weighted']), 0.001)
                    assert ratio > 10, "Expected significant difference in weighted distances"

    def _verify_distance_fields(self, distances):
        """Verify all required distance fields are present."""
        required_fields = [
            'target_topological', 'target_weighted',
            'reference_topological', 'reference_weighted',
            'total_topological', 'total_weighted',
            'component_count'
        ]
        
        for field in required_fields:
            assert field in distances, f"Missing required field: {field}"
            assert isinstance(distances[field], (int, float)), \
                f"Field {field} should be numeric, got {type(distances[field])}"

    def _verify_distance_relationships(self, distances):
        """Verify mathematical relationships between distance components."""
        # Totals should be sum of components
        assert abs(distances['total_topological'] - 
                  (distances['target_topological'] + distances['reference_topological'])) < 0.001
        
        assert abs(distances['total_weighted'] - 
                  (distances['target_weighted'] + distances['reference_weighted'])) < 0.001
        
        # All distances should be non-negative
        for key, value in distances.items():
            if 'distance' in key or 'topological' in key or 'weighted' in key:
                assert value >= 0, f"{key} should be non-negative, got {value}"
        
        # Component count should be positive integer if there are components
        if distances['component_count'] > 0:
            assert distances['component_count'] == int(distances['component_count'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])