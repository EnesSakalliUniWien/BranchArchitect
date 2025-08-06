#!/usr/bin/env python3
"""
Exact value tests for s-edge distances to verify the specific example from the summary.

This test verifies the exact distance values for the (C, D) s-edge example:
- Topological: Target 1.0, Reference 1.0, Total 2.0
- Weighted: Target 0.000, Reference 0.100, Total 0.100
- Components: 2 jumping taxa
"""

import pytest
from brancharchitect.parser import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import TreeInterpolationPipeline
from brancharchitect.movie_pipeline.types import PipelineConfig


class TestSEdgeDistancesExactValues:
    """Test exact distance values from the implementation summary."""

    def test_exact_cd_split_distances(self):
        """Test the exact distance values for (C, D) split from the summary."""
        # Use the exact trees from the test output
        trees = [
            parse_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;"),
            parse_newick("(A:0.2,(B:0.1,(C:0.15,D:0.2):0.1):0.2)E:0.0;")
        ]
        
        # Process with pipeline (optimization disabled for predictable results)
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(
                enable_rooting=True,
                optimization_iterations=0
            )
        )
        
        result = pipeline.process_trees(trees)
        
        # Get the pair_0_1 solution
        pair_solution = result['tree_pair_solutions']['pair_0_1']
        
        # Verify we have s-edge distances
        assert pair_solution['s_edge_distances']
        
        # Find the (C, D) s-edge - it should be represented as indices (2, 3)
        cd_distances = None
        for s_edge, distances in pair_solution['s_edge_distances'].items():
            # The s-edge should represent C and D taxa
            # Check if this is the (C, D) split
            if len(s_edge.indices) == 2 and 2 in s_edge.indices and 3 in s_edge.indices:
                cd_distances = distances
                break
        
        assert cd_distances is not None, "Could not find (C, D) s-edge distances"
        
        # Verify exact values from the summary
        self._assert_almost_equal(cd_distances['target_topological'], 1.0, 
                                 "Target topological distance should be 1.0 edges")
        self._assert_almost_equal(cd_distances['reference_topological'], 1.0,
                                 "Reference topological distance should be 1.0 edges")
        self._assert_almost_equal(cd_distances['total_topological'], 2.0,
                                 "Total topological distance should be 2.0 edges")
        
        # Note: The exact weighted values might vary slightly due to how paths are calculated
        # but should be close to the expected values
        self._assert_almost_equal(cd_distances['target_weighted'], 0.0, 
                                 "Target weighted distance should be ~0.000", tolerance=0.01)
        self._assert_almost_equal(cd_distances['reference_weighted'], 0.1,
                                 "Reference weighted distance should be ~0.100", tolerance=0.01)
        self._assert_almost_equal(cd_distances['total_weighted'], 0.1,
                                 "Total weighted distance should be ~0.100", tolerance=0.01)
        
        # Verify component count
        assert cd_distances['component_count'] == 2.0, "Should have 2 jumping taxa (C and D)"

    def test_distance_metrics_properties(self):
        """Test the properties of the improved distance metrics."""
        trees = [
            parse_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25)E:0.0;"),
            parse_newick("(A:0.2,(B:0.1,(C:0.15,D:0.2):0.1):0.2)E:0.0;")
        ]
        
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(enable_rooting=True, optimization_iterations=0)
        )
        
        result = pipeline.process_trees(trees)
        pair_solution = result['tree_pair_solutions']['pair_0_1']
        
        # Test all 7 metrics are present for each s-edge
        for s_edge, distances in pair_solution['s_edge_distances'].items():
            # Verify all 7 metrics exist
            assert len(distances) == 7, f"Should have exactly 7 metrics, got {len(distances)}"
            
            # 1. Topological distances (unweighted) - should be integers
            assert distances['target_topological'] == int(distances['target_topological'])
            assert distances['reference_topological'] == int(distances['reference_topological'])
            
            # 2. Weighted distances - can be floats
            assert isinstance(distances['target_weighted'], float)
            assert isinstance(distances['reference_weighted'], float)
            
            # 3. Both trees calculated
            assert 'target_topological' in distances and 'reference_topological' in distances
            assert 'target_weighted' in distances and 'reference_weighted' in distances
            
            # 4. Comprehensive totals
            assert distances['total_topological'] == (
                distances['target_topological'] + distances['reference_topological']
            )
            assert abs(distances['total_weighted'] - 
                      (distances['target_weighted'] + distances['reference_weighted'])) < 0.001
            
            # Component count
            assert distances['component_count'] >= 0

    def test_structural_vs_evolutionary_distance(self):
        """Test that metrics capture both structural complexity and evolutionary distance."""
        # Create trees where structural and evolutionary distances differ significantly
        # Same structure but very different branch lengths
        tree1 = parse_newick("((A:0.001,B:0.001):0.001,(C:0.001,D:0.001):0.001)E:0.0;")
        tree2 = parse_newick("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0)E:0.0;")
        
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(enable_rooting=False, optimization_iterations=0)
        )
        
        result = pipeline.process_trees([tree1, tree2])
        
        if result['tree_pair_solutions']:
            pair_solution = list(result['tree_pair_solutions'].values())[0]
            
            if pair_solution['s_edge_distances']:
                # For same topology, different branch lengths:
                # - Topological distances should be identical between trees
                # - Weighted distances should differ by ~1000x
                
                for s_edge, distances in pair_solution['s_edge_distances'].items():
                    # Structural complexity (topological) should be the same
                    assert distances['target_topological'] == distances['reference_topological'], \
                        "Same topology should have same topological distances"
                    
                    # Evolutionary distance (weighted) should differ significantly
                    if distances['target_weighted'] > 0 or distances['reference_weighted'] > 0:
                        min_val = min(distances['target_weighted'], distances['reference_weighted'])
                        max_val = max(distances['target_weighted'], distances['reference_weighted'])
                        
                        # Avoid division by zero
                        if min_val > 0:
                            ratio = max_val / min_val
                            assert ratio > 100, \
                                "Weighted distances should differ significantly for different branch lengths"

    def _assert_almost_equal(self, actual, expected, message, tolerance=0.001):
        """Assert that two floats are almost equal within tolerance."""
        assert abs(actual - expected) <= tolerance, \
            f"{message}: expected {expected}, got {actual} (diff: {abs(actual - expected)})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])