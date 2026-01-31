"""
Tests for subtree tracking in tree interpolation.

Tests cover:
- TreeInterpolationSequence field existence
- Subtree tracking propagation through SequentialInterpolationBuilder
- API response structure
"""

import sys
from unittest.mock import MagicMock

# Mock flask and flask_cors before they are imported by webapp modules
sys.modules["flask"] = MagicMock()
sys.modules["flask_cors"] = MagicMock()
sys.modules["msa_to_trees"] = MagicMock()
sys.modules["msa_to_trees.pipeline"] = MagicMock()

import unittest
from typing import Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.types import TreeInterpolationSequence


class TestTreeInterpolationSequenceSubtreeTracking(unittest.TestCase):
    """Test that TreeInterpolationSequence has the current_subtree_tracking field."""

    def test_field_exists_and_initializes_to_empty_list(self):
        """Verify current_subtree_tracking field exists and initializes to empty list."""
        seq = TreeInterpolationSequence()

        # Field should exist
        self.assertTrue(hasattr(seq, "current_subtree_tracking"))

        # Should initialize to empty list
        self.assertEqual(seq.current_subtree_tracking, [])
        self.assertIsInstance(seq.current_subtree_tracking, list)

    def test_field_can_be_set_in_constructor(self):
        """Verify current_subtree_tracking can be set via constructor."""
        encoding = {"A": 0, "B": 1}
        part_a = Partition((0,), encoding)

        tracking = [None, part_a, part_a, None]

        seq = TreeInterpolationSequence(current_subtree_tracking=tracking)

        self.assertEqual(len(seq.current_subtree_tracking), 4)
        self.assertIsNone(seq.current_subtree_tracking[0])
        self.assertEqual(seq.current_subtree_tracking[1], part_a)
        self.assertEqual(seq.current_subtree_tracking[2], part_a)
        self.assertIsNone(seq.current_subtree_tracking[3])

    def test_field_type_matches_pivot_edge_tracking(self):
        """Verify current_subtree_tracking has same type as current_pivot_edge_tracking."""
        seq = TreeInterpolationSequence()

        # Both should be list[Optional[Partition]]
        self.assertEqual(
            type(seq.current_subtree_tracking), type(seq.current_pivot_edge_tracking)
        )

    def test_field_independent_of_pivot_edge_tracking(self):
        """Verify the two tracking fields are independent."""
        encoding = {"A": 0, "B": 1}
        part_a = Partition((0,), encoding)
        part_b = Partition((1,), encoding)

        seq = TreeInterpolationSequence(
            current_pivot_edge_tracking=[None, part_a, None],
            current_subtree_tracking=[None, part_b, None],
        )

        # Should be independent
        self.assertNotEqual(
            seq.current_pivot_edge_tracking[1], seq.current_subtree_tracking[1]
        )


if __name__ == "__main__":
    unittest.main()


class TestSubtreeTrackingLengthInvariant(unittest.TestCase):
    """
    Property 1: Length Invariant

    For any TreeInterpolationSequence, the length of current_subtree_tracking
    SHALL equal the length of current_pivot_edge_tracking AND the length of
    interpolated_trees.

    **Feature: microsteps-api-integration, Property 1: Length Invariant**
    **Validates: Requirements 1.3, 3.1, 3.4**
    """

    def setUp(self):
        """Set up test trees."""
        from brancharchitect.parser import parse_newick

        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
        taxa_order = ["A", "B", "C", "D"]

        # Create two different trees for interpolation
        self.tree1 = parse_newick(
            "((A,B),(C,D));", order=taxa_order, encoding=self.encoding
        )
        self.tree2 = parse_newick(
            "((A,C),(B,D));", order=taxa_order, encoding=self.encoding
        )
        self.tree3 = parse_newick(
            "((A,D),(B,C));", order=taxa_order, encoding=self.encoding
        )

    def test_length_invariant_two_trees(self):
        """Test length invariant with 2 trees."""
        from brancharchitect.tree_interpolation.sequential_interpolation import (
            SequentialInterpolationBuilder,
        )

        builder = SequentialInterpolationBuilder()
        result = builder.build([self.tree1, self.tree2])

        # All three lists must have equal length
        self.assertEqual(
            len(result.interpolated_trees),
            len(result.current_pivot_edge_tracking),
            "interpolated_trees and current_pivot_edge_tracking must have equal length",
        )
        self.assertEqual(
            len(result.interpolated_trees),
            len(result.current_subtree_tracking),
            "interpolated_trees and current_subtree_tracking must have equal length",
        )

    def test_length_invariant_three_trees(self):
        """Test length invariant with 3 trees."""
        from brancharchitect.tree_interpolation.sequential_interpolation import (
            SequentialInterpolationBuilder,
        )

        builder = SequentialInterpolationBuilder()
        result = builder.build([self.tree1, self.tree2, self.tree3])

        # All three lists must have equal length
        self.assertEqual(
            len(result.interpolated_trees), len(result.current_pivot_edge_tracking)
        )
        self.assertEqual(
            len(result.interpolated_trees), len(result.current_subtree_tracking)
        )


class TestSubtreeTrackingPairingInvariant(unittest.TestCase):
    """
    Property 2: Pairing Invariant

    For any TreeInterpolationSequence and any index i,
    current_pivot_edge_tracking[i] is None if and only if
    current_subtree_tracking[i] is None.

    **Feature: microsteps-api-integration, Property 2: Pairing Invariant**
    **Validates: Requirements 1.4, 3.2, 3.3**
    """

    def setUp(self):
        """Set up test trees."""
        from brancharchitect.parser import parse_newick

        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
        taxa_order = ["A", "B", "C", "D"]

        self.tree1 = parse_newick(
            "((A,B),(C,D));", order=taxa_order, encoding=self.encoding
        )
        self.tree2 = parse_newick(
            "((A,C),(B,D));", order=taxa_order, encoding=self.encoding
        )

    def test_pairing_invariant(self):
        """Test that pivot_edge None ↔ subtree None for all indices."""
        from brancharchitect.tree_interpolation.sequential_interpolation import (
            SequentialInterpolationBuilder,
        )

        builder = SequentialInterpolationBuilder()
        result = builder.build([self.tree1, self.tree2])

        for i in range(len(result.interpolated_trees)):
            pivot_is_none = result.current_pivot_edge_tracking[i] is None
            subtree_is_none = result.current_subtree_tracking[i] is None

            self.assertEqual(
                pivot_is_none,
                subtree_is_none,
                f"At index {i}: pivot_edge is None ({pivot_is_none}) must equal "
                f"subtree is None ({subtree_is_none})",
            )

    def test_original_trees_have_none_tracking(self):
        """Test that original trees (delimiters) have None in both tracking lists."""
        from brancharchitect.tree_interpolation.sequential_interpolation import (
            SequentialInterpolationBuilder,
        )

        builder = SequentialInterpolationBuilder()
        result = builder.build([self.tree1, self.tree2])

        original_indices = result.get_original_tree_indices()

        for idx in original_indices:
            self.assertIsNone(
                result.current_pivot_edge_tracking[idx],
                f"Original tree at index {idx} should have None pivot_edge",
            )
            self.assertIsNone(
                result.current_subtree_tracking[idx],
                f"Original tree at index {idx} should have None subtree",
            )


class TestSerializationDeterminism(unittest.TestCase):
    """
    Property 3: Serialization Determinism

    For any Partition object, serializing it to index array format SHALL produce
    a sorted list of integer indices, and serializing the same Partition multiple
    times SHALL produce identical results.

    **Feature: microsteps-api-integration, Property 3: Serialization Determinism**
    **Validates: Requirements 2.2**
    """

    def test_serialization_produces_sorted_list(self):
        """Test that serialization produces sorted list of indices."""
        from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
            TreeInterpolationPipeline,
        )

        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Create partitions with indices in various orders
        part1 = Partition((2, 0, 1), encoding)  # Unsorted input
        part2 = Partition((3, 1), encoding)

        pipeline = TreeInterpolationPipeline()

        # Serialize
        result = pipeline._serialize_subtree_tracking([part1, part2, None])

        # Check sorted
        self.assertEqual(result[0], [0, 1, 2])  # Should be sorted
        self.assertEqual(result[1], [1, 3])  # Should be sorted
        self.assertIsNone(result[2])  # None stays None

    def test_serialization_is_deterministic(self):
        """Test that serializing the same partition multiple times gives identical results."""
        from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
            TreeInterpolationPipeline,
        )

        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
        part = Partition((2, 0, 3), encoding)

        pipeline = TreeInterpolationPipeline()

        # Serialize multiple times
        results = [pipeline._serialize_subtree_tracking([part])[0] for _ in range(5)]

        # All results should be identical
        for result in results:
            self.assertEqual(result, results[0])
            self.assertEqual(result, [0, 2, 3])

    def test_none_serialization(self):
        """Test that None values are preserved during serialization."""
        from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
            TreeInterpolationPipeline,
        )

        pipeline = TreeInterpolationPipeline()

        result = pipeline._serialize_subtree_tracking([None, None, None])

        self.assertEqual(result, [None, None, None])


class TestAPIResponseStructure(unittest.TestCase):
    """
    Test that subtree_tracking is correctly included in API response structure.

    **Feature: microsteps-api-integration, Task 5.4**
    **Validates: Requirements 2.1, 2.3**
    """

    def test_assemble_frontend_dict_includes_subtree_tracking(self):
        """Test that assemble_frontend_dict includes subtree_tracking field."""
        from webapp.services.trees.movie_data import MovieData
        from webapp.services.trees.frontend_builder import assemble_frontend_dict

        # Create MovieData with subtree_tracking
        movie_data = MovieData(
            interpolated_trees=[],
            tree_metadata=[],
            rfd_list=[],
            weighted_robinson_foulds_distance_list=[],
            sorted_leaves=["A", "B", "C"],
            tree_pair_solutions={},
            pivot_edge_tracking=[None, [0, 1], [0, 1], None],
            subtree_tracking=[None, [2], [2], None],
            file_name="test.nwk",
            window_size=1,
            window_step_size=1,
            msa_dict=None,
            pair_interpolation_ranges=[],
        )

        result = assemble_frontend_dict(movie_data)

        # Verify subtree_tracking is in response
        self.assertIn("subtree_tracking", result)
        self.assertEqual(result["subtree_tracking"], [None, [2], [2], None])

    def test_subtree_tracking_format_matches_pivot_edge_tracking(self):
        """Test that subtree_tracking has same format as pivot_edge_tracking."""
        from webapp.services.trees.movie_data import MovieData
        from webapp.services.trees.frontend_builder import assemble_frontend_dict

        movie_data = MovieData(
            interpolated_trees=[],
            tree_metadata=[],
            rfd_list=[],
            weighted_robinson_foulds_distance_list=[],
            sorted_leaves=["A", "B", "C", "D"],
            tree_pair_solutions={},
            pivot_edge_tracking=[None, [0, 1], None],
            subtree_tracking=[None, [2, 3], None],
            file_name="test.nwk",
            window_size=1,
            window_step_size=1,
            msa_dict=None,
            pair_interpolation_ranges=[],
        )

        result = assemble_frontend_dict(movie_data)

        # Both should be lists of same length
        self.assertEqual(
            len(result["subtree_tracking"]), len(result["pivot_edge_tracking"])
        )

        # Both should have same structure: List[Optional[List[int]]]
        for i in range(len(result["subtree_tracking"])):
            pivot_val = result["pivot_edge_tracking"][i]
            subtree_val = result["subtree_tracking"][i]

            # Both None or both list
            if pivot_val is None:
                self.assertIsNone(subtree_val)
            else:
                self.assertIsInstance(subtree_val, list)

    def test_create_empty_movie_data_includes_pivot_edge_tracking(self):
        """Test that create_empty_movie_data includes empty pivot_edge_tracking."""
        from webapp.services.trees.frontend_builder import create_empty_movie_data

        movie_data = create_empty_movie_data("empty.nwk")

        self.assertTrue(hasattr(movie_data, "pivot_edge_tracking"))
        self.assertEqual(movie_data.pivot_edge_tracking, [])

    def test_create_empty_movie_data_includes_subtree_tracking(self):
        """Test that create_empty_movie_data includes empty subtree_tracking."""
        from webapp.services.trees.frontend_builder import create_empty_movie_data

        movie_data = create_empty_movie_data("empty.nwk")

        self.assertTrue(hasattr(movie_data, "subtree_tracking"))
        self.assertEqual(movie_data.subtree_tracking, [])


class TestAggregationCorrectness(unittest.TestCase):
    """
    Property 4: Aggregation Correctness

    For any sequence of trees, the aggregated subtree_tracking in
    TreeInterpolationSequence SHALL contain exactly the concatenation of
    individual pair interpolation subtree tracking values, with None delimiters
    for original trees.

    **Feature: microsteps-api-integration, Property 4: Aggregation Correctness**
    **Validates: Requirements 1.1, 4.2**
    """

    def setUp(self):
        """Set up test trees."""
        from brancharchitect.parser import parse_newick

        self.encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
        taxa_order = ["A", "B", "C", "D"]

        self.tree1 = parse_newick(
            "((A,B),(C,D));", order=taxa_order, encoding=self.encoding
        )
        self.tree2 = parse_newick(
            "((A,C),(B,D));", order=taxa_order, encoding=self.encoding
        )
        self.tree3 = parse_newick(
            "((A,D),(B,C));", order=taxa_order, encoding=self.encoding
        )

    def test_aggregation_preserves_pair_tracking(self):
        """Test that aggregation preserves individual pair tracking values."""
        from brancharchitect.tree_interpolation.sequential_interpolation import (
            SequentialInterpolationBuilder,
        )

        builder = SequentialInterpolationBuilder()
        result = builder.build([self.tree1, self.tree2, self.tree3])

        # Get original tree indices (delimiters)
        original_indices = result.get_original_tree_indices()

        # For each original tree, tracking should be None
        for idx in original_indices:
            self.assertIsNone(
                result.current_subtree_tracking[idx],
                f"Original tree at index {idx} should have None subtree tracking",
            )

        # For interpolated trees, tracking should be non-None
        interpolated_indices = result.get_interpolated_tree_indices()
        for idx in interpolated_indices:
            self.assertIsNotNone(
                result.current_subtree_tracking[idx],
                f"Interpolated tree at index {idx} should have non-None subtree tracking",
            )

    def test_subtree_tracking_parallel_to_pivot_edge(self):
        """Test that subtree_tracking runs parallel to pivot_edge_tracking."""
        from brancharchitect.tree_interpolation.sequential_interpolation import (
            SequentialInterpolationBuilder,
        )

        builder = SequentialInterpolationBuilder()
        result = builder.build([self.tree1, self.tree2])

        # Both lists should have same length
        self.assertEqual(
            len(result.current_subtree_tracking),
            len(result.current_pivot_edge_tracking),
        )

        # None positions should match
        for i in range(len(result.current_subtree_tracking)):
            subtree_none = result.current_subtree_tracking[i] is None
            pivot_none = result.current_pivot_edge_tracking[i] is None
            self.assertEqual(
                subtree_none,
                pivot_none,
                f"At index {i}: subtree None ({subtree_none}) should match pivot None ({pivot_none})",
            )
