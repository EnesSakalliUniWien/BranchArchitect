"""
Tests for subtree highlights in tree interpolation.

Tests cover:
- TreeInterpolationSequence field existence
- Subtree highlight propagation through SequentialInterpolationBuilder
- API response structure
"""

import unittest

from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.types import TreeInterpolationSequence


class TestTreeInterpolationSequenceSubtreeHighlights(unittest.TestCase):
    """Test that TreeInterpolationSequence has the current_subtree_highlights field."""

    def test_field_exists_and_initializes_to_empty_list(self):
        """Verify current_subtree_highlights field exists and initializes to empty list."""
        seq = TreeInterpolationSequence()

        # Field should exist
        self.assertTrue(hasattr(seq, "current_subtree_highlights"))

        # Should initialize to empty list
        self.assertEqual(seq.current_subtree_highlights, [])
        self.assertIsInstance(seq.current_subtree_highlights, list)

    def test_field_can_be_set_in_constructor(self):
        """Verify current_subtree_highlights can be set via constructor."""
        encoding = {"A": 0, "B": 1}
        part_a = Partition((0,), encoding)

        tracking = [None, part_a, part_a, None]

        seq = TreeInterpolationSequence(current_subtree_highlights=tracking)

        self.assertEqual(len(seq.current_subtree_highlights), 4)
        self.assertIsNone(seq.current_subtree_highlights[0])
        self.assertEqual(seq.current_subtree_highlights[1], part_a)
        self.assertEqual(seq.current_subtree_highlights[2], part_a)
        self.assertIsNone(seq.current_subtree_highlights[3])

    def test_field_type_matches_active_pivot_edges(self):
        """Verify current_subtree_highlights has same type as active_pivot_edges."""
        seq = TreeInterpolationSequence()

        # Both should be list[Optional[Partition]]
        self.assertEqual(
            type(seq.current_subtree_highlights), type(seq.active_pivot_edges)
        )

    def test_field_independent_of_active_pivot_edges(self):
        """Verify the two frame-aligned highlight fields are independent."""
        encoding = {"A": 0, "B": 1}
        part_a = Partition((0,), encoding)
        part_b = Partition((1,), encoding)

        seq = TreeInterpolationSequence(
            active_pivot_edges=[None, part_a, None],
            current_subtree_highlights=[None, part_b, None],
        )

        # Should be independent
        self.assertNotEqual(
            seq.active_pivot_edges[1], seq.current_subtree_highlights[1]
        )


if __name__ == "__main__":
    unittest.main()


class TestSubtreeHighlightsLengthInvariant(unittest.TestCase):
    """
    Property 1: Length Invariant

    For any TreeInterpolationSequence, the length of current_subtree_highlights
    SHALL equal the length of active_pivot_edges AND the length of
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
            len(result.active_pivot_edges),
            "interpolated_trees and active_pivot_edges must have equal length",
        )
        self.assertEqual(
            len(result.interpolated_trees),
            len(result.current_subtree_highlights),
            "interpolated_trees and current_subtree_highlights must have equal length",
        )

    def test_length_invariant_three_trees(self):
        """Test length invariant with 3 trees."""
        from brancharchitect.tree_interpolation.sequential_interpolation import (
            SequentialInterpolationBuilder,
        )

        builder = SequentialInterpolationBuilder()
        result = builder.build([self.tree1, self.tree2, self.tree3])

        # All three lists must have equal length
        self.assertEqual(len(result.interpolated_trees), len(result.active_pivot_edges))
        self.assertEqual(
            len(result.interpolated_trees), len(result.current_subtree_highlights)
        )


class TestSubtreeHighlightsPairingInvariant(unittest.TestCase):
    """
    Property 2: Pairing Invariant

    For any TreeInterpolationSequence and any index i,
    active_pivot_edges[i] is None if and only if
    current_subtree_highlights[i] is None.

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
            pivot_is_none = result.active_pivot_edges[i] is None
            subtree_is_none = result.current_subtree_highlights[i] is None

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
                result.active_pivot_edges[idx],
                f"Original tree at index {idx} should have None pivot_edge",
            )
            self.assertIsNone(
                result.current_subtree_highlights[idx],
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
        from brancharchitect.io import serialize_subtree_highlights

        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Create partitions with indices in various orders
        part1 = Partition((2, 0, 1), encoding)  # Unsorted input
        part2 = Partition((3, 1), encoding)

        # Serialize
        result = serialize_subtree_highlights([[part1], [part2], None])

        # Check sorted
        self.assertEqual(result[0], [[0, 1, 2]])  # Should be sorted
        self.assertEqual(result[1], [[1, 3]])  # Should be sorted
        self.assertIsNone(result[2])  # None stays None

    def test_serialization_is_deterministic(self):
        """Test that serializing the same partition multiple times gives identical results."""
        from brancharchitect.io import serialize_subtree_highlights

        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
        part = Partition((2, 0, 3), encoding)

        # Serialize multiple times
        results = [serialize_subtree_highlights([[part]])[0] for _ in range(5)]

        # All results should be identical
        for result in results:
            self.assertEqual(result, results[0])
            self.assertEqual(result, [[0, 2, 3]])

    def test_none_serialization(self):
        """Test that None values are preserved during serialization."""
        from brancharchitect.io import serialize_subtree_highlights

        result = serialize_subtree_highlights([None, None, None])

        self.assertEqual(result, [None, None, None])


class TestAPIResponseStructure(unittest.TestCase):
    """
    Test that subtree_highlight_tracking is correctly included in API response structure.

    **Feature: microsteps-api-integration, Task 5.4**
    **Validates: Requirements 2.1, 2.3**
    """

    def test_assemble_frontend_metadata_includes_subtree_highlight_tracking(self):
        """Test that assemble_frontend_metadata includes subtree_highlight_tracking field."""
        from webapp.services.trees.movie_data import MovieData
        from webapp.services.trees.frontend_builder import assemble_frontend_metadata

        movie_data = MovieData(
            interpolated_trees=[],
            annotation_definitions=[],
            tree_name_definitions=[],
            split_definitions=[],
            frames=[],
            pairs=[],
            temporal_events=[],
            pair_metrics={"rows": [], "semantics": {}},
            subtree_highlight_tracking=[None, [[2]], [[2]], None],
            file_name="test.nwk",
            window_size=1,
            window_step_size=1,
            msa_dict=None,
        )

        result = assemble_frontend_metadata(movie_data)

        self.assertIn("subtree_highlight_tracking", result)
        self.assertEqual(
            result["subtree_highlight_tracking"], [None, [[2]], [[2]], None]
        )

    def test_subtree_highlight_tracking_is_frame_aligned(self):
        """Test that subtree_highlight_tracking stays aligned to frame rows."""
        from webapp.services.trees.movie_data import MovieData
        from webapp.services.trees.frontend_builder import assemble_frontend_metadata

        frames = [
            {"frame_index": 0},
            {"frame_index": 1},
            {"frame_index": 2},
        ]
        movie_data = MovieData(
            interpolated_trees=[],
            annotation_definitions=[],
            tree_name_definitions=[],
            split_definitions=[],
            frames=frames,
            pairs=[],
            temporal_events=[],
            pair_metrics={"rows": [], "semantics": {}},
            subtree_highlight_tracking=[None, [[2, 3]], None],
            file_name="test.nwk",
            window_size=1,
            window_step_size=1,
            msa_dict=None,
        )

        result = assemble_frontend_metadata(movie_data)

        self.assertEqual(
            len(result["subtree_highlight_tracking"]),
            len(result["frames"]),
        )

        for i in range(len(result["subtree_highlight_tracking"])):
            subtree_val = result["subtree_highlight_tracking"][i]

            if i != 1:
                self.assertIsNone(subtree_val)
            else:
                self.assertIsInstance(subtree_val, list)
                self.assertTrue(all(isinstance(group, list) for group in subtree_val))

    def test_create_empty_movie_data_excludes_duplicate_pivot_tracking(self):
        """Test that create_empty_movie_data has no duplicate pivot tracking payload."""
        from webapp.services.trees.frontend_builder import create_empty_movie_data

        movie_data = create_empty_movie_data("empty.nwk")

        self.assertFalse(hasattr(movie_data, "pivot_edge_tracking"))

    def test_create_empty_movie_data_includes_subtree_highlight_tracking(self):
        """Test that create_empty_movie_data includes empty subtree_highlight_tracking."""
        from webapp.services.trees.frontend_builder import create_empty_movie_data

        movie_data = create_empty_movie_data("empty.nwk")

        self.assertTrue(hasattr(movie_data, "subtree_highlight_tracking"))
        self.assertEqual(movie_data.subtree_highlight_tracking, [])


class TestAggregationCorrectness(unittest.TestCase):
    """
    Property 4: Aggregation Correctness

    For any sequence of trees, the aggregated subtree highlight data in
    TreeInterpolationSequence SHALL contain exactly the concatenation of
    individual pair interpolation subtree highlight values, with None delimiters
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
                result.current_subtree_highlights[idx],
                f"Original tree at index {idx} should have no subtree highlights",
            )

        # For interpolated trees, tracking should be non-None
        interpolated_indices = result.get_interpolated_tree_indices()
        for idx in interpolated_indices:
            self.assertIsNotNone(
                result.current_subtree_highlights[idx],
                f"Interpolated tree at index {idx} should have subtree highlights",
            )

    def test_subtree_highlights_parallel_to_pivot_edge(self):
        """Test that subtree highlight data runs parallel to active_pivot_edges."""
        from brancharchitect.tree_interpolation.sequential_interpolation import (
            SequentialInterpolationBuilder,
        )

        builder = SequentialInterpolationBuilder()
        result = builder.build([self.tree1, self.tree2])

        # Both lists should have same length
        self.assertEqual(
            len(result.current_subtree_highlights),
            len(result.active_pivot_edges),
        )

        # None positions should match
        for i in range(len(result.current_subtree_highlights)):
            subtree_none = result.current_subtree_highlights[i] is None
            pivot_none = result.active_pivot_edges[i] is None
            self.assertEqual(
                subtree_none,
                pivot_none,
                f"At index {i}: subtree None ({subtree_none}) should match pivot None ({pivot_none})",
            )
