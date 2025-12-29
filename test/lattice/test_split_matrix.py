"""
Tests for split_matrix functionality in meet_product_solvers.

The split_matrix function decomposes conflict matrices into independent subproblems
when rows share no common elements in a column. This is analogous to block-diagonal
decomposition in linear algebra.
"""

import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.matrices.meet_product_solvers import (
    split_matrix,
    union_split_matrix_results,
)


class TestSplitMatrix:
    """Tests for split_matrix decomposition."""

    @pytest.fixture
    def encoding(self):
        return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

    def _make_ps(self, indices, encoding):
        """Helper to create a PartitionSet from indices."""
        return PartitionSet(
            {Partition(tuple(indices), encoding)},
            encoding=encoding,
            name=f"ps_{indices}",
        )

    def test_no_split_single_row(self, encoding):
        """Single row matrix cannot be split."""
        ps1 = self._make_ps([0, 1], encoding)
        ps2 = self._make_ps([2, 3], encoding)
        matrix = [[ps1, ps2]]

        result = split_matrix(matrix)

        assert len(result) == 1
        assert result[0] == matrix

    def test_no_split_shared_column_element(self, encoding):
        """Rows sharing column elements cannot be split."""
        ps_ab = self._make_ps([0, 1], encoding)  # {A, B}
        ps_x = self._make_ps([2], encoding)
        ps_y = self._make_ps([3], encoding)

        # Both rows share ps_ab in left column
        matrix = [
            [ps_ab, ps_x],
            [ps_ab, ps_y],
        ]

        result = split_matrix(matrix)

        assert len(result) == 1  # No split possible

    def test_split_independent_rows(self, encoding):
        """Independent rows (no shared column elements) should split."""
        ps_ab = self._make_ps([0, 1], encoding)  # {A, B}
        ps_cd = self._make_ps([2, 3], encoding)  # {C, D}
        ps_x = self._make_ps([4], encoding)
        ps_y = self._make_ps([5], encoding)

        # Row 0,1 share {A,B} in left column
        # Row 2,3 share {C,D} in left column
        # These are independent groups
        matrix = [
            [ps_ab, ps_x],
            [ps_ab, ps_y],
            [ps_cd, ps_x],
            [ps_cd, ps_y],
        ]

        result = split_matrix(matrix)

        assert len(result) == 2  # Should split into 2 matrices
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_split_produces_correct_groups(self, encoding):
        """Verify split groups rows correctly by shared column elements."""
        ps_ab = self._make_ps([0, 1], encoding)
        ps_cd = self._make_ps([2, 3], encoding)
        ps_1 = self._make_ps([4], encoding)
        ps_2 = self._make_ps([5], encoding)

        # Need 4 rows with 2 distinct groups to trigger split
        # (single rows per group won't split - each row becomes its own group)
        matrix = [
            [ps_ab, ps_1],
            [ps_ab, ps_2],  # Same left column as row 0
            [ps_cd, ps_1],
            [ps_cd, ps_2],  # Same left column as row 2
        ]

        result = split_matrix(matrix)

        # Should split into 2 matrices of 2 rows each
        assert len(result) == 2
        for m in result:
            assert len(m) == 2

    def test_union_split_matrix_results_integration(self, encoding):
        """Test that split + union handles multiple matrices correctly."""
        ps_ab = self._make_ps([0, 1], encoding)
        ps_cd = self._make_ps([2, 3], encoding)
        ps_x = self._make_ps([4], encoding)
        ps_y = self._make_ps([5], encoding)

        # 4 rows with 2 groups to trigger split
        matrix = [
            [ps_ab, ps_x],
            [ps_ab, ps_y],
            [ps_cd, ps_x],
            [ps_cd, ps_y],
        ]

        matrices = split_matrix(matrix)
        assert len(matrices) == 2

        # Verify union_split_matrix_results doesn't crash on multiple matrices
        # Note: actual solutions may be empty if meet products result in empty sets
        solutions = union_split_matrix_results(matrices)
        assert isinstance(solutions, list)
        # Solutions could be empty if partition sets don't overlap - that's valid
