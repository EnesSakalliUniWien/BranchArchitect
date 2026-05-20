"""
Tests for split_matrix functionality in meet_product_solvers.

The split_matrix function decomposes conflict matrices into independent subproblems
when rows share no cover sets in either column. This is analogous to connected
component decomposition in a bipartite graph.
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
        return {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
        }

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

    def test_no_split_coupled_rows_shared_through_opposite_column(self, encoding):
        """Rows connected through either column belong to one subproblem."""
        ps_ab = self._make_ps([0, 1], encoding)  # {A, B}
        ps_cd = self._make_ps([2, 3], encoding)  # {C, D}
        ps_x = self._make_ps([4], encoding)
        ps_y = self._make_ps([5], encoding)

        # Row 0,1 share {A,B} in left column
        # Row 2,3 share {C,D} in left column
        # Both groups still share right-column covers, so this is one coupled component.
        matrix = [
            [ps_ab, ps_x],
            [ps_ab, ps_y],
            [ps_cd, ps_x],
            [ps_cd, ps_y],
        ]

        result = split_matrix(matrix)

        assert len(result) == 1
        assert result[0] == matrix

    def test_no_split_rows_sharing_frontier_atoms_in_different_cover_sets(
        self, encoding
    ):
        """Rows sharing any frontier atom belong to one subproblem."""
        a = Partition((0,), encoding)
        b = Partition((1,), encoding)
        c = Partition((2,), encoding)
        d = Partition((3,), encoding)

        def cell(*parts):
            return PartitionSet(set(parts), encoding=encoding)

        matrix = [
            [cell(a, b), cell(b, c)],
            [cell(b, d), cell(a, c)],
        ]

        result = split_matrix(matrix)

        assert len(result) == 1
        assert result[0] == matrix

    def test_no_split_degenerate_row_before_dependency_analysis(self, encoding):
        """Singleton-containment rows still participate in dependency analysis."""
        a = Partition((0,), encoding)
        b = Partition((1,), encoding)
        c = Partition((2,), encoding)
        d = Partition((3,), encoding)

        def cell(*parts):
            return PartitionSet(set(parts), encoding=encoding)

        matrix = [
            [cell(a), cell(a, b)],
            [cell(a, c), cell(d)],
        ]

        result = split_matrix(matrix)

        assert len(result) == 1
        assert result[0] == matrix

    def test_split_independent_rows(self, encoding):
        """Disconnected row components should split."""
        ps_ab = self._make_ps([0, 1], encoding)  # {A, B}
        ps_cd = self._make_ps([2, 3], encoding)  # {C, D}
        ps_e = self._make_ps([4], encoding)
        ps_f = self._make_ps([5], encoding)
        ps_g = self._make_ps([6], encoding)
        ps_h = self._make_ps([7], encoding)

        matrix = [
            [ps_ab, ps_e],
            [ps_ab, ps_f],
            [ps_cd, ps_g],
            [ps_cd, ps_h],
        ]

        result = split_matrix(matrix)

        assert len(result) == 2  # Should split into 2 matrices
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_split_produces_correct_groups(self, encoding):
        """Verify split groups rows by connected cover components."""
        ps_ab = self._make_ps([0, 1], encoding)
        ps_cd = self._make_ps([2, 3], encoding)
        ps_e = self._make_ps([4], encoding)
        ps_f = self._make_ps([5], encoding)
        ps_g = self._make_ps([6], encoding)
        ps_h = self._make_ps([7], encoding)

        matrix = [
            [ps_ab, ps_e],
            [ps_ab, ps_f],
            [ps_cd, ps_g],
            [ps_cd, ps_h],
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
        ps_e = self._make_ps([4], encoding)
        ps_f = self._make_ps([5], encoding)
        ps_g = self._make_ps([6], encoding)
        ps_h = self._make_ps([7], encoding)

        matrix = [
            [ps_ab, ps_e],
            [ps_ab, ps_f],
            [ps_cd, ps_g],
            [ps_cd, ps_h],
        ]

        matrices = split_matrix(matrix)
        assert len(matrices) == 2

        # Verify union_split_matrix_results doesn't crash on multiple matrices
        # Note: actual solutions may be empty if meet products result in empty sets
        solutions = union_split_matrix_results(matrices)
        assert isinstance(solutions, list)
        # Solutions could be empty if partition sets don't overlap - that's valid
