import unittest
from unittest.mock import MagicMock
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.parser.newick_parser import parse_newick


class TestIterationLimit(unittest.TestCase):
    def test_max_iters_ignored_when_isomorphic(self):
        """Verify that isomorphic trees return immediately regardless of max_iters."""
        tree_str = "((A,B),(C,D));"
        trees = parse_newick(tree_str + tree_str)

        solver = LatticeSolver(trees[0], trees[1])
        solver._identify_and_delete_jumping_taxa = MagicMock()

        solutions, deleted = solver.solve_iteratively(max_iters=0)

        assert solutions == {}
        assert deleted == []
        assert solver._identify_and_delete_jumping_taxa.call_count == 0


if __name__ == "__main__":
    unittest.main()
