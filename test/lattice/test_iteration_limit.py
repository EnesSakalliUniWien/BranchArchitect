import unittest
from unittest.mock import MagicMock
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.parser.newick_parser import parse_newick


class TestIterationLimit(unittest.TestCase):
    def test_max_iters_terminates_loop(self):
        """Verify that solve_iteratively stops when max_iters is reached."""
        tree1_str = "((A,B),(C,D));"
        tree2_str = "((A,C),(B,D));"
        trees = parse_newick(tree1_str + tree2_str)

        solver = LatticeSolver(trees[0], trees[1])

        # Mock _identify_and_delete_jumping_taxa to always return False (continue loop)
        # This simulates a situation where we never converge or run out of taxa
        # (avoiding the natural stop conditions to force the iteration limit check)
        solve_result = False  # Return False to indicate "don't break"
        solver._identify_and_delete_jumping_taxa = MagicMock(return_value=False)

        # Run with small limit
        limit = 5
        print(f"Starting solver with max_iters={limit}")
        result = solver.solve_iteratively(max_iters=limit)

        # Verify call count
        call_count = solver._identify_and_delete_jumping_taxa.call_count
        print(f"Solver stopped after {call_count} iterations")

        self.assertEqual(
            call_count, limit, f"Should have run exactly {limit} iterations"
        )


if __name__ == "__main__":
    unittest.main()
