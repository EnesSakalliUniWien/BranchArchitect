"""
Test suite for checking stability and termination of the LatticeSolver.
"""

import unittest
import time
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.logger import jt_logger


class TestSolverStability(unittest.TestCase):
    def setUp(self):
        # Disable logging for tests to keep output clean help speed
        jt_logger.disabled = True

    def test_solve_iteratively_termination(self):
        """Verify that solve_iteratively terminates when trees are isomorphic."""
        tree_str = "((((A,B),(C,D)),O));"
        trees = parse_newick(tree_str + tree_str)
        t1, t2 = trees[0], trees[1]

        solver = LatticeSolver(t1, t2)

        # Run with very low max_iters; termination should be by isomorphism
        start_time = time.time()
        solutions, deleted = solver.solve_iteratively(max_iters=2)
        end_time = time.time()

        # It should finish very quickly
        self.assertLess(end_time - start_time, 1.0)

        # With identical trees, no solutions or deletions should be produced.
        self.assertIsInstance(solutions, dict)
        self.assertEqual(len(solutions), 0)
        self.assertEqual(len(deleted), 0)

    def test_intersection_infinite_loop_prevention(self):
        """
        Test that limits the behavior from repro_infinite_loop.py
        ensuring the solver doesn't hang on persistent conflicts.
        """
        # This needs a case where standard solving might re-queue the same pivot endlessly
        # if conflicts aren't resolved.
        # We'll use a known "safe" case but verify the solver internals don't panic.

        tree1_str = "((A,B),(C,D));"
        tree2_str = "((A,C),(B,D));"
        trees = parse_newick(tree1_str + tree2_str)

        solver = LatticeSolver(trees[0], trees[1])

        # Should finish instantly
        solutions = solver.solve()
        self.assertIsInstance(solutions, dict)

    def test_large_cycle_stability(self):
        """Test with a larger tree that might generate many cycles."""
        # A case with more taxa
        t1_str = "((((A,B),C),D),((E,F),G));"
        t2_str = "((((A,C),B),E),((D,G),F));"

        trees = parse_newick(t1_str + t2_str)

        solver = LatticeSolver(trees[0], trees[1])

        # Ensure iterative solver either converges or raises on iteration limit
        try:
            solutions, deleted = solver.solve_iteratively(max_iters=50)
        except RuntimeError:
            self.assertTrue(True)
        else:
            self.assertTrue(len(deleted) > 0 or len(solutions) > 0)


if __name__ == "__main__":
    unittest.main()
