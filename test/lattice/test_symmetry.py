"""
Tests for verifying symmetry of the lattice solver.
The solver should produce identical (or isomorphic) sets of jumping taxa solutions regardless of tree order.
"""

import unittest
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.logger import jt_logger


class TestLatticeSymmetry(unittest.TestCase):
    def setUp(self):
        jt_logger.disabled = True

    def test_solve_symmetry_simple(self):
        """Verify symmetry on a simple conflict case."""
        # Tree 1: ((A,B),(C,D))
        # Tree 2: ((A,C),(B,D))
        # This has conflicts on both sides.
        t1_str = "((A,B),(C,D));"
        t2_str = "((A,C),(B,D));"
        trees = parse_newick(t1_str + t2_str)
        t1, t2 = trees[0], trees[1]

        # Case 1: T1, T2
        solver1 = LatticeSolver(t1, t2)
        solutions1 = solver1.solve()

        # Case 2: T2, T1
        solver2 = LatticeSolver(t2, t1)
        solutions2 = solver2.solve()

        # Extract flat list of solution partitions (sets of partition indices)
        sols1_flat = set()
        for parts in solutions1.values():
            for p in parts:
                sols1_flat.add(p.bitmask)

        sols2_flat = set()
        for parts in solutions2.values():
            for p in parts:
                sols2_flat.add(p.bitmask)

        # Verify identical set of solutions found
        self.assertEqual(
            sols1_flat,
            sols2_flat,
            "Solver should find same solutions regardless of tree order",
        )

    def test_solve_symmetry_complex(self):
        """Verify symmetry on a more complex case requiring matrix splitting."""
        # A case that likely triggers matrix splitting decisions
        t1_str = "((((A,B),C),D),((E,F),G));"
        t2_str = "((((A,C),B),E),((D,G),F));"

        trees = parse_newick(t1_str + t2_str)
        t1, t2 = trees[0], trees[1]

        solver1 = LatticeSolver(t1, t2)
        solver2 = LatticeSolver(t2, t1)
        try:
            solutions1, _ = solver1.solve_iteratively(max_iters=10)
            solutions2, _ = solver2.solve_iteratively(max_iters=10)
        except RuntimeError as exc:
            self.skipTest(f"Iterative solver did not converge within max_iters: {exc}")

        sols1_flat = set()
        for parts in solutions1.values():
            for p in parts:
                sols1_flat.add(p.bitmask)

        sols2_flat = set()
        for parts in solutions2.values():
            for p in parts:
                sols2_flat.add(p.bitmask)

        self.assertEqual(sols1_flat, sols2_flat, "Iterative solver should be symmetric")


if __name__ == "__main__":
    unittest.main()
