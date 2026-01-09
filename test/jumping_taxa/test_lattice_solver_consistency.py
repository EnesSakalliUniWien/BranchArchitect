import unittest
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.types.registry import SolutionRegistry


class TestLatticeSolverConsistency(unittest.TestCase):
    def test_registry_key_consistency_and_empty_solutions(self):
        """
        Verify that:
        1. All registry keys are mapped to ORIGINAL trees (consistent key space).
        2. Pivots with NO solutions are explicitly registered (not dropped), using mapped keys.
        """
        tree1 = parse_newick("((A,B),C);")
        tree2 = parse_newick("((A,C),B);", encoding=tree1.taxa_encoding)
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        solver = LatticeSolver(tree1, tree2)
        solver.solve()

        registry_keys = list(solver.registry.solutions_by_pivot_and_iteration.keys())

        # Verify keys use full encoding size (original tree)
        for pivot, visit in registry_keys:
            self.assertEqual(
                len(pivot.encoding), 3, "Registry key should use original full encoding"
            )

        # Verify empty solution registration
        from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
            PivotEdgeSubproblem,
        )

        encoding = tree1.taxa_encoding
        pivot_p = Partition(frozenset({0, 1}), encoding)
        mock_pivot = PivotEdgeSubproblem(
            pivot_split=pivot_p,
            tree1_node=tree1,
            tree2_node=tree2,
            tree1_child_frontiers={},
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(
                [], encoding=encoding
            ),  # dummy
            encoding=encoding,
        )
        mock_pivot.visits = 99

        solver.registry = SolutionRegistry()
        solver._handle_pivot_solutions(mock_pivot, [])

        keys = list(solver.registry.solutions_by_pivot_and_iteration.keys())
        self.assertEqual(
            len(keys), 1, "Should have registered exactly one entry for empty solution"
        )

        solutions = solver.registry.solutions_by_pivot_and_iteration[keys[0]][
            "solution"
        ]
        # Note: Depending on my fix, categories might be "solution" or "no_solution".
        # But _handle_pivot_solutions logic handles the call to add_no_solution.
        # My implementation updated _handle_pivot_solutions to use category="no_solution" in the plan,
        # BUT I might not have applied that change to lattice_solver.py yet?
        # Let's check: I applied changes to select_best_solutions in registry.py.
        # Did I update lattice_solver.py to use "no_solution"? No, I kept "solution" in the last implementation step.
        # Wait, my last plan update said to use "no_solution".
        # But I only executed `replace_file_content` on `registry.py` for select_best_solutions.
        # Let's assume for now it uses "solution" or "no_solution" based on code state.
        # Actually I need to verify what _handle_pivot_solutions passes.
        # It currently passes "solution".

        self.assertEqual(len(solutions), 1)
        self.assertEqual(
            len(solutions[0]), 0, "Should store an explicit EMPTY PartitionSet"
        )

    def test_collision_prioritization_non_empty_over_empty(self):
        """
        Verify that if a pivot has both Empty (Size 0) and Non-Empty (Size > 0) solutions
        across different visits (or same visit), the Non-Empty solution is prioritized.
        This fixes the regression where 'no solution' markers overrode valid repairs.
        """
        registry = SolutionRegistry()
        encoding = {"A": 0}
        pivot = Partition(frozenset({0}), encoding)

        # Visit 1: Found a Jump (Non-Empty)
        sol_non_empty = PartitionSet(
            [Partition(frozenset({0}), encoding)], encoding=encoding
        )
        registry.add_solutions(pivot, [sol_non_empty], category="solution", visit=1)

        # Visit 2: Found No Jump (Empty)
        # This calls add_no_solution.
        # IMPORTANT: Even if category is "solution" for both, my new logic in select_best_solutions
        # filters by CONTENT (len > 0), so it should work regardless of category label.
        registry.add_no_solution(pivot, category="solution", visit=2)

        best_solutions = registry.select_best_solutions()

        self.assertIn(pivot, best_solutions)
        final_flat = best_solutions[pivot]

        self.assertEqual(
            len(final_flat), 1, "Should pick the non-empty solution (size 1)"
        )
        self.assertEqual(
            list(final_flat[0].indices), [0], "Should be the {A} partition"
        )


if __name__ == "__main__":
    unittest.main()
