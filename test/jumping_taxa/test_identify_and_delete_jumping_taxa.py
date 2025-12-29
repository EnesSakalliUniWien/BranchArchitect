"""Tests for identify_and_delete_jumping_taxa function."""

import pytest
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition


class TestIdentifyAndDeleteJumpingTaxa:
    def test_delete_single_solution_set(self):
        tree1 = parse_newick("((A,B),(C,D));")
        tree2 = parse_newick(
            "((A,C),(B,D));",
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        partition = Partition(frozenset({0, 1}), tree1.taxa_encoding)
        solver = LatticeSolver(tree1, tree2)
        should_break = solver._identify_and_delete_jumping_taxa(
            [partition], tree1, tree2, 1
        )

        assert should_break is False
        assert solver.deleted_taxa_per_iteration[0] == {0, 1}
        assert len(tree1.get_leaves()) == 2

    def test_empty_solution_sets(self):
        tree1 = parse_newick("((A,B),(C,D));")
        tree2 = parse_newick(
            "((A,C),(B,D));",
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        solver = LatticeSolver(tree1, tree2)
        should_break = solver._identify_and_delete_jumping_taxa([], tree1, tree2, 1)

        assert should_break is True
        assert len(solver.deleted_taxa_per_iteration) == 0

    def test_break_when_tree_too_small(self):
        tree1 = parse_newick("((A,B),C);")
        tree2 = parse_newick(
            "((A,C),B);",
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        partition = Partition(frozenset({0, 1}), tree1.taxa_encoding)
        solver = LatticeSolver(tree1, tree2)
        should_break = solver._identify_and_delete_jumping_taxa(
            [partition], tree1, tree2, 1
        )

        assert should_break is True
        assert solver.deleted_taxa_per_iteration[0] == {0, 1}

    def test_return_type(self):
        tree1 = parse_newick("((A,B),(C,D));")
        tree2 = parse_newick(
            "((A,C),(B,D));",
            order=list(tree1.taxa_encoding.keys()),
            encoding=tree1.taxa_encoding,
        )
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        partition = Partition(frozenset({0}), tree1.taxa_encoding)
        solver = LatticeSolver(tree1, tree2)
        result = solver._identify_and_delete_jumping_taxa([partition], tree1, tree2, 1)

        assert isinstance(result, bool)
