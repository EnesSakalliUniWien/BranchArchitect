"""Tests for identify_and_delete_jumping_taxa function."""

import pytest
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.solvers.identify_jumping_taxa import (
    identify_and_delete_jumping_taxa,
)


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

        # Expect ValueError because {A,B} is not a clade in tree2
        with pytest.raises(ValueError, match="does not correspond to a valid subtree"):
            identify_and_delete_jumping_taxa(
                solver.current_t1,
                solver.current_t2,
                solver.deleted_taxa_per_iteration,
                {partition: [partition]},
                1,
            )

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
        should_break = identify_and_delete_jumping_taxa(
            solver.current_t1,
            solver.current_t2,
            solver.deleted_taxa_per_iteration,
            {},
            1,
        )

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
        # Mock pivot edge for dict key
        dummy_pivot = Partition(frozenset({0}), tree1.taxa_encoding)

        # Expect strict error because {A,B} is not a clade in tree2
        with pytest.raises(ValueError, match="does not correspond to a valid subtree"):
            identify_and_delete_jumping_taxa(
                solver.current_t1,
                solver.current_t2,
                solver.deleted_taxa_per_iteration,
                {dummy_pivot: [partition]},
                1,
            )
