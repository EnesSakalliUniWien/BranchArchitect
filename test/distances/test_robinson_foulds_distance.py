from brancharchitect.distances.distances import (
    relative_robinson_foulds_distance,
    weighted_robinson_foulds_distance,
)
from brancharchitect.parser.newick_parser import parse_newick


def test_relative_robinson_foulds_uses_textbook_unrooted_bipartitions():
    trees = parse_newick("((A:1,B:1):1,(C:1,D:1):1);((A:1,C:1):1,(B:1,D:1):1);")

    assert relative_robinson_foulds_distance(trees[0], trees[1]) == 1.0


def test_relative_robinson_foulds_is_root_invariant():
    trees = parse_newick("((A:1,B:1):1,(C:1,D:1):1);(A:1,(B:1,(C:1,D:1):1):1);")

    assert relative_robinson_foulds_distance(trees[0], trees[1]) == 0.0


def test_relative_robinson_foulds_unresolved_trees_without_internal_splits_are_zero():
    trees = parse_newick("(A:1,B:1,C:1,D:1);(A:1,B:1,C:1,D:1);")

    assert relative_robinson_foulds_distance(trees[0], trees[1]) == 0.0


def test_weighted_robinson_foulds_keeps_raw_branch_length_difference_above_one():
    trees = parse_newick("((A:1,B:1):1,C:1);((A:1,B:1):2.5,C:1);")

    assert weighted_robinson_foulds_distance(trees[0], trees[1]) == 1.5
