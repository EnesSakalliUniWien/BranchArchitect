import pytest

from brancharchitect.io import parse_newick


def _leaf_names(newick: str) -> list[list[str]]:
    # Sorted: the parser normalises child order.
    return [
        sorted(leaf.name for leaf in tree.get_leaves())
        for tree in parse_newick(newick, force_list=True)
    ]


@pytest.mark.parametrize(
    "newick",
    [
        "(A, B, (C, D));\n(A,C,(B,D));",  # padding around names
        "(A,B,(C,D));\r\n(A,C,(B,D));\r\n",  # Windows line endings
        "﻿(A,B,(C,D));\n(A,C,(B,D));",  # UTF-8 byte order mark
    ],
)
def test_layout_characters_are_not_part_of_taxon_names(newick: str) -> None:
    assert _leaf_names(newick) == [["A", "B", "C", "D"], ["A", "B", "C", "D"]]


def test_bare_leaf_trees_after_the_first_are_kept() -> None:
    assert _leaf_names("A;\nB;\nC;") == [["A"], ["B"], ["C"]]


def test_a_comment_after_the_last_tree_adds_no_tree() -> None:
    assert _leaf_names("((A,B),C); \n [comment]") == [["A", "B", "C"]]


@pytest.mark.parametrize(
    ("newick", "message"),
    [
        ("", "No Newick tree"),
        (" \r\n\t", "No Newick tree"),
        ("(A,B));", "no matching '\\('"),
        ("((A,B),C;", "never closed"),
        ("((A,B),C", "never closed"),
    ],
)
def test_empty_or_unbalanced_input_is_rejected(newick: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_newick(newick)
