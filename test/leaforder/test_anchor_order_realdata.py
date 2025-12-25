"""
Integration-style tests for anchor_order using repository Newick fixtures.

Files:
- current_testfiles/small_example.newick (first two trees)
- test-data/reverse_test_tree_moving_updwards.tree (two trees)
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.leaforder.anchor_order import derive_order_for_pair


def _parse_two_trees_from_file(path: str):
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    t1 = parse_newick(lines[0])
    t2 = parse_newick(lines[1], list(t1.get_current_order()))
    return t1, t2


def _are_contiguous(order: list[str], a: str, b: str) -> bool:
    ia = order.index(a)
    ib = order.index(b)
    return abs(ia - ib) == 1


def test_anchor_order_small_example_contiguous_outgroup():
    """
    On the small example trees, O1 and O2 are an outgroup in both.
    After derive_order_for_pair, O1 and O2 should remain contiguous in t1.
    """
    t1, t2 = _parse_two_trees_from_file("test-data/current_testfiles/small_example.newick")

    # Run ordering
    derive_order_for_pair(t1, t2)

    o1 = list(t1.get_current_order())
    assert _are_contiguous(o1, "O1", "O2")


def test_anchor_order_reverse_tree_outgroup_contiguous():
    """
    For the reverse moving upwards pair, (O1,O2) is a stable outgroup.
    After derive_order_for_pair, O1 and O2 remain adjacent in t1.
    """
    t1, t2 = _parse_two_trees_from_file(
        "test-data/reverse_test_tree_moving_updwards.tree"
    )

    derive_order_for_pair(t1, t2)

    o1 = list(t1.get_current_order())
    assert _are_contiguous(o1, "O1", "O2")

