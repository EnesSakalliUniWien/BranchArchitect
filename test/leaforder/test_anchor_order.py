"""
Tests for anchor-based leaf ordering (anchor_order.py).

We validate that:
- Movers are pushed to extremes with alternating ping-pong directions.
- Stable anchors are ordered deterministically and blocks are respected.
- Root-level alignment runs when there are no differing edges.
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.leaforder.anchor_order import (
    blocked_order_and_apply,
    derive_order_for_pair,
)
from brancharchitect.leaforder.split_analysis import get_common_splits


def _as_tree(parsed: Node | list[Node]) -> Node:
    if isinstance(parsed, list):
        return parsed[0]
    return parsed


def _pair(src: str, dst: str) -> tuple[Node, Node]:
    s = _as_tree(parse_newick(src))
    d = _as_tree(parse_newick(dst))
    d.initialize_split_indices(s.taxa_encoding)
    return s, d


def test_blocked_order_extremes_single_mover_left_right():
    """
    Single mover block should be placed at the far LEFT in t1 and
    far RIGHT in t2 (i=0 mover with negative/positive extreme weights).
    """
    t1, t2 = _pair(
        "(A:1,B:1,C:1,D:1,E:1);",
        "(A:1,B:1,C:1,D:1,E:1);",
    )

    # Edge = full set
    encoding = t1.taxa_encoding
    edge = Partition(tuple(sorted(encoding.values())), encoding)

    # Mover block = (C,D)
    mover = Partition((encoding["C"], encoding["D"]), encoding)
    sources = {mover: mover}
    destinations = {mover: mover}

    blocked_order_and_apply(edge, sources, destinations, t1, t2)

    order1 = list(t1.get_current_order())
    order2 = list(t2.get_current_order())

    # i=0 mover placed at left of t1 and right of t2
    assert order1[:2] == ["C", "D"] or order1[:2] == ["D", "C"]
    assert order2[-2:] == ["C", "D"] or order2[-2:] == ["D", "C"]


def test_blocked_order_extremes_two_movers_same_side():
    """
    All movers go to the same side to minimize anchor displacement:
    - All movers: left in t1 (band 0), right in t2 (band 2)
    - Larger groups (with smaller expand paths) are placed more extreme
    """
    t1, t2 = _pair(
        "(A:1,B:1,C:1,D:1,E:1,F:1);",
        "(A:1,B:1,C:1,D:1,E:1,F:1);",
    )

    enc = t1.taxa_encoding
    edge = Partition(tuple(sorted(enc.values())), enc)

    mover1 = Partition((enc["B"],), enc)  # smaller (1 taxon) -> i=1 after sort
    mover2 = Partition((enc["E"], enc["F"]), enc)  # larger (2 taxa) -> i=0 after sort
    sources = {mover1: mover1, mover2: mover2}
    destinations = {mover1: mover1, mover2: mover2}

    blocked_order_and_apply(edge, sources, destinations, t1, t2)

    o1 = list(t1.get_current_order())
    o2 = list(t2.get_current_order())

    # All movers go to left in t1, right in t2
    # Larger mover (E,F) is more extreme (leftmost in t1, rightmost in t2)
    anchor_taxa = {"A", "C", "D"}

    # In t1: movers are split because of alternation
    # i=0 (E,F) -> src_band=0 (left)
    # i=1 (B)   -> src_band=2 (right)

    # So E,F should be at the start (left)
    assert set(o1[:2]) == {"E", "F"}
    # B should be at the end (right)
    assert o1[-1] == "B"

    # In t2:
    # i=0 (E,F) -> dst_band=2 (right)
    # i=1 (B)   -> dst_band=0 (left)

    # So B should be at the start (left)
    assert o2[0] == "B"
    # E,F should be at the end (right)
    assert set(o2[-2:]) == {"E", "F"}

    # Anchors stay in the middle
    # In t1: E,F (left) ... Anchors ... B (right)
    assert set(o1[2:-1]) == anchor_taxa

    # In t2: B (left) ... Anchors ... E,F (right)
    assert set(o2[1:-2]) == anchor_taxa


def test_derive_order_for_pair_no_differences_preserves_visual_order():
    """
    When there are no differing edges between trees, derive_order_for_pair must
    not rewrite either tree's existing visual order.
    """
    t1, t2 = _pair("((A:1,B:1):1,(C:1,D:1):1);", "((C:1,D:1):1,(A:1,B:1):1);")

    derive_order_for_pair(t1, t2, anchor_weight_policy="destination")

    assert list(t1.get_current_order()) == ["A", "B", "C", "D"]
    assert list(t2.get_current_order()) == ["C", "D", "A", "B"]


def test_precomputed_common_splits_does_not_promote_leaf_anchors():
    t1, t2 = _pair("(A:1,B:1,C:1,D:1);", "(D:1,C:1,B:1,A:1);")
    common_splits = get_common_splits(t1, t2)

    derive_order_for_pair(
        t1,
        t2,
        anchor_weight_policy="destination",
        common_splits=common_splits,
    )

    assert list(t1.get_current_order()) == ["A", "B", "C", "D"]
    assert list(t2.get_current_order()) == ["D", "C", "B", "A"]


def test_root_alignment_keeps_unhandled_taxa_outside_anchor_block():
    """
    Root-level anchor alignment should keep explicit stable anchor blocks stable.
    Taxa outside anchor/mover blocks are not movers, so they behave like
    singleton anchors under the same anchor policy.
    """
    t1, t2 = _pair("((A:1,B:1):1,C:1,D:1);", "((A:1,B:1):1,D:1,C:1);")
    common_splits = get_common_splits(t1, t2)
    encoding = t1.taxa_encoding
    root_edge = Partition(tuple(sorted(encoding.values())), encoding)

    blocked_order_and_apply(
        root_edge,
        {},
        {},
        t1,
        t2,
        anchor_weight_policy="destination",
        common_splits=common_splits,
    )

    assert list(t1.get_current_order()) == ["A", "B", "D", "C"]
    assert list(t2.get_current_order()) == ["A", "B", "D", "C"]
