"""
Tests for anchor-based leaf ordering (anchor_order.py).

We validate that:
- Movers are pushed to extremes with alternating ping-pong directions.
- Stable anchors are ordered deterministically and blocks are respected.
- Root-level alignment runs when there are no differing edges.
"""

from typing import Tuple
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.anchor_order import (
    blocked_order_and_apply,
    derive_order_for_pair,
)


def _pair(src: str, dst: str):
    s = parse_newick(src)
    d = parse_newick(dst)
    d.taxa_encoding = s.taxa_encoding
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


def test_blocked_order_extremes_two_movers_ping_pong():
    """
    Two movers alternate: first goes left in t1/right in t2; second goes right in t1/left in t2.
    """
    t1, t2 = _pair(
        "(A:1,B:1,C:1,D:1,E:1,F:1);",
        "(A:1,B:1,C:1,D:1,E:1,F:1);",
    )

    enc = t1.taxa_encoding
    edge = Partition(tuple(sorted(enc.values())), enc)

    mover1 = Partition((enc["B"],), enc)  # i=0
    mover2 = Partition((enc["E"], enc["F"]), enc)  # i=1
    sources = {mover1: mover1, mover2: mover2}
    destinations = {mover1: mover1, mover2: mover2}

    blocked_order_and_apply(edge, sources, destinations, t1, t2)

    o1 = list(t1.get_current_order())
    o2 = list(t2.get_current_order())

    # mover1 left in t1, right in t2
    assert o1[0] == "B"
    assert o2[-1] == "B"

    # mover2 right in t1, left in t2 (block EF contiguous, order can be EF or FE)
    assert set(o1[-2:]) == {"E", "F"}
    assert set(o2[:2]) == {"E", "F"}


def test_derive_order_for_pair_no_differences_root_alignment():
    """
    When there are no differing edges between trees, derive_order_for_pair
    still applies ordering at the root without errors.
    """
    t1, t2 = _pair("(A:1,B:1,C:1);", "(A:1,B:1,C:1);")
    # Should not raise and should leave order unchanged
    derive_order_for_pair(t1, t2)
    assert list(t1.get_current_order()) == ["A", "B", "C"]
    assert list(t2.get_current_order()) == ["A", "B", "C"]
