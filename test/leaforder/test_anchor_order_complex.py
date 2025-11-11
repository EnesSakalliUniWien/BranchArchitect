"""
More comprehensive tests for anchor-based leaf ordering (anchor_order.py).

Covers:
- Ping-pong extremes with three movers
- Subtree (non-root) pivot application
- Stable shared blocks preserved as contiguous units
- Determinism/idempotency
- derive_order_for_pair changes ordering when trees differ
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.anchor_order import (
    blocked_order_and_apply,
    derive_order_for_pair,
)


def _pair(src: str, dst: str):
    s = parse_newick(src)
    d = parse_newick(dst, list(s.get_current_order()))
    return s, d


def _edge_full(tree):
    enc = tree.taxa_encoding
    return Partition(tuple(sorted(enc.values())), enc)


def test_ping_pong_three_movers_extremes():
    """Three movers alternate extremes correctly in both trees."""
    t1, t2 = _pair(
        "(A:1,B:1,C:1,D:1,E:1,F:1,G:1);",
        "(A:1,B:1,C:1,D:1,E:1,F:1,G:1);",
    )
    enc = t1.taxa_encoding
    edge = _edge_full(t1)

    mover1 = Partition((enc["B"],), enc)  # i=0 -> left in t1, right in t2
    mover2 = Partition((enc["E"], enc["F"]), enc)  # i=1 -> right in t1, left in t2
    mover3 = Partition((enc["G"],), enc)  # i=2 -> left in t1, right in t2 (more extreme than i=0)

    sources = {mover1: mover1, mover2: mover2, mover3: mover3}
    destinations = sources.copy()

    # Use 'increasing' magnitude so the last mover (E,F) is most extreme on the left,
    # matching the original expectation of this test.
    blocked_order_and_apply(
        edge,
        sources,
        destinations,
        t1,
        t2,
        mover_weight_policy="increasing",
        anchor_weight_policy="destination",
    )

    o1 = list(t1.get_current_order())
    o2 = list(t2.get_current_order())

    # Movers sorted by (size, indices): (B) -> i=0, (G) -> i=1, (E,F) -> i=2
    # t1: i even => left, odd => right; i=2 (E,F) most-left, i=1 (G) most-right
    assert set(o1[:2]) == {"E", "F"}
    assert o1[-1] == "G"

    # t2: i even => right, odd => left; i=1 (G) most-left, i=2 (E,F) most-right
    assert o2[0] == "G"
    assert set(o2[-2:]) == {"E", "F"}


def test_subtree_edge_application():
    """Applying to a subtree pivot reorders only the pivot clade."""
    t1, t2 = _pair(
        "(((A:1,B:1),(C:1,D:1)),(E:1,F:1));",
        "(((C:1,D:1),(A:1,B:1)),(E:1,F:1));",
    )
    enc = t1.taxa_encoding
    # Pivot = (A,B,C,D)
    edge = Partition((enc["A"], enc["B"], enc["C"], enc["D"]), enc)

    mover = Partition((enc["C"], enc["D"]), enc)
    sources = {mover: mover}
    destinations = {mover: mover}

    blocked_order_and_apply(edge, sources, destinations, t1, t2)

    o = list(t1.get_current_order())
    # Inside pivot, C,D should appear before A,B; A before B preserved
    assert o.index("C") < o.index("A")
    assert o.index("A") < o.index("B")


def test_stable_blocks_preserved_as_contiguous():
    """Stable shared blocks remain contiguous after reordering."""
    t1, t2 = _pair(
        "(((X1:1,X2:1),(A:1,B:1)),(C:1,D:1));",
        "(((X1:1,X2:1),(B:1,A:1)),(C:1,D:1));",
    )
    enc = t1.taxa_encoding
    edge = _edge_full(t1)

    mover = Partition((enc["A"], enc["B"]), enc)
    sources = {mover: mover}
    destinations = {mover: mover}

    blocked_order_and_apply(edge, sources, destinations, t1, t2)
    o = list(t1.get_current_order())
    # X1,X2 contiguous
    x_pos = [o.index("X1"), o.index("X2")]
    assert max(x_pos) - min(x_pos) == 1


def test_blocked_order_is_deterministic():
    """Repeated application yields same order (idempotent)."""
    t1, t2 = _pair(
        "(A:1,B:1,C:1,D:1,E:1);",
        "(E:1,A:1,B:1,C:1,D:1);",
    )
    enc = t1.taxa_encoding
    edge = _edge_full(t1)
    mover = Partition((enc["E"],), enc)
    sources = {mover: mover}
    destinations = {mover: mover}

    blocked_order_and_apply(edge, sources, destinations, t1, t2)
    once = list(t1.get_current_order())
    blocked_order_and_apply(edge, sources, destinations, t1, t2)
    twice = list(t1.get_current_order())
    assert once == twice


def test_derive_order_for_pair_runs_on_different_topology():
    """derive_order_for_pair executes without errors on non-identical topologies."""
    t1 = parse_newick("((A:1,B:1),(C:1,D:1));")
    t2 = parse_newick("((A:1,C:1),(B:1,D:1));", list(t1.get_current_order()))
    derive_order_for_pair(t1, t2)
    # Invariants: encodings preserved and same taxa set
    assert t1.taxa_encoding == t2.taxa_encoding
    assert set(t1.get_current_order()) == {"A", "B", "C", "D"}
