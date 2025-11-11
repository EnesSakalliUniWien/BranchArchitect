"""Tests for circular rotation policies in anchor_order."""

from __future__ import annotations

from brancharchitect.leaforder.anchor_order import blocked_order_and_apply
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition


def _pair(src: str, dst: str):
    # Helper to build two identical trees quickly
    from brancharchitect.parser.newick_parser import parse_newick

    trees = parse_newick(src + dst)
    t1: Node = trees[0]
    t2: Node = trees[1]
    return t1, t2


def _edge_full(t: Node) -> Partition:
    return Partition(tuple(sorted(t.taxa_encoding.values())), t.taxa_encoding)


def test_circular_largest_mover_at_zero():
    t1, t2 = _pair(
        "(A:1,B:1,C:1,D:1,E:1,F:1,G:1);",
        "(A:1,B:1,C:1,D:1,E:1,F:1,G:1);",
    )
    enc = t1.taxa_encoding
    edge = _edge_full(t1)

    mover_B = Partition((enc["B"],), enc)
    mover_G = Partition((enc["G"],), enc)
    mover_EF = Partition((enc["E"], enc["F"]), enc)
    movers = {mover_B: mover_B, mover_G: mover_G, mover_EF: mover_EF}

    blocked_order_and_apply(
        edge,
        movers,
        movers,
        t1,
        t2,
        mover_weight_policy="increasing",
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="largest_mover_at_zero",
    )

    order1 = list(t1.get_current_order())
    order2 = list(t2.get_current_order())

    # Largest mover is EF; index 0 should be either E or F in both trees
    assert order1[0] in {"E", "F"}
    assert order2[0] in {"E", "F"}


def test_circular_between_anchor_blocks_boundary_is_on_anchors():
    t1, t2 = _pair(
        "(A:1,B:1,C:1,D:1,E:1,F:1,G:1);",
        "(A:1,B:1,C:1,D:1,E:1,F:1,G:1);",
    )
    enc = t1.taxa_encoding
    edge = _edge_full(t1)

    mover_B = Partition((enc["B"],), enc)
    mover_EF = Partition((enc["E"], enc["F"]), enc)
    movers = {mover_B: mover_B, mover_EF: mover_EF}

    blocked_order_and_apply(
        edge,
        movers,
        movers,
        t1,
        t2,
        mover_weight_policy="increasing",
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="between_anchor_blocks",
    )

    order1 = list(t1.get_current_order())
    order2 = list(t2.get_current_order())
    mover_taxa = {"B", "E", "F"}

    # Boundary is between order[-1] and order[0]; both should be anchors (not movers)
    assert order1[-1] not in mover_taxa and order1[0] not in mover_taxa
    assert order2[-1] not in mover_taxa and order2[0] not in mover_taxa

