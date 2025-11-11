"""
Policy tests for anchor_order:
- Anchor weighting policy: destination vs preserve_source
- Mover weight policy: increasing vs decreasing
- Error behavior when pivot edge is missing
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.anchor_order import blocked_order_and_apply
import pytest


def _edge_full(tree):
    enc = tree.taxa_encoding
    return Partition(tuple(sorted(enc.values())), enc)


def test_anchor_weight_policy_destination_vs_preserve_source():
    """
    With destination policy, anchors in t1 can reorder to match t2 anchor order.
    With preserve_source, anchors in t1 keep their original relative source order.
    """
    # Source and destination orders differ; C is the only mover
    t1 = parse_newick("(A:1,B:1,C:1,D:1,E:1);")
    t2 = parse_newick("(E:1,D:1,C:1,B:1,A:1);", list(t1.get_current_order()))
    enc = t1.taxa_encoding
    edge = _edge_full(t1)

    mover_C = Partition((enc["C"],), enc)
    sources = {mover_C: mover_C}
    destinations = {mover_C: mover_C}

    # Destination-guided anchors: anchors in t1 follow t2 among themselves
    t1_dest = t1.deep_copy(); t2_dest = t2.deep_copy()
    # Force destination pivot order to reflect desired anchor guidance (E,D,C,B,A)
    dst_node = t2_dest.find_node_by_split(edge)
    assert dst_node is not None
    dst_node.reorder_taxa(["E", "D", "C", "B", "A"])  # Set explicit destination pivot order
    blocked_order_and_apply(
        edge,
        sources,
        destinations,
        t1_dest,
        t2_dest,
        mover_weight_policy="decreasing",
        anchor_weight_policy="destination",
    )
    order_dest = list(t1_dest.get_current_order())
    # Expect C at one extreme; anchors in destination order excluding C: E, D, B, A
    assert order_dest[0] == "C" or order_dest[-1] == "C"
    anchors_dest = [x for x in order_dest if x != "C"]
    assert anchors_dest == ["E", "D", "B", "A"]

    # Preserve source anchors: anchors retain source order excluding C: A, B, D, E
    t1_src = t1.deep_copy(); t2_src = t2.deep_copy()
    # Ensure destination has the same explicit pivot order for a fair comparison
    dst_node2 = t2_src.find_node_by_split(edge)
    assert dst_node2 is not None
    dst_node2.reorder_taxa(["E", "D", "C", "B", "A"])  # Same guidance
    blocked_order_and_apply(
        edge,
        sources,
        destinations,
        t1_src,
        t2_src,
        mover_weight_policy="decreasing",
        anchor_weight_policy="preserve_source",
    )
    order_src = list(t1_src.get_current_order())
    assert order_src[0] == "C" or order_src[-1] == "C"
    anchors_src = [x for x in order_src if x != "C"]
    assert anchors_src == ["A", "B", "D", "E"]


def test_mover_weight_policy_increasing_vs_decreasing_affects_leftmost_priority():
    """
    With multiple movers where two land on the same side (negative weights),
    'decreasing' policy makes the earliest (smallest) mover most extreme.
    'increasing' policy makes the latest (largest) mover most extreme.
    """
    t1 = parse_newick("(A:1,B:1,C:1,D:1,E:1,F:1,G:1);")
    t2 = parse_newick("(A:1,B:1,C:1,D:1,E:1,F:1,G:1);", list(t1.get_current_order()))
    enc = t1.taxa_encoding
    edge = _edge_full(t1)

    mover_B = Partition((enc["B"],), enc)  # size 1
    mover_G = Partition((enc["G"],), enc)  # size 1 (parity -> right side)
    mover_EF = Partition((enc["E"], enc["F"]), enc)  # size 2
    movers = {mover_B: mover_B, mover_G: mover_G, mover_EF: mover_EF}

    # Increasing: EF (i=2, negative, larger magnitude) goes further left than B (i=0)
    t1_inc = t1.deep_copy(); t2_inc = t2.deep_copy()
    blocked_order_and_apply(
        edge,
        movers,
        movers,
        t1_inc,
        t2_inc,
        mover_weight_policy="increasing",
        anchor_weight_policy="destination",
    )
    order_inc = list(t1_inc.get_current_order())
    # Leftmost two should be EF block then B under 'increasing'
    left_two_inc = order_inc[:2]
    assert set(left_two_inc).issubset({"E", "F", "B"})
    # EF block must be contiguous at the very left
    assert set(order_inc[:2]) == {"E", "F"}

    # Decreasing: B (i=0, negative, largest magnitude) goes further left than EF (i=2)
    t1_dec = t1.deep_copy(); t2_dec = t2.deep_copy()
    blocked_order_and_apply(
        edge,
        movers,
        movers,
        t1_dec,
        t2_dec,
        mover_weight_policy="decreasing",
        anchor_weight_policy="destination",
    )
    order_dec = list(t1_dec.get_current_order())
    # Leftmost should be B under 'decreasing'
    assert order_dec[0] == "B"
    # EF block should still be contiguous immediately after B on the left
    assert set(order_dec[1:3]) == {"E", "F"}


def test_blocked_order_and_apply_raises_on_missing_pivot():
    """
    If the provided pivot split does not exist in one or both trees, the
    function must raise ValueError (strict policy).
    """
    t1 = parse_newick("(A:1,B:1,C:1);")
    t2 = parse_newick("(C:1,A:1,B:1);", list(t1.get_current_order()))
    enc = t1.taxa_encoding

    # Non-existent internal split in a 3-leaf star tree: (A,C) does not exist
    missing = Partition((enc["A"], enc["C"]), enc)
    mover_C = Partition((enc["C"],), enc)

    with pytest.raises(ValueError):
        blocked_order_and_apply(
            missing,
            {mover_C: mover_C},
            {mover_C: mover_C},
            t1,
            t2,
        )
