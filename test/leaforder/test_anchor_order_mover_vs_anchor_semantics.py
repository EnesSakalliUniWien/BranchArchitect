"""
Semantics test: the solution (mapping key) should be treated as the mover,
and the mapped partition (mapping value) should act as anchor guidance.

This expresses the intended logic even if current implementation differs.
Marked xfail until the implementation is aligned.
"""

import pytest
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.anchor_order import blocked_order_and_apply


def _pivot_edge_full(tree):
    enc = tree.taxa_encoding
    return Partition(tuple(sorted(enc.values())), enc)


@pytest.mark.xfail(reason="Current implementation treats mapped values as movers, not solution keys")
def test_solution_keys_are_movers_block_moves_as_one_extreme():
    """
    Given mapping (solution -> mapped), the SOLUTION (key) is the mover.
    If solution is a block (A,B), both A and B should move together to the
    same extreme. The mapped side provides anchoring guidance only.
    """
    # Source (A,B,C); Destination (C,A,B)
    t1 = parse_newick("(A:1,B:1,C:1);")
    t2 = parse_newick("(C:1,A:1,B:1);", list(t1.get_current_order()))

    enc = t1.taxa_encoding
    edge = _pivot_edge_full(t1)

    # Fabricate mapping: solution key is block (A,B), mapped anchors differ per tree
    sol_AB = Partition((enc["A"], enc["B"]), enc)
    mapped_A = Partition((enc["A"],), enc)
    mapped_B = Partition((enc["B"],), enc)

    sources = {sol_AB: mapped_A}
    destinations = {sol_AB: mapped_B}

    blocked_order_and_apply(edge, sources, destinations, t1, t2)

    order1 = list(t1.get_current_order())
    # Expect A and B together at one extreme if solution block is moved as one
    assert order1[:2] == ["A", "B"] or order1[:2] == ["B", "A"] or \
           order1[-2:] == ["A", "B"] or order1[-2:] == ["B", "A"]

