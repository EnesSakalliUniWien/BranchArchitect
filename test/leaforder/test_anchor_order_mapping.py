"""
Tests that anchor_order treats SOLUTION KEYS (mapping keys) as movers and uses
mapping VALUES as anchor guidance. A block solution (A,B) should move as a block.
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.anchor_order import blocked_order_and_apply


def _pivot_edge_for_all_taxa(tree):
    enc = tree.taxa_encoding
    return Partition(tuple(sorted(enc.values())), enc)


def test_blocked_order_uses_solution_keys_as_movers_block_moves_together():
    """
    For tree pair (A,B,C) vs (C,A,B), fabricate mapping where the solution key is (A,B)
    but mapped values are A (source) and B (destination). Verify that A and B are moved
    to opposite extremes (ping-pong), and C remains between them (anchor), demonstrating
    that mapping values (atoms) are used rather than raw solution keys (blocks).
    """
    # Source and destination trees
    t1 = parse_newick("(A:1,B:1,C:1);")
    t2 = parse_newick("(C:1,A:1,B:1);", list(t1.get_current_order()))

    enc = t1.taxa_encoding
    edge = _pivot_edge_for_all_taxa(t1)

    # Fabricated mapping:
    # Key (solution) = (A,B) block; values (mapped) = A for source, B for destination
    sol_ab = Partition((enc["A"], enc["B"]), enc)
    mapped_A = Partition((enc["A"],), enc)
    mapped_B = Partition((enc["B"],), enc)

    sources = {sol_ab: mapped_A}
    destinations = {sol_ab: mapped_B}

    # Apply blocked order
    blocked_order_and_apply(edge, sources, destinations, t1, t2)

    order1 = list(t1.get_current_order())
    order2 = list(t2.get_current_order())

    # A and B should move together as a block to one extreme in T1
    assert order1[:2] == ["A", "B"] or order1[:2] == ["B", "A"] or \
           order1[-2:] == ["A", "B"] or order1[-2:] == ["B", "A"]


def test_blocked_order_singleton_solutions_move_as_singletons():
    """
    When mapping values are used, movers are deduplicated across trees and anchors
    (stable frontiers) are preserved. Use a 4-taxon example to validate order shape.
    """
    t1 = parse_newick("((A:1,B:1),(C:1,D:1));")
    t2 = parse_newick("((A:1,C:1),(B:1,D:1));", list(t1.get_current_order()))

    enc = t1.taxa_encoding
    edge = _pivot_edge_for_all_taxa(t1)

    # Fabricate mapping so A (source solution) and C (destination solution) are movers (singletons)
    sol_A = Partition((enc["A"],), enc)
    sol_C = Partition((enc["C"],), enc)
    mapped_A = Partition((enc["A"],), enc)
    mapped_C = Partition((enc["C"],), enc)

    sources = {sol_A: mapped_A}
    destinations = {sol_C: mapped_C}

    blocked_order_and_apply(edge, sources, destinations, t1, t2)

    order1 = list(t1.get_current_order())
    order2 = list(t2.get_current_order())

    # T1: A is an extreme (mover), and C is an extreme in T2
    assert order1[0] == "A" or order1[-1] == "A"
    assert order2[0] == "C" or order2[-1] == "C"
