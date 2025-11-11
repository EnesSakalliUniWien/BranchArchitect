"""
Tests for tree ordering API in brancharchitect/tree.py

Focus: Node.reorder_taxa (MINIMUM strategy default), subtree application,
error handling, and topology preservation.
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition


def _root_edge(tree):
    enc = tree.taxa_encoding
    return Partition(tuple(sorted(enc.values())), enc)


def test_reorder_taxa_minimum_strategy_matches_permutation_when_possible():
    """
    For a simple balanced tree, MINIMUM strategy can realize a full permutation
    by reordering children at each internal node.
    """
    t = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
    root = t
    target = ["D", "C", "B", "A"]

    root.reorder_taxa(target)
    assert list(t.get_current_order()) == target


def test_reorder_taxa_raises_on_wrong_permutation_set():
    t = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
    root = t
    # Missing D
    bad = ["C", "B", "A"]
    try:
        root.reorder_taxa(bad)
        assert False, "Expected ValueError for missing taxa in permutation"
    except ValueError:
        pass


def test_subtree_reordering_keeps_outside_unchanged():
    """
    Reordering a subtree should not reorder siblings outside that subtree.
    """
    t = parse_newick("((A:1,B:1):1,(C:1,(D:1,E:1):1):1);")
    # Subtree = (C,(D,E))
    enc = t.taxa_encoding
    subtree_edge = Partition((enc["C"], enc["D"], enc["E"]), enc)
    node = t.find_node_by_split(subtree_edge)
    assert node is not None

    original_left = list(t.children[0].get_current_order())
    node.reorder_taxa(["E", "D", "C"])  # reverse inside subtree

    # Left sibling (A,B) remains A,B
    assert list(t.children[0].get_current_order()) == original_left == ["A", "B"]
    # Entire tree order respects subtree change
    assert list(t.get_current_order()) == ["A", "B", "E", "D", "C"]


def test_reorder_preserves_topology_splits():
    """
    Reordering the taxa changes only embedding (child order), not the splits.
    """
    t = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
    before = t.to_splits()
    t.reorder_taxa(["B", "A", "D", "C"])
    after = t.to_splits()
    assert before == after


def test_reorder_tie_breaker_determinism_on_equal_minima():
    """
    When two children have equal MINIMUM leaf index, tie-breaker uses
    sorted leaf-name tuple for deterministic ordering.
    """
    t = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
    # Both subtrees get the same minimum index (A and C both at position 0)
    perm = ["A", "C", "B", "D"]
    t.reorder_taxa(perm)
    # Deterministic order: (A,B) comes before (C,D) due to tie-breaker
    assert list(t.get_current_order()) == ["A", "B", "C", "D"]

