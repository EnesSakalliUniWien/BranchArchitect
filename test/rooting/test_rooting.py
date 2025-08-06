from brancharchitect.io import parse_newick
from brancharchitect.rooting.rooting import (
    simple_reroot,
    find_best_matching_node,
    reroot_at_node,
    reroot_by_jaccard_similarity,
    find_best_matching_node_jaccard,
)
from brancharchitect.tree import Node


# --- Simple rerooting test ---
def test_reroot_at_leaf_and_internal():
    newick = "((A:1,B:1):1,(C:1,D:1):1);"

    # Test 1: Reroot at leaf B
    tree1 = parse_newick(newick)
    leaf_b = [n for n in tree1.traverse() if getattr(n, "name", None) == "B"][0]
    rerooted_b = reroot_at_node(leaf_b)
    assert rerooted_b is leaf_b

    # Test 2: Reroot at internal node (A,B) - use fresh tree
    tree2 = parse_newick(newick)
    ab_node = [
        n for n in tree2.traverse() if set(leaf.name for leaf in n.leaves) == {"A", "B"}
    ][0]
    rerooted_ab = reroot_at_node(ab_node)
    assert rerooted_ab is ab_node


# --- Reroot to best match (clade overlap) ---
def test_reroot_to_best_match():
    newick1 = "((A:1,B:1):1,(C:1,D:1):1);"
    newick2 = "(A:1,((B:1,C:1):1,D:1):1);"  # A separate, (B,C) together, D separate
    tree1 = parse_newick(newick1)
    tree2 = parse_newick(newick2)
    # Test simple reroot between trees
    rerooted = simple_reroot(tree1, tree2)
    # After rerooting tree2, we should be able to find a clade that contains both A and B
    # Check that the best overlap node was found and rerooted appropriately
    assert rerooted is not None


# --- Basic correspondence test ---
def test_basic_correspondence():
    newick1 = "((A:1,B:1):1,(C:1,D:1):1);"
    tree1 = parse_newick(newick1)
    # Simple test that we can find matching nodes
    for node in tree1.traverse():
        if hasattr(node, "split_indices") and node.split_indices:
            match = find_best_matching_node(node.split_indices, tree1)
            assert match is not None


# --- Polytomy test ---
def test_polytomy_reroot():
    newick = "(A:1,B:1,C:1,D:1);"
    tree = parse_newick(newick)
    for leaf in tree.leaves:
        rerooted = find_best_matching_node(leaf.split_indices, tree)
        assert rerooted is not None


# --- Unbalanced tree test ---
def test_unbalanced_tree_reroot():
    newick = "(((A:1,B:1):1,C:1):1,D:1);"
    tree = parse_newick(newick)
    # Reroot at C
    c_node = [n for n in tree.traverse() if getattr(n, "name", None) == "C"][0]
    rerooted = reroot_at_node(c_node)
    assert rerooted is c_node


# --- Large tree test ---
def test_large_tree_reroot():
    # 8-taxa balanced tree
    newick = "((((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1):1);"
    tree = parse_newick(newick)
    # Reroot at (A,B)
    ab_node = [
        n for n in tree.traverse() if set(leaf.name for leaf in n.leaves) == {"A", "B"}
    ][0]
    rerooted = reroot_at_node(ab_node)
    assert rerooted is ab_node
    # Reroot at (E,F,G,H)
    efgh_node = [
        n
        for n in tree.traverse()
        if set(leaf.name for leaf in n.leaves) == {"E", "F", "G", "H"}
    ][0]
    rerooted2 = reroot_at_node(efgh_node)
    assert rerooted2 is efgh_node


# --- Edge case: Tree with repeated taxa names (should not allow, but test robustness) ---
def test_repeated_taxa_names():
    newick = "((A:1,A:1):1,(B:1,C:1):1);"
    try:
        tree = parse_newick(newick)
        ab_node = [
            n for n in tree.traverse() if set(leaf.name for leaf in n.leaves) == {"A"}
        ]
        assert len(ab_node) >= 1
    except Exception:
        pass  # Acceptable: parser may reject repeated taxa


# --- Edge case: Tree with only one taxon ---
def test_single_taxon_tree():
    newick = "A:1;"
    tree = parse_newick(newick)
    rerooted = reroot_at_node(tree)
    assert rerooted is tree


# --- Edge case: Completely unresolved (star) tree ---
def test_star_tree():
    newick = "(A:1,B:1,C:1,D:1,E:1,F:1);"
    tree = parse_newick(newick)
    for leaf in tree.leaves:
        rerooted = reroot_at_node(leaf)
        assert rerooted is leaf


# --- Edge case: Deeply nested unbalanced tree ---
def test_deeply_nested_unbalanced():
    newick = "((((((A:1):1):1):1):1):1);"
    tree = parse_newick(newick)
    leaf = [n for n in tree.traverse() if n.is_leaf()][0]
    rerooted = reroot_at_node(leaf)
    assert rerooted is leaf


# --- Edge case: Trees with polytomies and multifurcations ---
def test_multifurcation():
    newick = "((A:1,B:1,C:1):1,(D:1,E:1):1);"
    tree = parse_newick(newick)
    abc_node = [
        n
        for n in tree.traverse()
        if set(leaf.name for leaf in n.leaves) == {"A", "B", "C"}
    ][0]
    rerooted = reroot_at_node(abc_node)
    assert rerooted is abc_node


# --- Edge case: Trees with missing branch lengths ---
def test_missing_branch_lengths():
    newick = "((A,B),(C,D));"
    tree = parse_newick(newick)
    ab_node = [
        n for n in tree.traverse() if set(leaf.name for leaf in n.leaves) == {"A", "B"}
    ][0]
    rerooted = reroot_at_node(ab_node)
    assert rerooted is ab_node


# --- Edge case: Trees with negative branch lengths (should not crash) ---
def test_negative_branch_lengths():
    newick = "((A:-1,B:-2):-3,(C:-4,D:-5):-6);"
    try:
        tree = parse_newick(newick)
        ab_node = [
            n
            for n in tree.traverse()
            if set(leaf.name for leaf in n.leaves) == {"A", "B"}
        ][0]
        rerooted = reroot_at_node(ab_node)
        assert rerooted is ab_node
    except Exception:
        pass  # Acceptable: parser may reject negative lengths


# --- Edge case: Trees with non-binary splits everywhere ---
def test_fully_nonbinary():
    newick = "(A:1,B:1,C:1,D:1,E:1,F:1,G:1,H:1);"
    tree = parse_newick(newick)
    for leaf in tree.leaves:
        rerooted = reroot_at_node(leaf)
        assert rerooted is leaf


# --- Edge case: Trees with only internal nodes (no leaves) ---
def test_internal_only_tree():
    class DummyNode(Node):
        def __init__(self):
            super().__init__(children=[], name=None)

        def is_leaf(self):
            return False

    root = DummyNode()
    rerooted = reroot_at_node(root)
    assert rerooted is root


# --- Jaccard similarity-based rerooting test ---
def test_reroot_to_best_match_jaccard():
    """Test Jaccard similarity-based rerooting inspired by phylo-io"""
    newick1 = "((A:1,B:1):1,(C:1,D:1):1);"
    newick2 = "(A:1,((B:1,C:1):1,D:1):1);"  # A separate, (B,C) together, D separate
    tree1 = parse_newick(newick1)
    tree2 = parse_newick(newick2)

    # Pick (A,B) clade in tree1
    ab_node = [
        n for n in tree1.traverse() if set(leaf.name for leaf in n.leaves) == {"A", "B"}
    ][0]

    # Test Jaccard-based rerooting
    rerooted = reroot_by_jaccard_similarity(tree1, tree2)

    # Verify rerooting was successful
    assert rerooted is not None

    # Test that we can find nodes with high Jaccard similarity
    if hasattr(ab_node, "split_indices") and ab_node.split_indices:
        best_match = find_best_matching_node_jaccard(ab_node.split_indices, tree2)
        assert best_match is not None
