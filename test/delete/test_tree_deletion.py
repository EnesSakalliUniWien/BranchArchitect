import pytest
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition

# Helper: build a simple test tree
#      A
#     / \
#    B   C
#   / \   \
#  D   E   F
# Leaves: D, E, F


def build_test_tree():
    encoding = {"D": 0, "E": 1, "F": 2}
    leaf_d = Node(children=[], name="D", split_indices=(0,), taxa_encoding=encoding)
    leaf_e = Node(children=[], name="E", split_indices=(1,), taxa_encoding=encoding)
    leaf_f = Node(children=[], name="F", split_indices=(2,), taxa_encoding=encoding)
    node_b = Node(
        children=[leaf_d, leaf_e],
        name="B",
        split_indices=(0, 1),
        taxa_encoding=encoding,
    )
    node_c = Node(
        children=[leaf_f], name="C", split_indices=(2,), taxa_encoding=encoding
    )
    root = Node(
        children=[node_b, node_c],
        name="A",
        split_indices=(0, 1, 2),
        taxa_encoding=encoding,
    )
    return root


def test_delete_taxa_removes_leaf_and_updates_indices():
    tree = build_test_tree()
    # Delete leaf E (index 1)
    tree.delete_taxa([1])
    leaves = [leaf.name for leaf in tree.leaves]
    assert set(leaves) == {"D", "F"}
    # All leaves should have names
    assert all(leaf for leaf in leaves)
    # Split indices should not contain 1
    for node in tree.traverse():
        if hasattr(node, "split_indices"):
            assert 1 not in node.split_indices


def test_delete_taxa_removes_multiple_leaves():
    tree = build_test_tree()
    # Delete D and F (indices 0 and 2)
    tree.delete_taxa([0, 2])
    leaves = [leaf.name for leaf in tree.leaves]
    assert leaves == ["E"]
    # All leaves should have names
    assert all(leaf for leaf in leaves)
    # Split indices should only contain 1
    for node in tree.traverse():
        if hasattr(node, "split_indices"):
            assert set(node.split_indices).issubset({1})


def test_no_unnamed_leaves_after_deletion():
    tree = build_test_tree()
    tree.delete_taxa([0])  # Remove D
    for leaf in tree.leaves:
        assert leaf.name != ""


def test_split_indices_consistency():
    tree = build_test_tree()
    tree.delete_taxa([2])  # Remove F
    # After deletion, check that split indices match the leaves
    leaf_names = {leaf.name for leaf in tree.leaves}
    for node in tree.traverse():
        for idx in node.split_indices:
            # Each index should correspond to a leaf name in encoding
            name = [k for k, v in node.taxa_encoding.items() if v == idx][0]
            assert name in leaf_names


def test_tree_structure_after_deletion():
    tree = build_test_tree()
    tree.delete_taxa([0, 1])  # Remove D and E
    # Only F should remain
    assert len(tree.leaves) == 1
    assert tree.leaves[0].name == "F"
    # After proper phylogenetic tree pruning:
    # - Node B becomes empty (D and E deleted) and gets removed
    # - Node C has only one child (F) and gets collapsed
    # - F gets connected directly to root A
    # Root should have only one child (F) - no intermediate nodes with single children
    assert len(tree.children) == 1
    assert tree.children[0].name == "F"
