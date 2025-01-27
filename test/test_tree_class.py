# Assume Node, ReorderStrategy, and circular_distance are already defined from the given code.
from brancharchitect.tree import Node, ReorderStrategy


def create_linear_tree(names):
    """
    Create a simple linear tree: names = [A, B, C]
    A
     \
      B
       \
        C
    """
    nodes = [Node(name=n) for n in names]
    for i in range(len(nodes) - 1):
        nodes[i].append_child(nodes[i + 1])
    return nodes[0]


def create_star_tree(root_name, leaf_names):
    """
    Create a star-like tree:
        Root
       / | \
      L1 L2 L3 ...
    """
    root = Node(name=root_name)
    for ln in leaf_names:
        root.append_child(Node(name=ln))
    return root


def create_balanced_tree():
    """
    Create a balanced binary tree:
         A
        / \
       B   C
      / \ / \
     D  E F  G
    """
    A = Node(name="A")
    B = Node(name="B")
    C = Node(name="C")
    D = Node(name="D")
    E = Node(name="E")
    F = Node(name="F")
    G = Node(name="G")

    B.children = [D, E]
    C.children = [F, G]
    A.children = [B, C]
    return A


# 1. Test is_internal
def test_is_internal():
    leaf = Node(name="Leaf")
    # Fix the method:
    # def is_internal(self) -> bool:
    #     return bool(self.children)

    assert not leaf.is_internal(), "Leaf should not be internal"
    root = create_star_tree("Root", ["L1", "L2"])
    assert root.is_internal(), "Root with children should be internal"


# 2. Test append_child
def test_append_child():
    root = Node(name="Root")
    child = Node(name="Child")
    root.append_child(child)
    assert child in root.children, "Child should be appended to root children"
    assert len(root.children) == 1, "Root should have exactly one child"


# 3. Test deep_copy
def test_deep_copy():
    tree = create_linear_tree(["A", "B", "C"])
    copy_tree = tree.deep_copy()
    assert copy_tree is not tree, "deep_copy should create a distinct object"
    assert (
        copy_tree.get_current_order() == tree.get_current_order()
    ), "Copied tree should have same structure and leaves"


# 4. Test to_dict
def test_to_dict():
    tree = create_star_tree("Root", ["L1", "L2", "L3"])
    d = tree.to_dict()
    assert isinstance(d, dict), "to_dict should return a dictionary"
    assert "name" in d, "Dictionary should contain node attributes"
    assert d["name"] == "Root", "Name should match"


# 5. Test to_newick
def test_to_newick():
    tree = create_linear_tree(["A", "B", "C"])
    newick_str = tree.to_newick(lengths=False)
    assert newick_str.endswith(";"), "Newick string should end with semicolon"
    # Just a basic check, exact structure depends on implementation
    assert (
        "(" in newick_str and "A" in newick_str
    ), "Newick should represent tree structure"


# 6. Test _to_list
def test_to_list():
    tree = create_balanced_tree()
    lst = tree._to_list()

    """
    Create a balanced binary tree:
         A
        / \
       B   C
      / \ / \
     D  E F  G
    """

    # Expect ["A", ["B", ["C"]]] for linear structure
    assert lst == [["D", "E"], ["F", "G"]], "_to_list should reflect tree structure"


# 7. Test get_current_order
def test_get_current_order():
    tree = create_balanced_tree()
    order = tree.get_current_order()
    assert order == (
        "D",
        "E",
        "F",
        "G",
    ), "Order should match leaf traversal in a linear tree"


# 9. Test _initialize_split_indices
def test_initialize_split_indices():
    tree = create_balanced_tree()
    order = ["A", "B", "C", "D", "E", "F", "G"]
    tree._initialize_split_indices(order)
    # Check that each leaf got correct index
    leaves = tree.get_leaves()
    leaf_names = [leaf.name for leaf in leaves]
    for leaf in leaves:
        assert len(leaf.split_indices) == 1, "Each leaf should have exactly one index"
        assert leaf.name in leaf_names, "Leaf name should be recognized"


# 10. Test traverse
def test_traverse():
    tree = create_balanced_tree()
    all_nodes = list(tree.traverse())
    # Balanced tree has A,B,C,D,E,F,G total 7 nodes
    assert len(all_nodes) == 7, "Should traverse all nodes in the tree"
    node_names = {n.name for n in all_nodes}
    assert node_names == {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
    }, "All nodes should be traversed"


# 11. Test to_splits
def test_to_splits():
    tree = create_balanced_tree()
    order = ["A", "B", "C", "D", "E", "F", "G"]
    tree._initialize_split_indices(order)
    splits = tree.to_splits()
    assert isinstance(splits, set), "to_splits should return a set"
    # We know A,B,C are internal. So at least A,B,C node splits present
    assert len(splits) > 0, "Should have internal splits"


# 12. Test _index
def test_index_method():
    tree = create_balanced_tree()
    tree._order = ["A", "B", "C", "D", "E", "F", "G"]
    comp = ("A", "C")
    idx = tree._index(comp)
    # A=0, C=2 => idx=(0,2)
    assert idx == (0, 2), "_index should return sorted indices of given component"


# 13. Test get_leaves
def test_get_leaves():
    tree = create_balanced_tree()
    leaves = tree.get_leaves()
    leaf_names = [l.name for l in leaves]
    assert set(leaf_names) == {"D", "E", "F", "G"}, "Leaves should match expected set"


# 14. Test _fix_child_order
def test_fix_child_order():
    tree = create_balanced_tree()
    tree._order = ["A", "B", "C", "D", "E", "F", "G"]
    tree._initialize_split_indices(tree._order)
    # Swap children of A
    tree.children.reverse()
    tree._fix_child_order()
    # B should come before C as B's indices are smaller than C's
    assert [ch.name for ch in tree.children] == [
        "B",
        "C",
    ], "Child order should be fixed by indices"


# 17. Test to_hierarchy
def test_to_hierarchy():
    tree = create_balanced_tree()
    h = tree.to_hierarchy()
    assert isinstance(h, dict), "Hierarchy should be a dictionary"
    assert (
        "name" in h and "children" in h
    ), "Hierarchy should contain 'name' and 'children'"


# 18. Test swap_children
def test_swap_children():
    tree = create_star_tree("Root", ["L1", "L2", "L3"])
    original_order = [ch.name for ch in tree.children]
    tree.swap_children()
    new_order = [ch.name for ch in tree.children]
    # Just ensure order changed if >1 child
    if len(original_order) > 1:
        assert new_order != original_order, "Children order should be swapped"


# 19. Test to_weighted_splits
def test_to_weighted_splits():
    tree = create_balanced_tree()
    w_splits = tree.to_weighted_splits()
    # At least root's split is recorded
    assert isinstance(w_splits, dict), "Should return a dict"
    # Keys are tuples (split_indices), values can be None or float
    for k, v in w_splits.items():
        assert isinstance(k, tuple), "Keys should be tuples"


# 20. Test reorder_taxa
def test_reorder_taxa():
    # reorder_taxa requires same sets of leaves in permutation and tree
    tree = create_balanced_tree()
    # Leaves: D,E,F,G
    # Let's reorder to D,F,E,G according to MINIMUM strategy
    permutation = ["D", "E", "F", "G"]  # full set
    tree.reorder_taxa(permutation, strategy=ReorderStrategy.MINIMUM)
    # Just ensure no error and same set of leaves
    leaves = tree.get_current_order()
    assert set(leaves) == {"D", "E", "F", "G"}, "Taxa set unchanged"


# 20. Test reorder_taxa
def test_reorder_taxa_minimum():
    # reorder_taxa requires same sets of leaves in permutation and tree
    tree = create_balanced_tree()
    # Let's reorder to D,F,E,G according to MINIMUM strategy
    permutation = ["F", "G", "D", "E"]  # full set
    tree.reorder_taxa(permutation, strategy=ReorderStrategy.MINIMUM)
    # Just ensure no error and same set of leaves
    leaves = tree.get_current_order()
    assert set(leaves) == {"F", "G", "D", "E"}, "Taxa set unchanged"

    # 20. Test reorder_taxa


def test_reorder_taxa_minimum_2():
    # reorder_taxa requires same sets of leaves in permutation and tree
    tree = create_balanced_tree()
    # Let's reorder to D,F,E,G according to MINIMUM strategy
    permutation = ["F", "E", "D", "G"]  # full set
    tree.reorder_taxa(permutation, strategy=ReorderStrategy.MINIMUM)
    # Just ensure no error and same set of leaves
    leaves = tree.get_current_order()
    assert set(leaves) == {"F", "G", "E", "D"}, "Taxa set unchanged"
