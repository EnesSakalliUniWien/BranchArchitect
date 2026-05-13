from brancharchitect.tree import Node


def test_deep_copy_builds_root_split_index_with_copied_nodes(monkeypatch):
    encoding = {"A": 0, "B": 1}
    tree = Node(
        children=[
            Node(name="A", taxa_encoding=encoding, split_indices=(0,)),
            Node(name="B", taxa_encoding=encoding, split_indices=(1,)),
        ],
        taxa_encoding=encoding,
        split_indices=(0, 1),
    )

    copied = tree.deep_copy()
    copied_leaf_split = copied.children[0].split_indices

    def fail_rebuild(self):
        raise AssertionError("deep_copy should already populate the root split index")

    monkeypatch.setattr(Node, "build_split_index", fail_rebuild)

    found = copied.find_node_by_split(copied_leaf_split)
    assert found is copied.children[0]
    assert found is not tree.children[0]


def test_deep_copy_empty_values_are_not_shared_between_copied_nodes():
    encoding = {"A": 0, "B": 1}
    tree = Node(
        children=[
            Node(name="A", taxa_encoding=encoding, split_indices=(0,)),
            Node(name="B", taxa_encoding=encoding, split_indices=(1,)),
        ],
        taxa_encoding=encoding,
        split_indices=(0, 1),
    )

    copied = tree.deep_copy()
    first_child, second_child = copied.children

    first_child.values["label"] = "changed"

    assert second_child.values == {}
    assert first_child.values is not second_child.values


def test_deep_copy_does_not_dispatch_per_node_shallow_copy(monkeypatch):
    """deep_copy should keep the hot node-copy loop local."""
    encoding = {"A": 0, "B": 1}
    tree = Node(
        children=[
            Node(name="A", taxa_encoding=encoding, split_indices=(0,)),
            Node(name="B", taxa_encoding=encoding, split_indices=(1,)),
        ],
        taxa_encoding=encoding,
        split_indices=(0, 1),
    )

    def fail_shallow_copy(self):
        raise AssertionError("deep_copy should not dispatch per-node helper calls")

    monkeypatch.setattr(
        Node, "_create_shallow_node_copy", fail_shallow_copy, raising=False
    )

    copied = tree.deep_copy()

    assert copied is not tree
    assert copied.children[0] is not tree.children[0]
    assert copied.to_newick() == tree.to_newick()


def test_deep_copy_can_defer_split_index_until_lookup():
    encoding = {"A": 0, "B": 1}
    tree = Node(
        children=[
            Node(name="A", taxa_encoding=encoding, split_indices=(0,)),
            Node(name="B", taxa_encoding=encoding, split_indices=(1,)),
        ],
        taxa_encoding=encoding,
        split_indices=(0, 1),
    )

    copied = tree.deep_copy(build_split_index=False)

    assert copied._split_index is None
    assert copied.find_node_by_split(copied.children[0].split_indices) is copied.children[0]
    assert copied._split_index is not None
