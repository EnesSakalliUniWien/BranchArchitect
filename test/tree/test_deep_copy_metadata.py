from brancharchitect.tree import Node


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
