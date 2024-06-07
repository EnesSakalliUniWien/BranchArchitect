from typing import Optional
from brancharchitect.newick_parser import Node, parse_newick


# def name_unnamed_nodes(node: Node):
#     for child in node.children:
#         name_unnamed_nodes(child)
#     if node.name == "":
#         node.name = NodeName("".join([child.name for child in node.children]))


### Deletion Algorithm
def delete_taxa(root: Node, nodes_to_delete: list[Node]) -> Node:
    r = _delete_taxa(root, nodes_to_delete, None)
    r = _delete_superfluous_nodes(r)
    return r


def _get_end_child(node: Node):
    if len(node.children) != 1:
        return node
    else:
        return _get_end_child(node.children[0])


def _delete_superfluous_nodes(node: Node):
    node.children = [_get_end_child(child) for child in node.children]
    for child in node.children:
        _delete_superfluous_nodes(child)
    return node


def _delete_taxa(
    node: Node, nodes_to_delete: list[Node], parent: Optional[Node]
) -> Node:
    is_leaf = len(node.children) == 0
    node = delete_children(node, nodes_to_delete)
    if not is_leaf and len(node.children) == 0 and parent:
        parent.children.remove(node)
    for child in node.children:
        _delete_taxa(child, nodes_to_delete, node)
    return node


def delete_children(node, children):
    node.children = [child for child in node.children if child.name not in children]
    return node


def name_unnamed_nodes_by_indices(node: Node):
    for child in node.children:
        name_unnamed_nodes(child)

    if node.name == "":
        print([child.name for child in node.children])
        node.name = [child.name for child in node.children]


def serialize_to_newick(node: Node):
    return _serialize_to_newick(node) + ";"


def _serialize_to_newick(node: Node):
    if node.children:
        return (
            "("
            + ",".join([_serialize_to_newick(child) for child in node.children])
            + ")"
            + node.name
        )
    else:
        return node.name


def get_external_indices(node: Node, leave_indices):
    for child in node.children:
        if child.children:
            get_external_indices(child, leave_indices)
        else:
            leave_indices.append(child.name)
    return leave_indices


### Parser Tests
def get_child(node, *path):
    for i in path:
        if isinstance(node, Node):
            children = node.children
        else:
            children = node['children']
        assert len(children) >= i
        node = children[i]
    return node


### Test Deletion Algorithm
def test_del_1():
    s = "((A,(B,C),((D,E),((F,G),H))),I);"
    root = parse_newick(s)
    root = delete_taxa(root, ["B"])

    assert get_child(root, 0, 0).name == "A"
    assert get_child(root, 0, 1).name == "C"
    assert get_child(root, 0, 2, 0, 0).name == "D"
    assert get_child(root, 0, 2, 0, 1).name == "E"
    assert get_child(root, 0, 2, 1, 0, 0).name == "F"
    assert get_child(root, 0, 2, 1, 0, 1).name == "G"
    assert get_child(root, 0, 2, 1, 1).name == "H"
    assert get_child(root, 1).name == "I"
