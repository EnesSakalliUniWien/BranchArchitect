from brancharchitect.newick_parser import Node, parse_newick

from typing import Optional

# def name_unnamed_nodes(node: Node):
#     for child in node.children:
#         name_unnamed_nodes(child)
#     if node.name == "":
#         node.name = NodeName("".join([child.name for child in node.children]))


### Deletion Algorithm
def delete_taxa(root: Node, indices_to_delete: list[int]) -> Node:
    r = _delete_taxa(root, indices_to_delete)
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


def _delete_taxa(node: Node, indices_to_delete: list[int]) -> Node:
    node.children = [child for child in node.children if any(idx not in indices_to_delete for idx in child.split_indices)]
    node.split_indices = tuple([idx for idx in node.split_indices if idx not in indices_to_delete])

    for child in node.children:
        _delete_taxa(child, indices_to_delete)
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
