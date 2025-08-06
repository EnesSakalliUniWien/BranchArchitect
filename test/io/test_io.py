import os
from typing import Dict, List, LiteralString
from brancharchitect.io import serialize_tree_list_to_json, read_newick
from brancharchitect.io import write_json
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.plot.circular_tree import generate_multiple_circular_trees_svg
from brancharchitect.tree import Node
import json
import tempfile


def test_serialize_tree_list_to_json_with_five_taxa_permutations(tmp_path):
    # Use the five_taxa_all_permutations.newick file
    newick_path: LiteralString = os.path.join(
        "notebooks", "data", "five_taxa_all_permutations.newick"
    )
    trees: Node | List[Node] = read_newick(newick_path, force_list=True)
    data: List[Dict[str, Any]] = serialize_tree_list_to_json(trees)
    # Should be a list of dicts, one per tree
    assert isinstance(data, list)
    assert all(isinstance(tree, dict) for tree in data)

    # Check split_indices are lists of ints for all nodes in all trees
    def check_split_indices(node, path=None):
        if path is None:
            path = []
        assert isinstance(node["split_indices"], list), (
            f"split_indices at path {path} is {node['split_indices']} of type {type(node['split_indices'])} in node: {node}"
        )
        for idx, child in enumerate(node["children"]):
            check_split_indices(child, path + [idx])

    for tree in data:
        check_split_indices(tree)
    # Write to json and reload
    out_path = tmp_path / "five_taxa_trees.json"
    import json

    with open(out_path, "w") as f:
        json.dump(data, f)
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_serialize_tree_list_to_json_with_six_taxa_permutations(tmp_path):
    # Use the six_taxa_all_permutations.newick file
    newick_path = os.path.join("test", "six_taxa_all_permutations.newick")
    trees = read_newick(newick_path, force_list=True)
    data = serialize_tree_list_to_json(trees)
    assert isinstance(data, list)
    assert all(isinstance(tree, dict) for tree in data)

    def check_split_indices(node, path=None):
        if path is None:
            path = []
        assert isinstance(node["split_indices"], list), (
            f"split_indices at path {path} is {node['split_indices']} of type {type(node['split_indices'])} in node: {node}"
        )
        for idx, child in enumerate(node["children"]):
            check_split_indices(child, path + [idx])

    for tree in data:
        check_split_indices(tree)
    # Write to json and reload
    out_path = tmp_path / "six_taxa_trees.json"
    import json

    with open(out_path, "w") as f:
        json.dump(data, f)
    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_serialize_tree_list_to_json_basic():
    # Simple tree: (A,B);
    nA = Node(name="A")
    nB = Node(name="B")
    root = Node(children=[nA, nB])
    result = serialize_tree_list_to_json([root])
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert "children" in result[0]
    assert result[0]["children"][0]["name"] == "A"
    assert result[0]["children"][1]["name"] == "B"


def test_serialize_tree_list_to_json_multiple():
    nA = Node(name="A")
    nB = Node(name="B")
    nC = Node(name="C")
    root1 = Node(children=[nA, nB])
    root2 = Node(children=[nC])
    result = serialize_tree_list_to_json([root1, root2])
    assert len(result) == 2
    assert result[0]["children"][0]["name"] == "A"
    assert result[1]["children"][0]["name"] == "C"


def test_serialize_tree_list_to_json_json_dump(tmp_path):
    nA = Node(name="A")
    nB = Node(name="B")
    root = Node(children=[nA, nB])
    tree_list = [root]
    data = serialize_tree_list_to_json(tree_list)
    file_path = tmp_path / "trees.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    # Read back and check
    with open(file_path) as f:
        loaded = json.load(f)
    assert loaded[0]["children"][0]["name"] == "A"
    assert loaded[0]["children"][1]["name"] == "B"


def test_serialize_tree_list_to_json_with_split_indices():
    # Test that split_indices are lists of ints
    nA = Node(name="A")
    nB = Node(name="B")
    root = Node(children=[nA, nB])
    data = serialize_tree_list_to_json([root])
    assert isinstance(data[0]["split_indices"], list)
    for child in data[0]["children"]:
        assert isinstance(child["split_indices"], list)


### Parser Tests
def get_child(node, *path):
    for i in path:
        if isinstance(node, Node):
            children = node.children
        else:
            children = node["children"]
        assert len(children) >= i
        node = children[i]
    return node


def test_read_newick_write_json():
    newick = "((A[value=3],(B,C),((D,E),((F,G),H))),I);"

    with tempfile.NamedTemporaryFile(mode="w") as f_in:
        f_in.write(newick)
        f_in.flush()
        tree = read_newick(f_in.name)

    assert len(tree.children) == 2
    assert len(get_child(tree, 0).children) == 3

    assert get_child(tree, 0, 0).name == "A"
    assert len(get_child(tree, 0, 0).children) == 0

    assert get_child(tree, 0, 1, 0).name == "B"
    assert get_child(tree, 0, 1, 1).name == "C"
    assert get_child(tree, 0, 2, 0, 0).name == "D"
    assert get_child(tree, 0, 2, 0, 1).name == "E"
    assert get_child(tree, 0, 2, 1, 0, 0).name == "F"
    assert get_child(tree, 0, 2, 1, 0, 1).name == "G"
    assert get_child(tree, 0, 2, 1, 1).name == "H"
    assert get_child(tree, 1).name == "I"

    with tempfile.NamedTemporaryFile(mode="r") as f_out:
        write_json(tree, f_out.name)
        json_data = f_out.read()

    tree2 = json.loads(json_data)

    assert get_child(tree2, 0, 1, 0)["name"] == "B"
    assert get_child(tree2, 0, 1, 1)["name"] == "C"
    assert get_child(tree2, 0, 2, 0, 0)["name"] == "D"
    assert get_child(tree2, 0, 2, 0, 1)["name"] == "E"
    assert get_child(tree2, 0, 2, 1, 0, 0)["name"] == "F"
    assert get_child(tree2, 0, 2, 1, 0, 1)["name"] == "G"
    assert get_child(tree2, 0, 2, 1, 1)["name"] == "H"
    assert get_child(tree2, 1)["name"] == "I"


def test_generate_svg():
    newick = "((A[value=3],(B,C),((D,E),((F,G),H))),I);"
    tree = parse_newick(newick)
    svg = generate_multiple_circular_trees_svg([tree], 100)
