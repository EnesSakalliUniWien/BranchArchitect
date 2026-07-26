#!/usr/bin/env python3
"""
Test script to verify Partition JSON serialization works correctly.
This simulates the issue described in the conversation summary.
"""

import json

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.io import UUIDEncoder, dump_json
from brancharchitect.parser.newick_parser import parse_newick


def test_partition_direct_serialization():
    """Test direct Partition serialization with JSON."""
    # Create a Partition object
    partition = Partition((0, 1, 2), {"A": 0, "B": 1, "C": 2})

    # Standard json can't serialize a Partition without the custom encoder.
    try:
        json.dumps(partition)
    except TypeError:
        pass
    else:
        raise AssertionError("Standard JSON serialization should raise TypeError for Partition")

    # UUIDEncoder should handle it.
    result = json.dumps(partition, cls=UUIDEncoder)
    assert result


def test_node_with_partition_split_indices():
    """Test Node serialization when split_indices is a Partition object."""
    # Create a Node with Partition split_indices
    partition = Partition((0, 1), {"A": 0, "B": 1})
    node = Node(name="TestNode", split_indices=partition)

    node_dict = node.to_dict()
    assert "split_indices" in node_dict

    result = json.dumps(node_dict)
    assert result


def test_tree_with_partition_split_indices():
    """Test tree parsing and serialization with Partition objects."""
    newick = "((A,B),C);"
    tree = parse_newick(newick)

    tree_dict = tree.to_dict()
    assert json.dumps(tree_dict)
    assert json.dumps(tree_dict, cls=UUIDEncoder)


def test_dump_json_function(tmp_path):
    """Test the dump_json function from io.py"""
    newick = "((A,B),C);"
    tree = parse_newick(newick)
    tree_dict = tree.to_dict()

    temp_path = tmp_path / "tree.json"
    with open(temp_path, "w") as f:
        dump_json(tree_dict, f)

    content = temp_path.read_text()
    assert content


def test_create_problematic_scenario():
    """Create a scenario that would trigger the original error."""
    newick = "(((A,B),(C,D)),((E,F),(G,H)));"
    tree = parse_newick(newick)

    # Manually set some split_indices to Partition objects (simulating the issue)
    # This might happen during tree processing/interpolation
    for node in tree.traverse():
        if isinstance(node.split_indices, tuple):
            partition = Partition(
                node.split_indices, tree._order if hasattr(tree, "_order") else {}
            )
            node.split_indices = partition

    # This should not raise - serialization must succeed on a tree containing
    # Partition split_indices.
    tree_dict = tree.to_dict()
    assert json.dumps(tree_dict, cls=UUIDEncoder)


if __name__ == "__main__":
    test_partition_direct_serialization()
    test_node_with_partition_split_indices()
    test_tree_with_partition_split_indices()
    test_create_problematic_scenario()
    print("All tests completed")
