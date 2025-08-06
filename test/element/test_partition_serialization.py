#!/usr/bin/env python3
"""
Test script to verify Partition JSON serialization works correctly.
This simulates the issue described in the conversation summary.
"""

print("Starting JSON serialization test...")

import json

print("Imported json")

from brancharchitect.elements.partition import Partition

print("Imported Partition")

from brancharchitect.tree import Node

print("Imported Node")

from brancharchitect.io import UUIDEncoder, dump_json

print("Imported UUIDEncoder and dump_json")

from brancharchitect.parser.newick_parser import parse_newick

print("Imported parse_newick")


def test_partition_direct_serialization():
    """Test direct Partition serialization with JSON."""
    print("=== Testing Direct Partition Serialization ===")

    # Create a Partition object
    partition = Partition((0, 1, 2), {"A": 0, "B": 1, "C": 2})
    print(f"Partition: {partition}")
    print(f"Partition indices: {partition.indices}")

    # Try to serialize with standard json
    try:
        result = json.dumps(partition)
        print(f"Standard JSON serialization failed - this should not happen")
    except TypeError as e:
        print(f"Expected TypeError with standard JSON: {e}")

    # Try to serialize with UUIDEncoder
    try:
        result = json.dumps(partition, cls=UUIDEncoder)
        print(f"UUIDEncoder serialization successful: {result}")
    except Exception as e:
        print(f"UUIDEncoder serialization failed: {e}")


def test_node_with_partition_split_indices():
    """Test Node serialization when split_indices is a Partition object."""
    print("\n=== Testing Node with Partition split_indices ===")

    # Create a Node with Partition split_indices
    partition = Partition((0, 1), {"A": 0, "B": 1})
    node = Node(name="TestNode", split_indices=partition)
    print(f"Node split_indices type: {type(node.split_indices)}")
    print(f"Node split_indices: {node.split_indices}")

    # Test to_dict() method
    try:
        node_dict = node.to_dict()
        print(f"Node.to_dict() successful")
        print(
            f"split_indices in dict: {node_dict['split_indices']} (type: {type(node_dict['split_indices'])})"
        )
    except Exception as e:
        print(f"Node.to_dict() failed: {e}")

    # Test JSON serialization of node dict
    try:
        result = json.dumps(node_dict)
        print(f"JSON serialization of node dict successful: {result}")
    except Exception as e:
        print(f"JSON serialization of node dict failed: {e}")


def test_tree_with_partition_split_indices():
    """Test tree parsing and serialization with Partition objects."""
    print("\n=== Testing Tree with Partition Objects ===")

    # Parse a simple tree
    newick = "((A,B),C);"
    tree = parse_newick(newick)

    # Check the types of split_indices in the tree
    def check_split_indices_types(node, level=0):
        indent = "  " * level
        print(
            f"{indent}Node '{node.name}': split_indices type = {type(node.split_indices)}, value = {node.split_indices}"
        )
        for child in node.children:
            check_split_indices_types(child, level + 1)

    print("Tree structure and split_indices types:")
    check_split_indices_types(tree)

    # Test serialization
    try:
        tree_dict = tree.to_dict()
        print(f"Tree.to_dict() successful")

        # Try to serialize the tree dict
        result = json.dumps(tree_dict)
        print(f"JSON serialization successful")

        # Try with UUIDEncoder
        result_uuid = json.dumps(tree_dict, cls=UUIDEncoder)
        print(f"UUIDEncoder serialization successful")

    except Exception as e:
        print(f"Tree serialization failed: {e}")
        import traceback

        traceback.print_exc()


def test_dump_json_function():
    """Test the dump_json function from io.py"""
    print("\n=== Testing dump_json Function ===")

    # Create a tree with mixed types
    newick = "((A,B),C);"
    tree = parse_newick(newick)
    tree_dict = tree.to_dict()

    # Test dump_json with file
    import tempfile
    import os

    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            dump_json(tree_dict, f)
            temp_path = f.name

        # Read back and verify
        with open(temp_path, "r") as f:
            content = f.read()
            print(f"dump_json successful, file content length: {len(content)}")

        # Clean up
        os.unlink(temp_path)

    except Exception as e:
        print(f"dump_json failed: {e}")
        import traceback

        traceback.print_exc()


def test_create_problematic_scenario():
    """Create a scenario that would trigger the original error."""
    print("\n=== Testing Problematic Scenario ===")

    # Create a complex tree structure
    newick = "(((A,B),(C,D)),((E,F),(G,H)));"
    tree = parse_newick(newick)

    # Manually set some split_indices to Partition objects (simulating the issue)
    # This might happen during tree processing/interpolation
    for node in tree.traverse():
        if isinstance(node.split_indices, tuple):
            # Convert tuple to Partition object
            partition = Partition(
                node.split_indices, tree._order if hasattr(tree, "_order") else {}
            )
            node.split_indices = partition
            print(
                f"Converted node '{node.name}' split_indices to Partition: {partition}"
            )

    # Now try to serialize this tree - this should trigger the original error if not fixed
    try:
        tree_dict = tree.to_dict()
        result = json.dumps(tree_dict, cls=UUIDEncoder)
        print(f"Complex tree serialization successful!")
    except Exception as e:
        print(f"Complex tree serialization failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_partition_direct_serialization()
    test_node_with_partition_split_indices()
    test_tree_with_partition_split_indices()
    test_dump_json_function()
    test_create_problematic_scenario()
    print("\n=== All tests completed ===")
