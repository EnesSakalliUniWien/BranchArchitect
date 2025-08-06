#!/usr/bin/env python3
"""Test the fixes we implemented for Node class"""

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import Partition


def test_fixes():
    print("=== TESTING FIXES ===")

    # Test 1: split_indices sharing (should be fixed now)
    print("1. Testing split_indices sharing fix:")
    node1 = Node()
    node2 = Node()
    sharing_fixed = not (node1.split_indices is node2.split_indices)
    print(
        f"   node1.split_indices is node2.split_indices: {node1.split_indices is node2.split_indices}"
    )
    print(f"   Expected: False - {'✅ FIXED!' if sharing_fixed else '❌ STILL BROKEN'}")

    # Test 2: Parent pointers (should be set now)
    print("\n2. Testing parent pointer fix:")
    child1 = Node()
    child2 = Node()
    parent_with_children = Node(children=[child1, child2])
    parent_fix_works = (
        child1.parent is parent_with_children and child2.parent is parent_with_children
    )
    print(
        f"   child1.parent is parent_with_children: {child1.parent is parent_with_children}"
    )
    print(
        f"   child2.parent is parent_with_children: {child2.parent is parent_with_children}"
    )
    print(
        f"   Expected: True, True - {'✅ FIXED!' if parent_fix_works else '❌ STILL BROKEN'}"
    )

    # Test 3: Basic functionality still works
    print("\n3. Testing basic functionality:")
    node = Node(name="test", length=0.5)
    print(f"   node.name: {node.name}")
    print(f"   node.length: {node.length}")
    print(f"   node.split_indices: {node.split_indices}")
    functionality_works = node.name == "test" and node.length == 0.5
    print(
        f"   Expected: name=test, length=0.5 - {'✅ WORKING!' if functionality_works else '❌ BROKEN'}"
    )

    # Test 4: No more dataclass confusion
    print("\n4. Testing dataclass removal:")
    has_dataclass_fields = hasattr(Node, "__dataclass_fields__")
    print(f"   Has dataclass fields: {has_dataclass_fields}")
    print(
        f"   Expected: False - {'✅ FIXED!' if not has_dataclass_fields else '❌ STILL HAS DATACLASS'}"
    )

    print(f"\n{'=' * 50}")
    if (
        sharing_fixed
        and parent_fix_works
        and functionality_works
        and not has_dataclass_fields
    ):
        print("🎉 ALL FIXES WORKING CORRECTLY!")
    else:
        print("⚠️  Some fixes may need more work")
    print("=" * 50)


if __name__ == "__main__":
    test_fixes()
