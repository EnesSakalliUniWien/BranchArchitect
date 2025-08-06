#!/usr/bin/env python3

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import Partition


def test_split_indices_sharing():
    """Test the critical split_indices sharing issue."""

    print("=" * 60)
    print("CRITICAL ISSUE: SPLIT_INDICES SHARING")
    print("=" * 60)

    # Create multiple nodes with default parameters
    node1 = Node()
    node2 = Node()
    node3 = Node()

    print("\n1. Testing split_indices object identity:")
    print(f"node1.split_indices: {node1.split_indices}")
    print(f"node2.split_indices: {node2.split_indices}")
    print(f"node3.split_indices: {node3.split_indices}")
    print(f"node1 is node2: {node1.split_indices is node2.split_indices}")
    print(f"node2 is node3: {node2.split_indices is node3.split_indices}")
    print(f"node1 is node3: {node1.split_indices is node3.split_indices}")

    print(
        f"\nAll split_indices point to same object: {node1.split_indices is node2.split_indices is node3.split_indices}"
    )

    if node1.split_indices is node2.split_indices:
        print("❌ CONFIRMED: All nodes share the same split_indices object!")

        print("\n2. Testing if modification affects other instances:")

        # The Partition class might be immutable, so let's test
        print(f"Original split_indices: {node1.split_indices}")
        print(f"Type: {type(node1.split_indices)}")

        # Try to see what happens when we assign new split_indices
        original_partition = node1.split_indices

        # Create a new partition for node1
        new_partition = Partition((0, 1), {"A": 0, "B": 1})
        node1.split_indices = new_partition

        print(f"\nAfter assigning new partition to node1:")
        print(f"node1.split_indices: {node1.split_indices}")
        print(f"node2.split_indices: {node2.split_indices}")
        print(f"node3.split_indices: {node3.split_indices}")

        print(
            f"node1 is node2 after change: {node1.split_indices is node2.split_indices}"
        )

        # Check if the original shared partition is still shared between node2 and node3
        print(
            f"node2 is node3 after change: {node2.split_indices is node3.split_indices}"
        )
        print(
            f"original partition is node2: {original_partition is node2.split_indices}"
        )

        print("\n3. Testing what happens with the default Partition:")

        # Check if the default partition has any mutable state
        default_partition = Partition((), {})
        another_default = Partition((), {})

        print(f"Two empty partitions are equal: {default_partition == another_default}")
        print(
            f"Two empty partitions are same object: {default_partition is another_default}"
        )

    else:
        print("✅ split_indices are properly isolated")

    print("\n4. Testing the root cause in __init__:")
    print("The issue is in the default parameter:")
    print("split_indices: Partition = Partition((), {})")
    print("")
    print("This creates ONE Partition object at function definition time,")
    print("and ALL instances that don't provide split_indices get the SAME object!")

    print("\n5. Testing with explicit parameters:")
    # Test nodes created with explicit parameters
    node4 = Node(split_indices=Partition((), {}))
    node5 = Node(split_indices=Partition((), {}))

    print(
        f"Explicitly created partitions same object: {node4.split_indices is node5.split_indices}"
    )

    if node4.split_indices is node5.split_indices:
        print(
            "❌ Even explicit creation shares objects (if same Partition instance passed)"
        )
    else:
        print("✅ Explicitly created partitions are different objects")

    print("\n6. Potential impact:")
    print("- If Partition objects are mutable, this could cause data corruption")
    print("- Even if immutable, this wastes memory and could cause confusion")
    print("- Any modifications to the shared partition affect ALL nodes using defaults")

    print("\n" + "=" * 60)
    print("RECOMMENDATION: Fix the mutable default argument")
    print("Change: split_indices: Partition = Partition((), {})")
    print("To:     split_indices: Optional[Partition] = None")
    print("And in __init__:")
    print(
        "self.split_indices = split_indices if split_indices is not None else Partition((), {})"
    )
    print("=" * 60)


if __name__ == "__main__":
    test_split_indices_sharing()
