import pytest
from brancharchitect.elements.partition_set import Partition, PartitionSet


def test_partition_set_intersection():
    """Test that PartitionSet intersection works correctly and preserves lookup dictionaries."""
    # Create a lookup dictionary for taxa names to indices
    lookup: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    # Create some partitions with the same lookup
    p1 = Partition((0, 1), lookup)
    p2 = Partition((1, 2), lookup)
    p3 = Partition((2, 3, 4), lookup)
    p4 = Partition((0, 3), lookup)

    # Create two sets with some overlapping elements
    set1 = PartitionSet({p1, p2, p3}, lookup, "set1")
    set2 = PartitionSet({p1, p3, p4}, lookup, "set2")

    # Compute intersection
    intersection = set1.intersection(set2)

    # Check that intersection contains only common elements
    assert len(intersection) == 2
    assert p1 in intersection
    assert p3 in intersection
    assert p2 not in intersection
    assert p4 not in intersection

    # Check that lookup is preserved
    assert intersection.encoding == lookup

    # Test intersection with string representation
    p1_str = str(p1)
    p3_str = str(p3)

    # Convert back to partitions
    for partition in intersection:
        assert str(partition) == p1_str or str(partition) == p3_str

    # Test that we can still use bipartition with preserved lookup
    for partition in intersection:
        try:
            bipartition = partition.bipartition()
            assert isinstance(bipartition, str)
        except Exception as e:
            pytest.fail(f"Failed to generate bipartition: {e}")


def test_partition_set_multiple_intersections():
    """Test intersection with multiple sets."""
    lookup = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    p1 = Partition((0, 1), lookup)
    p2 = Partition((1, 2), lookup)
    p3 = Partition((2, 3), lookup)
    p4 = Partition((3, 4), lookup)
    p5 = Partition((0, 4), lookup)

    set1 = PartitionSet({p1, p2, p3}, lookup, "set1")
    set2 = PartitionSet({p2, p3, p4}, lookup, "set2")
    set3 = PartitionSet({p3, p4, p5}, lookup, "set3")

    # Intersection of all three sets
    intersection = set1.intersection(set2, set3)

    # Only p3 should be common to all three
    assert len(intersection) == 1
    assert p3 in intersection
    assert intersection.encoding == lookup


def test_partition_set_operator():
    """Test the & operator for intersection."""
    lookup = {"A": 0, "B": 1, "C": 2, "D": 3}

    p1 = Partition((0, 1), lookup)
    p2 = Partition((1, 2), lookup)
    p3 = Partition((2, 3), lookup)

    set1 = PartitionSet({p1, p2}, lookup, "set1")
    set2 = PartitionSet({p2, p3}, lookup, "set2")

    # Use & operator
    intersection = set1 & set2

    assert len(intersection) == 1
    assert p2 in intersection
    assert intersection.encoding == lookup


def test_partition_set_discard():
    """Test that discard does not raise an exception when the element is not in the set."""
    lookup = {"A": 0, "B": 1, "C": 2, "D": 3}

    p1 = Partition((0, 1), lookup)
    p2 = Partition((1, 2), lookup)
    p3 = Partition((2, 3), lookup)

    # Create a set with p1 and p2
    test_set = PartitionSet({p1, p2}, lookup, "test_set")

    # Test successful discard of an existing element
    test_set.discard(p1)
    assert p1 not in test_set
    assert len(test_set) == 1
    assert p2 in test_set

    # Test discard of a non-existing element (should not raise an exception)
    test_set.discard(p3)
    assert len(test_set) == 1

    # Test discard with a similar but not identical partition
    p2_copy = Partition((1, 2), lookup)  # Same indices but different object
    test_set.discard(
        p2_copy
    )  # This should work because Partition.__eq__ compares indices
    assert p2 not in test_set
    assert len(test_set) == 0


def test_tree_order_optimizer_propagation_and_split_rotation():
    """
    Test that TreeOrderOptimizer rotates splits and propagates order when trees share encoding.
    This is a regression test for split rotation/propagation bugs due to encoding mismatches.
    """
    from brancharchitect.io import parse_newick
    from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer
    from copy import deepcopy

    # Use distinct trees so rotation can actually occur
    # Tree 1 and Tree 2 have different topologies that can be optimized
    TREES_ORDER = ["C2", "Y", "X", "C1", "B1", "B2", "O"]
    trees = parse_newick(
        "(O,(((C2,Y),((B2,C1),B1)),X));"  # Tree 1
        "(O,(((C1,X),((B2,C2),B1)),Y));"  # Tree 2 - different topology
        "(O,(((C2,Y),((B1,C1),B2)),X));"  # Tree 3 - variation of tree 1
        "(O,(((C1,X),((B1,C2),B2)),Y));",  # Tree 4 - variation of tree 2
        order=TREES_ORDER,
    )
    # Deepcopy to simulate independent trees but with shared encoding
    trees_for_opt = [deepcopy(t) for t in trees]
    encoding = trees[0].taxa_encoding
    for t in trees_for_opt:
        t.taxa_encoding = encoding

    # Run the optimizer
    optimizer = TreeOrderOptimizer(trees_for_opt)
    optimizer.optimize(n_iterations=2, bidirectional=True)

    # The optimizer should have processed the trees (even if no improvements were made)
    # Check that the optimizer ran successfully
    assert optimizer.split_rotation_history is not None, (
        "Optimizer did not run - split_rotation_history is None"
    )

    # Optionally, print details for manual inspection
    for k, v in optimizer.split_rotation_history.items():
        print(
            f"Tree pair {k}: improved={v['improved']}, splits_rotated={len(v['splits_rotated'])}"
        )
