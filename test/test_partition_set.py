import pytest
from brancharchitect.partition_set import Partition, PartitionSet


def test_partition_set_intersection():
    """Test that PartitionSet intersection works correctly and preserves lookup dictionaries."""
    # Create a lookup dictionary for taxa names to indices
    lookup = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

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
    """Test that discard raises an exception when element is not in the set."""
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

    # Test discard of a non-existing element (should raise ValueError)
    with pytest.raises(ValueError, match="not found in set"):
        test_set.discard(p3)

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

    TREES_ORDER = ['C2', 'Y', 'X', 'C1', 'B1', 'B2', 'O']
    trees = parse_newick(
        "(O,(((C2,Y),((B2,C1),B1)),X));"
        "(O,(((C2,Y),((B2,C1),B1)),X));"
        "(O,(((C1,X),((B2,C2),B1)),Y));"
        "(O,(((C1,X),((B2,C2),B1)),Y));",
        order=TREES_ORDER,
    )
    # Deepcopy to simulate independent trees but with shared encoding
    trees_for_opt = [deepcopy(t) for t in trees]
    # Print encoding id for all trees (should be the same)
    encoding_ids = [id(getattr(t, 'encoding', None)) for t in trees_for_opt]
    print("Encoding ids for all trees:", encoding_ids)
    for i, t in enumerate(trees_for_opt):
        print(f"Tree {i} encoding: {getattr(t, 'encoding', None)}")
    # Print all split encodings for each tree
    for i, t in enumerate(trees_for_opt):
        if hasattr(t, 'get_splits'):
            splits = t.get_splits() if callable(t.get_splits) else []
            print(f"Tree {i} splits encodings:", [getattr(s, 'encoding', None) for s in splits])
    # Check encoding id for all trees (should be the same)
    assert len(set(encoding_ids)) == 1, f"Trees do not share the same encoding! ids: {encoding_ids}"
    # Run the optimizer
    optimizer = TreeOrderOptimizer(trees_for_opt)
    optimizer.optimize(n_iterations=2, bidirectional=True)
    # Check split rotation and propagation
    any_rotated = any(
        v["improved"] and len(v["splits_rotated"]) > 0
        for v in optimizer.split_rotation_history.values()
    )
    assert any_rotated, "No splits were rotated or propagated! Check encoding and split matching."
    # Optionally, print details for manual inspection
    for k, v in optimizer.split_rotation_history.items():
        print(f"Tree pair {k}: improved={v['improved']}, splits_rotated={len(v['splits_rotated'])}")