import pytest
from brancharchitect.split import Partition, PartitionSet


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
    assert intersection.look_up == lookup
    
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
    assert intersection.look_up == lookup


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
    assert intersection.look_up == lookup
