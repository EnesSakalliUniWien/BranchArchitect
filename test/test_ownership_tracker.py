"""
Tests for the OwnershipTracker class.

This test suite verifies all core functionality including:
- Claim and release operations
- Ownership queries (unique, shared, last owner)
- Categorization (unique vs shared resources)
- Bulk operations
- Edge cases and atomic behavior
"""

import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.ownership_tracker import (
    OwnershipTracker,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def encoding():
    """Standard taxa encoding for tests."""
    return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


@pytest.fixture
def tracker(encoding):
    """Create a fresh OwnershipTracker instance."""
    return OwnershipTracker(encoding)


@pytest.fixture
def splits(encoding):
    """Create test splits."""
    return {
        "AB": Partition(frozenset([0, 1]), encoding),
        "CD": Partition(frozenset([2, 3]), encoding),
        "ABC": Partition(frozenset([0, 1, 2]), encoding),
        "DE": Partition(frozenset([3, 4]), encoding),
        "A": Partition(frozenset([0]), encoding),
    }


@pytest.fixture
def subtrees(encoding):
    """Create test subtrees."""
    return {
        "AB": Partition(frozenset([0, 1]), encoding),
        "CD": Partition(frozenset([2, 3]), encoding),
        "E": Partition(frozenset([4]), encoding),
    }


# ============================================================================
# Test: Basic Claim and Release
# ============================================================================


def test_initial_state_empty(tracker):
    """Test that tracker starts with no resources or owners."""
    assert len(tracker.get_all_resources()) == 0
    assert len(tracker.get_all_owners()) == 0
    stats = tracker.get_stats()
    assert stats["total_resources"] == 0
    assert stats["total_owners"] == 0


def test_claim_single_resource(tracker, splits, subtrees):
    """Test claiming a single resource."""
    tracker.claim(splits["AB"], subtrees["AB"])

    assert tracker.has_resource(splits["AB"])
    assert tracker.has_owner(subtrees["AB"])
    assert tracker.get_owner_count(splits["AB"]) == 1
    assert subtrees["AB"] in tracker.get_owners(splits["AB"])


def test_claim_idempotent(tracker, splits, subtrees):
    """Test that claiming the same resource multiple times is idempotent."""
    tracker.claim(splits["AB"], subtrees["AB"])
    tracker.claim(splits["AB"], subtrees["AB"])
    tracker.claim(splits["AB"], subtrees["AB"])

    assert tracker.get_owner_count(splits["AB"]) == 1
    assert len(tracker.get_resources(subtrees["AB"])) == 1


def test_release_single_resource(tracker, splits, subtrees):
    """Test releasing a resource."""
    tracker.claim(splits["AB"], subtrees["AB"])
    tracker.release(splits["AB"], subtrees["AB"])

    assert not tracker.has_resource(splits["AB"])
    assert not tracker.has_owner(subtrees["AB"])
    assert tracker.get_owner_count(splits["AB"]) == 0


def test_release_nonexistent_resource(tracker, splits, subtrees):
    """Test that releasing a non-existent resource doesn't crash."""
    # Should not raise any exception
    tracker.release(splits["AB"], subtrees["AB"])
    assert tracker.get_owner_count(splits["AB"]) == 0


# ============================================================================
# Test: Unique vs Shared Ownership
# ============================================================================


def test_unique_ownership(tracker, splits, subtrees):
    """Test unique ownership detection."""
    tracker.claim(splits["AB"], subtrees["AB"])

    assert tracker.is_unique(splits["AB"], subtrees["AB"])
    assert not tracker.is_shared(splits["AB"])
    assert tracker.get_owner_count(splits["AB"]) == 1


def test_shared_ownership(tracker, splits, subtrees):
    """Test shared ownership detection."""
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["CD"])

    assert tracker.is_shared(splits["ABC"])
    assert not tracker.is_unique(splits["ABC"], subtrees["AB"])
    assert not tracker.is_unique(splits["ABC"], subtrees["CD"])
    assert tracker.get_owner_count(splits["ABC"]) == 2


def test_transition_from_unique_to_shared(tracker, splits, subtrees):
    """Test that ownership correctly transitions from unique to shared."""
    # Start with unique
    tracker.claim(splits["ABC"], subtrees["AB"])
    assert tracker.is_unique(splits["ABC"], subtrees["AB"])
    assert not tracker.is_shared(splits["ABC"])

    # Add second owner -> becomes shared
    tracker.claim(splits["ABC"], subtrees["CD"])
    assert tracker.is_shared(splits["ABC"])
    assert not tracker.is_unique(splits["ABC"], subtrees["AB"])


def test_transition_from_shared_to_unique(tracker, splits, subtrees):
    """Test that ownership correctly transitions from shared to unique."""
    # Start with shared
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["CD"])
    assert tracker.is_shared(splits["ABC"])

    # Release one owner -> becomes unique
    tracker.release(splits["ABC"], subtrees["CD"])
    assert tracker.is_unique(splits["ABC"], subtrees["AB"])
    assert not tracker.is_shared(splits["ABC"])


# ============================================================================
# Test: Last Owner Detection
# ============================================================================


def test_is_last_owner_when_unique(tracker, splits, subtrees):
    """Test last owner detection for unique resources."""
    tracker.claim(splits["AB"], subtrees["AB"])
    assert tracker.is_last_owner(splits["AB"], subtrees["AB"])


def test_is_last_owner_when_shared(tracker, splits, subtrees):
    """Test last owner detection for shared resources."""
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["CD"])

    # Neither is last owner when shared
    assert not tracker.is_last_owner(splits["ABC"], subtrees["AB"])
    assert not tracker.is_last_owner(splits["ABC"], subtrees["CD"])


def test_is_last_owner_after_release(tracker, splits, subtrees):
    """Test last owner detection after releasing other owners."""
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["CD"])
    tracker.release(splits["ABC"], subtrees["CD"])

    # Now AB is the last owner
    assert tracker.is_last_owner(splits["ABC"], subtrees["AB"])


def test_is_last_owner_false_when_not_owner(tracker, splits, subtrees):
    """Test that is_last_owner returns False when not an owner."""
    tracker.claim(splits["AB"], subtrees["AB"])
    assert not tracker.is_last_owner(splits["AB"], subtrees["CD"])


# ============================================================================
# Test: Get Owners and Resources
# ============================================================================


def test_get_owners(tracker, splits, subtrees):
    """Test getting all owners of a resource."""
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["CD"])

    owners = tracker.get_owners(splits["ABC"])
    assert len(owners) == 2
    assert subtrees["AB"] in owners
    assert subtrees["CD"] in owners


def test_get_resources(tracker, splits, subtrees):
    """Test getting all resources owned by an owner."""
    tracker.claim(splits["AB"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["A"], subtrees["AB"])

    resources = tracker.get_resources(subtrees["AB"])
    assert len(resources) == 3
    assert splits["AB"] in resources
    assert splits["ABC"] in resources
    assert splits["A"] in resources


def test_get_owners_returns_immutable(tracker, splits, subtrees):
    """Test that get_owners returns immutable frozenset."""
    tracker.claim(splits["AB"], subtrees["AB"])
    owners = tracker.get_owners(splits["AB"])

    assert isinstance(owners, frozenset)
    # Should not be able to modify
    with pytest.raises(AttributeError):
        owners.add(subtrees["CD"])


def test_get_resources_returns_immutable(tracker, splits, subtrees):
    """Test that get_resources returns immutable frozenset."""
    tracker.claim(splits["AB"], subtrees["AB"])
    resources = tracker.get_resources(subtrees["AB"])

    assert isinstance(resources, frozenset)
    # Should not be able to modify
    with pytest.raises(AttributeError):
        resources.add(splits["CD"])


# ============================================================================
# Test: Categorization (Unique vs Shared)
# ============================================================================


def test_get_unique_resources(tracker, splits, subtrees):
    """Test getting unique resources for an owner."""
    tracker.claim(splits["AB"], subtrees["AB"])  # Unique
    tracker.claim(splits["ABC"], subtrees["AB"])  # Will be shared
    tracker.claim(splits["ABC"], subtrees["CD"])  # Shared

    unique = tracker.get_unique_resources(subtrees["AB"])
    assert len(unique) == 1
    assert splits["AB"] in unique
    assert splits["ABC"] not in unique


def test_get_shared_resources(tracker, splits, subtrees):
    """Test getting shared resources for an owner."""
    tracker.claim(splits["AB"], subtrees["AB"])  # Unique
    tracker.claim(splits["ABC"], subtrees["AB"])  # Shared
    tracker.claim(splits["ABC"], subtrees["CD"])  # Shared

    shared = tracker.get_shared_resources(subtrees["AB"])
    assert len(shared) == 1
    assert splits["ABC"] in shared
    assert splits["AB"] not in shared


def test_get_all_unique_resources(tracker, splits, subtrees):
    """Test getting all unique resources across all owners."""
    tracker.claim(splits["AB"], subtrees["AB"])  # Unique to AB
    tracker.claim(splits["CD"], subtrees["CD"])  # Unique to CD
    tracker.claim(splits["ABC"], subtrees["AB"])  # Shared
    tracker.claim(splits["ABC"], subtrees["CD"])  # Shared

    all_unique = tracker.get_all_unique_resources()
    assert len(all_unique) == 2
    assert all_unique[splits["AB"]] == subtrees["AB"]
    assert all_unique[splits["CD"]] == subtrees["CD"]
    assert splits["ABC"] not in all_unique


def test_get_all_shared_resources(tracker, splits, subtrees):
    """Test getting all shared resources across all owners."""
    tracker.claim(splits["AB"], subtrees["AB"])  # Unique
    tracker.claim(splits["ABC"], subtrees["AB"])  # Shared
    tracker.claim(splits["ABC"], subtrees["CD"])  # Shared
    tracker.claim(splits["DE"], subtrees["CD"])  # Shared
    tracker.claim(splits["DE"], subtrees["E"])  # Shared

    all_shared = tracker.get_all_shared_resources()
    assert len(all_shared) == 2
    assert splits["ABC"] in all_shared
    assert splits["DE"] in all_shared
    assert splits["AB"] not in all_shared

    # Check owners are correct
    assert len(all_shared[splits["ABC"]]) == 2
    assert subtrees["AB"] in all_shared[splits["ABC"]]
    assert subtrees["CD"] in all_shared[splits["ABC"]]


# ============================================================================
# Test: Global Release Operations
# ============================================================================


def test_release_all(tracker, splits, subtrees):
    """Test releasing a resource from all owners."""
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["CD"])
    tracker.claim(splits["ABC"], subtrees["E"])

    tracker.release_all(splits["ABC"])

    assert not tracker.has_resource(splits["ABC"])
    assert tracker.get_owner_count(splits["ABC"]) == 0
    # Owners should still exist if they have other resources
    assert len(tracker.get_resources(subtrees["AB"])) == 0


def test_release_all_removes_from_owner_indices(tracker, splits, subtrees):
    """Test that release_all removes resource from all owner indices."""
    tracker.claim(splits["AB"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["AB"])

    tracker.release_all(splits["ABC"])

    resources = tracker.get_resources(subtrees["AB"])
    assert splits["ABC"] not in resources
    assert splits["AB"] in resources  # Other resource still there


def test_release_owner_from_all_resources(tracker, splits, subtrees):
    """Test releasing an owner from all their resources."""
    tracker.claim(splits["AB"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["A"], subtrees["AB"])

    tracker.release_owner_from_all_resources(subtrees["AB"])

    assert not tracker.has_owner(subtrees["AB"])
    assert len(tracker.get_resources(subtrees["AB"])) == 0


def test_release_owner_updates_shared_resources(tracker, splits, subtrees):
    """Test that releasing owner from all resources updates shared resources."""
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["CD"])

    tracker.release_owner_from_all_resources(subtrees["AB"])

    # ABC should now be unique to CD
    assert tracker.is_unique(splits["ABC"], subtrees["CD"])
    assert tracker.get_owner_count(splits["ABC"]) == 1


# ============================================================================
# Test: Bulk Operations
# ============================================================================


def test_claim_batch(tracker, splits, subtrees, encoding):
    """Test claiming multiple resources at once."""
    resources = PartitionSet(
        [splits["AB"], splits["CD"], splits["ABC"]], encoding=encoding
    )

    tracker.claim_batch(resources, subtrees["AB"])

    assert tracker.get_resource_count(subtrees["AB"]) == 3
    for resource in resources:
        assert tracker.has_resource(resource)
        assert subtrees["AB"] in tracker.get_owners(resource)


def test_release_batch(tracker, splits, subtrees, encoding):
    """Test releasing multiple resources at once."""
    resources = PartitionSet(
        [splits["AB"], splits["CD"], splits["ABC"]], encoding=encoding
    )

    tracker.claim_batch(resources, subtrees["AB"])
    tracker.release_batch(resources, subtrees["AB"])

    assert not tracker.has_owner(subtrees["AB"])
    for resource in resources:
        assert not tracker.has_resource(resource)


# ============================================================================
# Test: Complex Scenarios
# ============================================================================


def test_multiple_owners_multiple_resources(tracker, splits, subtrees):
    """Test complex scenario with multiple owners and resources."""
    # AB owns AB (unique) and ABC (shared)
    tracker.claim(splits["AB"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["AB"])

    # CD owns CD (unique) and ABC (shared)
    tracker.claim(splits["CD"], subtrees["CD"])
    tracker.claim(splits["ABC"], subtrees["CD"])

    # E owns DE (unique)
    tracker.claim(splits["DE"], subtrees["E"])

    # Verify unique resources
    assert tracker.is_unique(splits["AB"], subtrees["AB"])
    assert tracker.is_unique(splits["CD"], subtrees["CD"])
    assert tracker.is_unique(splits["DE"], subtrees["E"])

    # Verify shared resource
    assert tracker.is_shared(splits["ABC"])
    assert tracker.get_owner_count(splits["ABC"]) == 2

    # Verify resource counts
    assert tracker.get_resource_count(subtrees["AB"]) == 2
    assert tracker.get_resource_count(subtrees["CD"]) == 2
    assert tracker.get_resource_count(subtrees["E"]) == 1


def test_expand_last_strategy_simulation(tracker, splits, subtrees):
    """Test simulation of the 'expand-last' strategy."""
    # Three subtrees share a resource
    tracker.claim(splits["ABC"], subtrees["AB"])
    tracker.claim(splits["ABC"], subtrees["CD"])
    tracker.claim(splits["ABC"], subtrees["E"])

    # First subtree finishes - releases its claim
    tracker.release(splits["ABC"], subtrees["AB"])
    assert not tracker.is_last_owner(splits["ABC"], subtrees["CD"])
    assert not tracker.is_last_owner(splits["ABC"], subtrees["E"])

    # Second subtree finishes
    tracker.release(splits["ABC"], subtrees["CD"])
    assert tracker.is_last_owner(splits["ABC"], subtrees["E"])

    # Last owner can now expand
    assert tracker.get_owner_count(splits["ABC"]) == 1


def test_tabula_rasa_simulation(tracker, splits, subtrees, encoding):
    """Test simulation of tabula rasa (first subtree takes all)."""
    # All collapse splits initially assigned to multiple subtrees
    all_splits = PartitionSet(
        [splits["AB"], splits["CD"], splits["ABC"], splits["DE"], splits["A"]],
        encoding=encoding,
    )

    # Distribute to multiple subtrees
    tracker.claim_batch(all_splits, subtrees["AB"])
    tracker.claim_batch(all_splits, subtrees["CD"])
    tracker.claim_batch(all_splits, subtrees["E"])

    # First subtree processes everything (tabula rasa)
    # Release all splits globally
    for split in all_splits:
        tracker.release_all(split)

    # Verify all splits are gone
    assert len(tracker.get_all_resources()) == 0


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_empty_tracker_queries(tracker, splits, subtrees):
    """Test queries on empty tracker."""
    assert tracker.get_owner_count(splits["AB"]) == 0
    assert tracker.get_resource_count(subtrees["AB"]) == 0
    assert len(tracker.get_owners(splits["AB"])) == 0
    assert len(tracker.get_resources(subtrees["AB"])) == 0
    assert not tracker.is_unique(splits["AB"], subtrees["AB"])
    assert not tracker.is_shared(splits["AB"])
    assert not tracker.is_last_owner(splits["AB"], subtrees["AB"])


def test_stats_after_operations(tracker, splits, subtrees):
    """Test statistics after various operations."""
    # Add unique and shared resources
    tracker.claim(splits["AB"], subtrees["AB"])  # Unique
    tracker.claim(splits["CD"], subtrees["CD"])  # Unique
    tracker.claim(splits["ABC"], subtrees["AB"])  # Shared
    tracker.claim(splits["ABC"], subtrees["CD"])  # Shared

    stats = tracker.get_stats()
    assert stats["total_resources"] == 3
    assert stats["total_owners"] == 2
    assert stats["unique_resources"] == 2
    assert stats["shared_resources"] == 1


def test_repr(tracker):
    """Test string representation."""
    repr_str = repr(tracker)
    assert "OwnershipTracker" in repr_str
    assert "resources=" in repr_str
    assert "owners=" in repr_str


# ============================================================================
# Test: Atomic Behavior
# ============================================================================


def test_claim_is_atomic(tracker, splits, subtrees):
    """Test that claim operation is atomic (all or nothing)."""
    tracker.claim(splits["AB"], subtrees["AB"])

    # Both indices should be updated
    assert splits["AB"] in tracker.get_all_resources()
    assert subtrees["AB"] in tracker.get_all_owners()
    assert subtrees["AB"] in tracker.get_owners(splits["AB"])
    assert splits["AB"] in tracker.get_resources(subtrees["AB"])


def test_release_is_atomic(tracker, splits, subtrees):
    """Test that release operation is atomic (all or nothing)."""
    tracker.claim(splits["AB"], subtrees["AB"])
    tracker.release(splits["AB"], subtrees["AB"])

    # Both indices should be updated
    assert splits["AB"] not in tracker.get_all_resources()
    assert subtrees["AB"] not in tracker.get_all_owners()
    assert len(tracker.get_owners(splits["AB"])) == 0
    assert len(tracker.get_resources(subtrees["AB"])) == 0


# ============================================================================
# Run tests if executed directly
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
