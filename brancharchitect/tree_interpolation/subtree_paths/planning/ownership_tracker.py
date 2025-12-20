from typing import Dict, Set, FrozenSet
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet


class OwnershipTracker:
    """
    Tracks ownership and usage of splits (resources) by subtrees (consumers).

    This class centralizes the ownership logic that was previously scattered
    across multiple data structures and methods in PivotSplitRegistry.

    Key concepts:
    - Resource: A split (Partition) that can be owned/used
    - Owner: A subtree (Partition) that owns or uses a split
    - Unique ownership: Split has exactly one owner
    - Shared ownership: Split has multiple owners
    - Atomic operations: Claims and releases are transactional

    Example:
        tracker = OwnershipTracker(encoding)
        tracker.claim(split_1, subtree_A)
        tracker.claim(split_2, subtree_A)
        tracker.claim(split_2, subtree_B)  # Now shared

        assert tracker.is_unique(split_1, subtree_A)  # True
        assert tracker.is_shared(split_2)  # True
        assert tracker.get_owner_count(split_2) == 2
    """

    def __init__(self, encoding: Dict[str, int]) -> None:
        """
        Initialize the ownership tracker.

        Args:
            encoding: The taxa encoding dictionary (e.g., {'A': 0, 'B': 1, ...})
        """
        self.encoding = encoding

        # Core ownership mapping: resource -> set of owners
        self._resource_to_owners: Dict[Partition, Set[Partition]] = {}

        # Reverse index for fast lookups: owner -> set of resources
        self._owner_to_resources: Dict[Partition, Set[Partition]] = {}

    # ========================================================================
    # Core Operations: Claim and Release
    # ========================================================================

    def claim(self, resource: Partition, owner: Partition) -> None:
        """
        Owner claims ownership of a resource.

        This is idempotent - claiming the same resource multiple times has no effect.

        Args:
            resource: The split being claimed
            owner: The subtree claiming the split

        Example:
            tracker.claim(split_1, subtree_A)
            tracker.claim(split_1, subtree_B)  # Now shared
        """
        # Add to resource->owners mapping
        if resource not in self._resource_to_owners:
            self._resource_to_owners[resource] = set()
        self._resource_to_owners[resource].add(owner)

        # Add to owner->resources reverse index
        if owner not in self._owner_to_resources:
            self._owner_to_resources[owner] = set()
        self._owner_to_resources[owner].add(resource)

    def release(self, resource: Partition, owner: Partition) -> None:
        """
        Owner releases ownership of a resource.

        If this was the last owner, the resource is fully released.

        Args:
            resource: The split being released
            owner: The subtree releasing the split

        Example:
            tracker.release(split_1, subtree_A)
        """
        # Remove from resource->owners mapping
        if resource in self._resource_to_owners:
            self._resource_to_owners[resource].discard(owner)
            # Clean up if no more owners
            if not self._resource_to_owners[resource]:
                del self._resource_to_owners[resource]

        # Remove from owner->resources reverse index
        if owner in self._owner_to_resources:
            self._owner_to_resources[owner].discard(resource)
            # Clean up if owner has no more resources
            if not self._owner_to_resources[owner]:
                del self._owner_to_resources[owner]

    def release_all(self, resource: Partition) -> None:
        """
        Release a resource from ALL owners (global deletion).

        This is used when a collapse split is processed - all subtrees
        lose access to it.

        Args:
            resource: The split being globally released

        Example:
            tracker.release_all(split_1)  # All owners lose this split
        """
        if resource not in self._resource_to_owners:
            return

        # Get all current owners
        owners = self._resource_to_owners[resource].copy()

        # Release from each owner
        for owner in owners:
            if owner in self._owner_to_resources:
                self._owner_to_resources[owner].discard(resource)
                if not self._owner_to_resources[owner]:
                    del self._owner_to_resources[owner]

        # Remove resource entry
        del self._resource_to_owners[resource]

    def release_owner_from_all_resources(self, owner: Partition) -> None:
        """
        Release an owner from ALL their resources.

        This is used when a subtree is finished processing - it drops
        all remaining claims.

        Args:
            owner: The subtree releasing all claims

        Example:
            tracker.release_owner_from_all_resources(subtree_A)
        """
        if owner not in self._owner_to_resources:
            return

        # Get all resources owned
        resources = self._owner_to_resources[owner].copy()

        # Release each resource
        for resource in resources:
            if resource in self._resource_to_owners:
                self._resource_to_owners[resource].discard(owner)
                if not self._resource_to_owners[resource]:
                    del self._resource_to_owners[resource]

        # Remove owner entry
        del self._owner_to_resources[owner]

    # ========================================================================
    # Query Operations: Ownership Information
    # ========================================================================

    def get_owners(self, resource: Partition) -> FrozenSet[Partition]:
        """
        Get all owners of a resource.

        Returns an immutable set to prevent external modification.

        Args:
            resource: The split to query

        Returns:
            Frozen set of owners (empty if resource has no owners)
        """
        return frozenset(self._resource_to_owners.get(resource, set()))

    def get_resources(self, owner: Partition) -> FrozenSet[Partition]:
        """
        Get all resources owned by an owner.

        Returns an immutable set to prevent external modification.

        Args:
            owner: The subtree to query

        Returns:
            Frozen set of resources (empty if owner has no resources)
        """
        return frozenset(self._owner_to_resources.get(owner, set()))

    def get_owner_count(self, resource: Partition) -> int:
        """
        Get the number of owners for a resource.

        Args:
            resource: The split to query

        Returns:
            Number of owners (0 if resource is not tracked)
        """
        return len(self._resource_to_owners.get(resource, set()))

    def get_resource_count(self, owner: Partition) -> int:
        """
        Get the number of resources owned by an owner.

        Args:
            owner: The subtree to query

        Returns:
            Number of resources (0 if owner has no resources)
        """
        return len(self._owner_to_resources.get(owner, set()))

    # ========================================================================
    # Ownership Type Queries
    # ========================================================================

    def is_unique(self, resource: Partition, owner: Partition) -> bool:
        """
        Check if owner is the ONLY owner of this resource.

        Args:
            resource: The split to check
            owner: The subtree to check

        Returns:
            True if owner is the sole owner, False otherwise
        """
        owners = self._resource_to_owners.get(resource, set())
        return owners == {owner}

    def is_shared(self, resource: Partition) -> bool:
        """
        Check if a resource has multiple owners.

        Args:
            resource: The split to check

        Returns:
            True if resource has 2+ owners, False otherwise
        """
        return self.get_owner_count(resource) > 1

    def is_last_owner(self, resource: Partition, owner: Partition) -> bool:
        """
        Check if owner is the last remaining owner of this resource.

        This is used for the "expand-last" strategy where a subtree
        expands a split when it becomes the last user.

        Args:
            resource: The split to check
            owner: The subtree to check

        Returns:
            True if owner is the only remaining owner
        """
        owners = self._resource_to_owners.get(resource, set())
        return len(owners) == 1 and owner in owners

    def has_resource(self, resource: Partition) -> bool:
        """
        Check if a resource is tracked (has any owners).

        Args:
            resource: The split to check

        Returns:
            True if resource has at least one owner
        """
        return resource in self._resource_to_owners

    def has_owner(self, owner: Partition) -> bool:
        """
        Check if an owner is tracked (owns any resources).

        Args:
            owner: The subtree to check

        Returns:
            True if owner has at least one resource
        """
        return owner in self._owner_to_resources

    # ========================================================================
    # Categorization: Unique vs Shared
    # ========================================================================

    def get_unique_resources(self, owner: Partition) -> PartitionSet[Partition]:
        """
        Get all resources uniquely owned by this owner.

        Args:
            owner: The subtree to query

        Returns:
            Set of resources where owner is the sole owner
        """
        return PartitionSet(
            {
                resource
                for resource in self._owner_to_resources.get(owner, set())
                if self.is_unique(resource, owner)
            },
            encoding=self.encoding,
        )

    def get_shared_resources(self, owner: Partition) -> PartitionSet[Partition]:
        """
        Get all resources shared by this owner with others.

        Args:
            owner: The subtree to query

        Returns:
            Set of resources where owner shares with other owners
        """
        return PartitionSet(
            {
                resource
                for resource in self._owner_to_resources.get(owner, set())
                if self.is_shared(resource)
            },
            encoding=self.encoding,
        )

    def get_all_unique_resources(self) -> Dict[Partition, Partition]:
        """
        Get all uniquely-owned resources across all owners.

        Returns:
            Dictionary mapping resource -> sole owner
        """
        return {
            resource: next(iter(owners))
            for resource, owners in self._resource_to_owners.items()
            if len(owners) == 1
        }

    def get_all_shared_resources(self) -> Dict[Partition, FrozenSet[Partition]]:
        """
        Get all shared resources across all owners.

        Returns:
            Dictionary mapping resource -> set of owners (frozen)
        """
        return {
            resource: frozenset(owners)
            for resource, owners in self._resource_to_owners.items()
            if len(owners) > 1
        }

    def get_all_owners(self) -> FrozenSet[Partition]:
        """
        Get all owners that have any resources.

        Returns:
            Frozen set of all tracked owners
        """
        return frozenset(self._owner_to_resources.keys())

    def get_all_resources(self) -> FrozenSet[Partition]:
        """
        Get all resources that have any owners.

        Returns:
            Frozen set of all tracked resources
        """
        return frozenset(self._resource_to_owners.keys())

    # ========================================================================
    # Bulk Operations
    # ========================================================================

    def claim_batch(self, resources: PartitionSet[Partition], owner: Partition) -> None:
        """
        Claim multiple resources atomically.

        Args:
            resources: Set of splits to claim
            owner: The subtree claiming them
        """
        for resource in resources:
            self.claim(resource, owner)

    def release_batch(
        self, resources: PartitionSet[Partition], owner: Partition
    ) -> None:
        """
        Release multiple resources atomically.

        Args:
            resources: Set of splits to release
            owner: The subtree releasing them
        """
        for resource in resources:
            self.release(resource, owner)

    # ========================================================================
    # Debug and Inspection
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"OwnershipTracker("
            f"resources={len(self._resource_to_owners)}, "
            f"owners={len(self._owner_to_resources)})"
        )

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about current ownership state.

        Returns:
            Dictionary with counts of resources, owners, unique, shared
        """
        unique_count = sum(
            1 for owners in self._resource_to_owners.values() if len(owners) == 1
        )
        shared_count = sum(
            1 for owners in self._resource_to_owners.values() if len(owners) > 1
        )

        return {
            "total_resources": len(self._resource_to_owners),
            "total_owners": len(self._owner_to_resources),
            "unique_resources": unique_count,
            "shared_resources": shared_count,
        }
