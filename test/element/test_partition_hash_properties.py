"""
Property-based tests for Partition hash and equality operations.

Feature: pipeline-performance-optimization
Tests validate correctness properties for the optimized hash caching implementation.
"""

import pytest
from hypothesis import given, strategies as st, settings

from brancharchitect.elements.partition import Partition


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating valid partition indices (non-negative integers)
indices_strategy = st.lists(
    st.integers(min_value=0, max_value=31),  # Limit to 32 taxa for reasonable bitmasks
    min_size=1,
    max_size=10,
    unique=True,
).map(tuple)


# Strategy for generating encoding dictionaries
def encoding_for_indices(indices: tuple) -> dict:
    """Create an encoding dict that covers the given indices."""
    max_idx = max(indices) if indices else 0
    return {f"T{i}": i for i in range(max_idx + 1)}


# =============================================================================
# Property 1: Hash-Equality Contract
# Validates: Requirements 3.1
# =============================================================================


class TestHashEqualityContract:
    """Property 1: Hash-Equality Contract

    For any two Partitions p1 and p2, if p1 == p2 then hash(p1) == hash(p2).
    """

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_equal_partitions_have_equal_hashes(self, indices: tuple):
        """If two partitions are equal, their hashes must be equal."""
        # Feature: pipeline-performance-optimization, Property 1: Hash-Equality Contract
        encoding = encoding_for_indices(indices)

        # Create two partitions with same indices
        p1 = Partition(indices, encoding)
        p2 = Partition(indices, encoding)

        # They should be equal
        assert p1 == p2, f"Partitions with same indices should be equal"

        # Their hashes must be equal (hash-equality contract)
        assert hash(p1) == hash(p2), (
            f"Equal partitions must have equal hashes: "
            f"hash({p1.indices}) = {hash(p1)}, hash({p2.indices}) = {hash(p2)}"
        )

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_from_bitmask_equal_to_init(self, indices: tuple):
        """Partitions created via __init__ and from_bitmask should be equal and have same hash."""
        # Feature: pipeline-performance-optimization, Property 1: Hash-Equality Contract
        encoding = encoding_for_indices(indices)

        # Create via __init__
        p1 = Partition(indices, encoding)

        # Create via from_bitmask with same bitmask
        p2 = Partition.from_bitmask(p1.bitmask, encoding)

        # They should be equal
        assert p1 == p2, "Partitions with same bitmask should be equal"

        # Their hashes must be equal
        assert hash(p1) == hash(p2), (
            f"Equal partitions must have equal hashes regardless of construction method"
        )


# =============================================================================
# Property 2: Bitmask-Based Equality
# Validates: Requirements 1.1, 1.2, 1.3
# =============================================================================


class TestBitmaskBasedEquality:
    """Property 2: Bitmask-Based Equality

    For any two Partitions p1 and p2, p1 == p2 if and only if p1.bitmask == p2.bitmask.
    """

    @given(indices_strategy, indices_strategy)
    @settings(max_examples=100)
    def test_equality_determined_by_bitmask(self, indices1: tuple, indices2: tuple):
        """Equality is determined solely by bitmask comparison."""
        # Feature: pipeline-performance-optimization, Property 2: Bitmask-Based Equality
        max_idx = max(max(indices1, default=0), max(indices2, default=0))
        encoding = {f"T{i}": i for i in range(max_idx + 1)}

        p1 = Partition(indices1, encoding)
        p2 = Partition(indices2, encoding)

        # Equality should match bitmask equality
        bitmasks_equal = p1.bitmask == p2.bitmask
        partitions_equal = p1 == p2

        assert bitmasks_equal == partitions_equal, (
            f"Partition equality should match bitmask equality: "
            f"bitmasks_equal={bitmasks_equal}, partitions_equal={partitions_equal}"
        )

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_same_bitmask_means_equal(self, indices: tuple):
        """Partitions with same bitmask are equal."""
        # Feature: pipeline-performance-optimization, Property 2: Bitmask-Based Equality
        encoding = encoding_for_indices(indices)

        p1 = Partition(indices, encoding)
        # Create another with same bitmask via from_bitmask
        p2 = Partition.from_bitmask(p1.bitmask, encoding)

        assert p1.bitmask == p2.bitmask, "Bitmasks should be equal"
        assert p1 == p2, "Partitions with equal bitmasks should be equal"

    @given(indices_strategy, indices_strategy)
    @settings(max_examples=100)
    def test_different_bitmask_means_not_equal(self, indices1: tuple, indices2: tuple):
        """Partitions with different bitmasks are not equal."""
        # Feature: pipeline-performance-optimization, Property 2: Bitmask-Based Equality
        max_idx = max(max(indices1, default=0), max(indices2, default=0))
        encoding = {f"T{i}": i for i in range(max_idx + 1)}

        p1 = Partition(indices1, encoding)
        p2 = Partition(indices2, encoding)

        if p1.bitmask != p2.bitmask:
            assert p1 != p2, "Partitions with different bitmasks should not be equal"


# =============================================================================
# Property 3: Hash Consistency
# Validates: Requirements 2.2, 2.3
# =============================================================================


class TestHashConsistency:
    """Property 3: Hash Consistency

    For any Partition p, calling hash(p) multiple times returns the same value.
    """

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_hash_is_consistent_across_calls(self, indices: tuple):
        """Hash returns the same value on repeated calls."""
        # Feature: pipeline-performance-optimization, Property 3: Hash Consistency
        encoding = encoding_for_indices(indices)
        p = Partition(indices, encoding)

        # Call hash multiple times
        h1 = hash(p)
        h2 = hash(p)
        h3 = hash(p)

        assert h1 == h2 == h3, f"Hash should be consistent: {h1}, {h2}, {h3}"

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_hash_equals_cached_hash(self, indices: tuple):
        """Hash returns the cached value."""
        # Feature: pipeline-performance-optimization, Property 3: Hash Consistency
        encoding = encoding_for_indices(indices)
        p = Partition(indices, encoding)

        assert hash(p) == p._cached_hash, (
            f"hash(p) should equal p._cached_hash: {hash(p)} vs {p._cached_hash}"
        )

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_hash_based_on_bitmask(self, indices: tuple):
        """Hash is based on bitmask value."""
        # Feature: pipeline-performance-optimization, Property 3: Hash Consistency
        encoding = encoding_for_indices(indices)
        p = Partition(indices, encoding)

        # Hash should equal hash of bitmask
        assert hash(p) == hash(p.bitmask), (
            f"Partition hash should equal hash of bitmask: {hash(p)} vs {hash(p.bitmask)}"
        )


# =============================================================================
# Property 4: Set Membership Correctness
# Validates: Requirements 3.3
# =============================================================================


class TestSetMembershipCorrectness:
    """Property 4: Set Membership Correctness

    For any Partition p and set s containing p, membership check p in s returns True,
    and for any Partition q where q == p, q in s also returns True.
    """

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_partition_in_set(self, indices: tuple):
        """Partition can be found in a set containing it."""
        # Feature: pipeline-performance-optimization, Property 4: Set Membership Correctness
        encoding = encoding_for_indices(indices)
        p = Partition(indices, encoding)

        s = {p}

        assert p in s, "Partition should be found in set containing it"

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_equal_partition_in_set(self, indices: tuple):
        """Equal partition can be found in set."""
        # Feature: pipeline-performance-optimization, Property 4: Set Membership Correctness
        encoding = encoding_for_indices(indices)

        p1 = Partition(indices, encoding)
        p2 = Partition(indices, encoding)  # Equal to p1

        s = {p1}

        assert p2 in s, "Equal partition should be found in set"

    @given(indices_strategy)
    @settings(max_examples=100)
    def test_from_bitmask_partition_in_set(self, indices: tuple):
        """Partition created via from_bitmask can be found in set."""
        # Feature: pipeline-performance-optimization, Property 4: Set Membership Correctness
        encoding = encoding_for_indices(indices)

        p1 = Partition(indices, encoding)
        p2 = Partition.from_bitmask(p1.bitmask, encoding)

        s = {p1}

        assert p2 in s, (
            "from_bitmask partition should be found in set containing equal partition"
        )

    @given(
        st.lists(
            indices_strategy, min_size=1, max_size=5, unique_by=lambda x: frozenset(x)
        )
    )
    @settings(max_examples=100)
    def test_multiple_partitions_in_set(self, indices_list):
        """Multiple partitions can be stored and retrieved from set."""
        # Feature: pipeline-performance-optimization, Property 4: Set Membership Correctness
        max_idx = max(max(idx, default=0) for idx in indices_list)
        encoding = {f"T{i}": i for i in range(max_idx + 1)}

        partitions = [Partition(idx, encoding) for idx in indices_list]
        s = set(partitions)

        # All partitions should be in the set
        for p in partitions:
            assert p in s, f"Partition {p.indices} should be in set"


# =============================================================================
# Property 5: Dictionary Key Correctness
# Validates: Requirements 3.3
# =============================================================================


class TestDictionaryKeyCorrectness:
    """Property 5: Dictionary Key Correctness

    For any Partition p used as a dictionary key with value v,
    retrieving with an equal Partition q (where p == q) returns v.
    """

    @given(indices_strategy, st.integers())
    @settings(max_examples=100)
    def test_partition_as_dict_key(self, indices: tuple, value: int):
        """Partition can be used as dictionary key."""
        # Feature: pipeline-performance-optimization, Property 5: Dictionary Key Correctness
        encoding = encoding_for_indices(indices)
        p = Partition(indices, encoding)

        d = {p: value}

        assert d[p] == value, "Should retrieve value using same partition"

    @given(indices_strategy, st.integers())
    @settings(max_examples=100)
    def test_equal_partition_retrieves_value(self, indices: tuple, value: int):
        """Equal partition retrieves the same value from dict."""
        # Feature: pipeline-performance-optimization, Property 5: Dictionary Key Correctness
        encoding = encoding_for_indices(indices)

        p1 = Partition(indices, encoding)
        p2 = Partition(indices, encoding)  # Equal to p1

        d = {p1: value}

        assert d[p2] == value, "Equal partition should retrieve same value"

    @given(indices_strategy, st.integers())
    @settings(max_examples=100)
    def test_from_bitmask_partition_retrieves_value(self, indices: tuple, value: int):
        """Partition created via from_bitmask retrieves value from dict."""
        # Feature: pipeline-performance-optimization, Property 5: Dictionary Key Correctness
        encoding = encoding_for_indices(indices)

        p1 = Partition(indices, encoding)
        p2 = Partition.from_bitmask(p1.bitmask, encoding)

        d = {p1: value}

        assert d[p2] == value, "from_bitmask partition should retrieve same value"

    @given(
        st.lists(
            indices_strategy, min_size=1, max_size=5, unique_by=lambda x: frozenset(x)
        )
    )
    @settings(max_examples=100)
    def test_multiple_partitions_as_dict_keys(self, indices_list):
        """Multiple partitions can be used as dictionary keys."""
        # Feature: pipeline-performance-optimization, Property 5: Dictionary Key Correctness
        max_idx = max(max(idx, default=0) for idx in indices_list)
        encoding = {f"T{i}": i for i in range(max_idx + 1)}

        partitions = [Partition(idx, encoding) for idx in indices_list]
        d = {p: i for i, p in enumerate(partitions)}

        # All partitions should retrieve their values
        for i, p in enumerate(partitions):
            assert d[p] == i, f"Partition {p.indices} should retrieve value {i}"
