"""
Property-based tests for PathGroupManager.

Tests the path-based grouping and topological sorting functionality
using Hypothesis for property-based testing.

Feature: expand-path-grouping
"""

from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Set, List, FrozenSet, Optional, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.planning.path_group_manager import (
    PathGroupManager,
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


def create_encoding(num_taxa: int) -> Dict[str, int]:
    """Create a simple encoding for testing."""
    return {f"T{i}": i for i in range(num_taxa)}


def create_partition(indices: FrozenSet[int], encoding: Dict[str, int]) -> Partition:
    """Create a partition from indices."""
    return Partition(indices=indices, encoding=encoding)


def create_partition_set(
    partitions: Set[FrozenSet[int]], encoding: Dict[str, int]
) -> PartitionSet[Partition]:
    """Create a PartitionSet from a set of index sets."""
    ps = PartitionSet(encoding=encoding)
    for indices in partitions:
        ps.add(create_partition(indices, encoding))
    return ps


# Hypothesis strategies for generating test data
@st.composite
def partition_indices(draw, max_taxa: int = 10):
    """Generate a frozenset of indices representing a partition."""
    num_taxa = draw(st.integers(min_value=1, max_value=max_taxa))
    indices = draw(
        st.frozensets(
            st.integers(min_value=0, max_value=max_taxa - 1),
            min_size=1,
            max_size=num_taxa,
        )
    )
    return indices


@st.composite
def expand_paths_data(
    draw, min_subtrees: int = 1, max_subtrees: int = 5, max_taxa: int = 10
):
    """
    Generate expand paths data for testing.

    Returns:
        Tuple of (expand_splits_by_subtree dict, encoding dict)
    """
    encoding = create_encoding(max_taxa)
    num_subtrees = draw(st.integers(min_value=min_subtrees, max_value=max_subtrees))

    expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]] = {}

    for i in range(num_subtrees):
        # Create subtree partition
        subtree_indices = draw(
            st.frozensets(
                st.integers(min_value=0, max_value=max_taxa - 1),
                min_size=1,
                max_size=max_taxa,
            )
        )
        subtree = create_partition(subtree_indices, encoding)

        # Create expand path (set of splits)
        num_splits = draw(st.integers(min_value=0, max_value=5))
        path_splits: Set[FrozenSet[int]] = set()
        for _ in range(num_splits):
            split_indices = draw(
                st.frozensets(
                    st.integers(min_value=0, max_value=max_taxa - 1),
                    min_size=1,
                    max_size=max_taxa,
                )
            )
            path_splits.add(split_indices)

        expand_splits_by_subtree[subtree] = create_partition_set(path_splits, encoding)

    return expand_splits_by_subtree, encoding


@st.composite
def overlapping_paths_data(draw, max_taxa: int = 10):
    """
    Generate two subtrees with guaranteed overlapping expand paths.

    Returns:
        Tuple of (expand_splits_by_subtree dict, encoding dict, shared_split)
    """
    encoding = create_encoding(max_taxa)

    # Create a shared split that will be in both paths
    shared_indices = draw(
        st.frozensets(
            st.integers(min_value=0, max_value=max_taxa - 1),
            min_size=1,
            max_size=max_taxa,
        )
    )

    # Create two subtrees
    subtree_a_indices = draw(
        st.frozensets(
            st.integers(min_value=0, max_value=max_taxa - 1),
            min_size=1,
            max_size=max_taxa,
        )
    )
    subtree_b_indices = draw(
        st.frozensets(
            st.integers(min_value=0, max_value=max_taxa - 1),
            min_size=1,
            max_size=max_taxa,
        )
    )

    # Ensure subtrees are different
    assume(subtree_a_indices != subtree_b_indices)

    subtree_a = create_partition(subtree_a_indices, encoding)
    subtree_b = create_partition(subtree_b_indices, encoding)

    # Create paths that both contain the shared split
    path_a_splits = {shared_indices}
    path_b_splits = {shared_indices}

    # Add some unique splits to each path
    num_unique_a = draw(st.integers(min_value=0, max_value=3))
    for _ in range(num_unique_a):
        unique_indices = draw(
            st.frozensets(
                st.integers(min_value=0, max_value=max_taxa - 1),
                min_size=1,
                max_size=max_taxa,
            )
        )
        if unique_indices != shared_indices:
            path_a_splits.add(unique_indices)

    num_unique_b = draw(st.integers(min_value=0, max_value=3))
    for _ in range(num_unique_b):
        unique_indices = draw(
            st.frozensets(
                st.integers(min_value=0, max_value=max_taxa - 1),
                min_size=1,
                max_size=max_taxa,
            )
        )
        if unique_indices != shared_indices:
            path_b_splits.add(unique_indices)

    expand_splits_by_subtree = {
        subtree_a: create_partition_set(path_a_splits, encoding),
        subtree_b: create_partition_set(path_b_splits, encoding),
    }

    shared_split = create_partition(shared_indices, encoding)

    return expand_splits_by_subtree, encoding, shared_split


@st.composite
def containment_paths_data(draw, max_taxa: int = 10):
    """
    Generate two subtrees where one's expand path is a proper subset of the other's.

    Returns:
        Tuple of (expand_splits_by_subtree dict, encoding dict, contained_subtree, container_subtree)
    """
    encoding = create_encoding(max_taxa)

    # Create the smaller (contained) path first
    num_contained_splits = draw(st.integers(min_value=1, max_value=3))
    contained_splits: Set[FrozenSet[int]] = set()
    for _ in range(num_contained_splits):
        split_indices = draw(
            st.frozensets(
                st.integers(min_value=0, max_value=max_taxa - 1),
                min_size=1,
                max_size=max_taxa,
            )
        )
        contained_splits.add(split_indices)

    # Create the larger (container) path that includes all contained splits plus more
    container_splits = contained_splits.copy()
    num_extra_splits = draw(st.integers(min_value=1, max_value=3))
    for _ in range(num_extra_splits):
        extra_indices = draw(
            st.frozensets(
                st.integers(min_value=0, max_value=max_taxa - 1),
                min_size=1,
                max_size=max_taxa,
            )
        )
        if extra_indices not in contained_splits:
            container_splits.add(extra_indices)

    # Ensure container is strictly larger
    assume(len(container_splits) > len(contained_splits))
    assume(contained_splits < container_splits)  # Proper subset

    # Create two different subtrees
    subtree_a_indices = draw(
        st.frozensets(
            st.integers(min_value=0, max_value=max_taxa - 1),
            min_size=1,
            max_size=max_taxa,
        )
    )
    subtree_b_indices = draw(
        st.frozensets(
            st.integers(min_value=0, max_value=max_taxa - 1),
            min_size=1,
            max_size=max_taxa,
        )
    )
    assume(subtree_a_indices != subtree_b_indices)

    contained_subtree = create_partition(subtree_a_indices, encoding)
    container_subtree = create_partition(subtree_b_indices, encoding)

    expand_splits_by_subtree = {
        contained_subtree: create_partition_set(contained_splits, encoding),
        container_subtree: create_partition_set(container_splits, encoding),
    }

    return expand_splits_by_subtree, encoding, contained_subtree, container_subtree


# ============================================================================
# Property Tests
# ============================================================================


class TestPathOverlapDetection:
    """
    Property 1: Path Overlap Detection

    For any two subtrees A and B with expand paths that share at least one split,
    the system should mark them as having path overlap.

    Feature: expand-path-grouping, Property 1: Path Overlap Detection
    Validates: Requirements 1.2
    """

    @given(overlapping_paths_data())
    @settings(max_examples=100)
    def test_overlapping_paths_detected(self, data):
        """
        Property: For any two subtrees with overlapping paths, has_overlap returns True.
        """
        expand_splits_by_subtree, encoding, shared_split = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        subtrees = list(expand_splits_by_subtree.keys())
        assert len(subtrees) == 2

        subtree_a, subtree_b = subtrees

        # Both directions should report overlap
        assert manager.has_overlap(subtree_a, subtree_b), (
            f"Expected overlap between subtrees with shared split {shared_split}"
        )
        assert manager.has_overlap(subtree_b, subtree_a), (
            "Expected symmetric overlap detection"
        )

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_overlap_symmetry(self, data):
        """
        Property: Overlap detection is symmetric - if A overlaps B, then B overlaps A.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        subtrees = list(expand_splits_by_subtree.keys())

        for i, subtree_a in enumerate(subtrees):
            for subtree_b in subtrees[i + 1 :]:
                overlap_ab = manager.has_overlap(subtree_a, subtree_b)
                overlap_ba = manager.has_overlap(subtree_b, subtree_a)
                assert overlap_ab == overlap_ba, (
                    f"Overlap should be symmetric: {overlap_ab} != {overlap_ba}"
                )

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_no_false_positives(self, data):
        """
        Property: If two paths have no common splits, has_overlap returns False.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        subtrees = list(expand_splits_by_subtree.keys())

        for i, subtree_a in enumerate(subtrees):
            for subtree_b in subtrees[i + 1 :]:
                path_a = set(expand_splits_by_subtree[subtree_a])
                path_b = set(expand_splits_by_subtree[subtree_b])

                actual_overlap = bool(path_a & path_b)
                detected_overlap = manager.has_overlap(subtree_a, subtree_b)

                assert actual_overlap == detected_overlap, (
                    f"Overlap detection mismatch: actual={actual_overlap}, detected={detected_overlap}"
                )


class TestPathContainmentDetection:
    """
    Property 2: Path Containment Detection

    For any two subtrees A and B where A's expand path is a proper subset of B's,
    the system should mark them as having a containment relationship.

    Feature: expand-path-grouping, Property 2: Path Containment Detection
    Validates: Requirements 1.3
    """

    @given(containment_paths_data())
    @settings(max_examples=100)
    def test_containment_detected(self, data):
        """
        Property: For any two subtrees where one path is a proper subset of the other,
        has_containment returns True for the correct direction.
        """
        expand_splits_by_subtree, encoding, contained_subtree, container_subtree = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Contained should be marked as contained in container
        assert manager.has_containment(contained_subtree, container_subtree), (
            f"Expected containment: {contained_subtree} in {container_subtree}"
        )

        # Reverse should NOT be true
        assert not manager.has_containment(container_subtree, contained_subtree), (
            "Containment should not be symmetric"
        )

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_containment_correctness(self, data):
        """
        Property: Containment is detected if and only if one path is a proper subset.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        subtrees = list(expand_splits_by_subtree.keys())

        for subtree_a in subtrees:
            for subtree_b in subtrees:
                if subtree_a == subtree_b:
                    continue

                path_a = set(expand_splits_by_subtree[subtree_a])
                path_b = set(expand_splits_by_subtree[subtree_b])

                # A is contained in B if A is a non-empty proper subset of B
                # Empty sets are not considered contained in anything
                actual_containment = bool(path_a) and bool(path_b) and path_a < path_b
                detected_containment = manager.has_containment(subtree_a, subtree_b)

                assert actual_containment == detected_containment, (
                    f"Containment detection mismatch for {subtree_a} in {subtree_b}: "
                    f"actual={actual_containment}, detected={detected_containment}, "
                    f"path_a={path_a}, path_b={path_b}"
                )

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_containment_antisymmetry(self, data):
        """
        Property: If A is contained in B, then B is NOT contained in A.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        containment_edges = manager.get_containment_edges()

        for contained, container in containment_edges:
            # Reverse should not exist
            assert (container, contained) not in containment_edges, (
                f"Containment should be antisymmetric: both ({contained}, {container}) "
                f"and ({container}, {contained}) exist"
            )


class TestConnectedComponentFormation:
    """
    Property 3: Connected Component Formation

    For any set of subtrees with overlapping paths forming a chain
    (A overlaps B, B overlaps C), all subtrees in the chain should be
    assigned to the same path group (transitive closure).

    Feature: expand-path-grouping, Property 3: Connected Component Formation
    Validates: Requirements 2.1, 2.2
    """

    def test_explicit_chain_forms_single_group(self):
        """
        Property: A chain of overlapping subtrees should all be in the same group.

        Uses explicit test data to verify chain behavior.
        """
        encoding = create_encoding(10)

        # Create 3 subtrees with a chain of overlaps: A-B-C
        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)
        subtree_c = create_partition(frozenset({2}), encoding)

        # Shared splits: A and B share split_ab, B and C share split_bc
        split_ab = create_partition(frozenset({3, 4}), encoding)
        split_bc = create_partition(frozenset({5, 6}), encoding)
        split_a_only = create_partition(frozenset({7}), encoding)
        split_c_only = create_partition(frozenset({8}), encoding)

        expand_splits_by_subtree = {
            subtree_a: PartitionSet({split_ab, split_a_only}, encoding=encoding),
            subtree_b: PartitionSet({split_ab, split_bc}, encoding=encoding),
            subtree_c: PartitionSet({split_bc, split_c_only}, encoding=encoding),
        }

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # All three should be in the same group
        group_a = manager.get_group(subtree_a)
        group_b = manager.get_group(subtree_b)
        group_c = manager.get_group(subtree_c)

        assert group_a == group_b == group_c, (
            f"Chain A-B-C should be in same group, got: A={group_a}, B={group_b}, C={group_c}"
        )

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_transitive_closure(self, data):
        """
        Property: If A overlaps B and B overlaps C, then A and C are in the same group.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        subtrees = list(expand_splits_by_subtree.keys())

        # Build transitive closure of overlap relation
        # Start with direct overlaps
        same_group: Dict[Partition, Set[Partition]] = {s: {s} for s in subtrees}

        for subtree_a in subtrees:
            for subtree_b in subtrees:
                if manager.has_overlap(subtree_a, subtree_b):
                    same_group[subtree_a].add(subtree_b)
                    same_group[subtree_b].add(subtree_a)

        # Compute transitive closure
        changed = True
        while changed:
            changed = False
            for subtree in subtrees:
                current_group = same_group[subtree].copy()
                for member in current_group:
                    if same_group[member] - same_group[subtree]:
                        same_group[subtree].update(same_group[member])
                        changed = True

        # Verify: subtrees in same transitive closure should have same group
        for subtree_a in subtrees:
            for subtree_b in same_group[subtree_a]:
                group_a = manager.get_group(subtree_a)
                group_b = manager.get_group(subtree_b)
                assert group_a == group_b, (
                    f"Subtrees {subtree_a} and {subtree_b} should be in same group "
                    f"(transitive closure), but got groups {group_a} and {group_b}"
                )


class TestSubtreeToGroupInvariant:
    """
    Property 4: Subtree-to-Group Assignment Invariant

    For any set of subtrees, each subtree should be assigned to exactly one path group.

    Feature: expand-path-grouping, Property 4: Subtree-to-Group Assignment Invariant
    Validates: Requirements 2.3
    """

    @given(expand_paths_data(min_subtrees=1, max_subtrees=5))
    @settings(max_examples=100)
    def test_every_subtree_has_group(self, data):
        """
        Property: Every subtree should be assigned to exactly one group.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        for subtree in expand_splits_by_subtree:
            group = manager.get_group(subtree)
            assert group is not None, f"Subtree {subtree} has no group assignment"
            assert isinstance(group, int), (
                f"Group should be an integer, got {type(group)}"
            )
            assert 0 <= group < manager.get_num_groups(), (
                f"Group index {group} out of range [0, {manager.get_num_groups()})"
            )

    @given(expand_paths_data(min_subtrees=1, max_subtrees=5))
    @settings(max_examples=100)
    def test_groups_partition_subtrees(self, data):
        """
        Property: Groups should partition the set of subtrees (no overlap, complete coverage).
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        all_subtrees = set(expand_splits_by_subtree.keys())
        subtrees_in_groups: Set[Partition] = set()

        for group_idx in range(manager.get_num_groups()):
            group_members = manager.get_group_members(group_idx)

            # Check no overlap with previous groups
            overlap = subtrees_in_groups & group_members
            assert not overlap, f"Groups overlap: {overlap}"

            subtrees_in_groups.update(group_members)

        # Check complete coverage
        assert subtrees_in_groups == all_subtrees, (
            f"Groups don't cover all subtrees. Missing: {all_subtrees - subtrees_in_groups}"
        )


class TestSingletonGroups:
    """
    Property 5: Singleton Groups for Isolated Subtrees

    For any subtree with no path overlap with any other subtree,
    the system should place it in its own singleton group.

    Feature: expand-path-grouping, Property 5: Singleton Groups for Isolated Subtrees
    Validates: Requirements 2.4
    """

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_isolated_subtrees_get_singleton_groups(self, data):
        """
        Property: Subtrees with no overlaps should be in singleton groups.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        subtrees = list(expand_splits_by_subtree.keys())

        for subtree in subtrees:
            # Check if this subtree has any overlaps
            has_overlap = any(
                manager.has_overlap(subtree, other)
                for other in subtrees
                if other != subtree
            )

            if not has_overlap:
                # Should be in a singleton group
                group_idx = manager.get_group(subtree)
                if group_idx is not None:
                    group_members = manager.get_group_members(group_idx)
                    assert len(group_members) == 1, (
                        f"Isolated subtree {subtree} should be in singleton group, "
                        f"but group has {len(group_members)} members"
                    )


class TestContainmentOrdering:
    """
    Property 6: Containment Implies Ordering

    For any two subtrees A and B where A's expand path is contained in B's,
    A should be processed before B in the output order.

    Feature: expand-path-grouping, Property 6: Containment Implies Ordering
    Validates: Requirements 3.3, 5.1
    """

    @given(containment_paths_data())
    @settings(max_examples=100)
    def test_contained_processed_before_container(self, data):
        """
        Property: If A's path is contained in B's path, A is processed before B.
        """
        expand_splits_by_subtree, encoding, contained_subtree, container_subtree = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Get the processing order
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        # Find positions in order
        contained_pos = (
            order.index(contained_subtree) if contained_subtree in order else -1
        )
        container_pos = (
            order.index(container_subtree) if container_subtree in order else -1
        )

        assert contained_pos != -1, "Contained subtree not in order"
        assert container_pos != -1, "Container subtree not in order"
        assert contained_pos < container_pos, (
            f"Contained subtree should be processed before container: "
            f"contained at {contained_pos}, container at {container_pos}"
        )

    def test_explicit_containment_ordering(self):
        """
        Property: Explicit test for containment ordering with known data.
        """
        encoding = create_encoding(10)

        # Create subtrees with containment: A's path ⊂ B's path
        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)

        # A has path {split1}, B has path {split1, split2}
        split1 = create_partition(frozenset({2, 3}), encoding)
        split2 = create_partition(frozenset({4, 5}), encoding)

        expand_splits_by_subtree = {
            subtree_a: PartitionSet({split1}, encoding=encoding),
            subtree_b: PartitionSet({split1, split2}, encoding=encoding),
        }

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Get processing order
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        assert order == [subtree_a, subtree_b], (
            f"Expected [A, B] order due to containment, got {order}"
        )


class TestSmallestFirstSelection:
    """
    Property 7: Smallest-First Selection Among Ready Nodes

    For any set of subtrees with zero in-degree (no unprocessed dependencies),
    the system should select the one with the smallest expand path size.

    Feature: expand-path-grouping, Property 7: Smallest-First Selection Among Ready Nodes
    Validates: Requirements 3.4, 4.1, 4.2, 4.4
    """

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_smallest_selected_first(self, data):
        """
        Property: Among ready subtrees, the smallest is always selected first.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Get the first subtree selected
        first_subtree = manager.get_next_subtree(set())

        if first_subtree is None:
            return  # Empty input

        first_size = manager.get_path_size(first_subtree)

        # Check that no other subtree with zero in-degree has a smaller path
        for subtree in expand_splits_by_subtree:
            if subtree == first_subtree:
                continue

            # Check if this subtree has zero in-degree (no dependencies)
            has_dependency = any(
                manager.has_containment(other, subtree)
                for other in expand_splits_by_subtree
                if other != subtree
            )

            if not has_dependency:
                other_size = manager.get_path_size(subtree)
                # First selected should have smallest or equal size
                assert first_size <= other_size or manager.get_group(
                    first_subtree
                ) != manager.get_group(subtree), (
                    f"First subtree has size {first_size}, but another ready subtree "
                    f"in same group has smaller size {other_size}"
                )

    def test_explicit_smallest_first(self):
        """
        Property: Explicit test for smallest-first selection.
        """
        encoding = create_encoding(10)

        # Create 3 subtrees with different path sizes, no containment
        subtree_small = create_partition(frozenset({0}), encoding)
        subtree_medium = create_partition(frozenset({1}), encoding)
        subtree_large = create_partition(frozenset({2}), encoding)

        # Different sized paths with no overlap (so they're in different groups)
        split_s = create_partition(frozenset({3}), encoding)
        split_m1 = create_partition(frozenset({4}), encoding)
        split_m2 = create_partition(frozenset({5}), encoding)
        split_l1 = create_partition(frozenset({6}), encoding)
        split_l2 = create_partition(frozenset({7}), encoding)
        split_l3 = create_partition(frozenset({8}), encoding)

        expand_splits_by_subtree = {
            subtree_small: PartitionSet({split_s}, encoding=encoding),  # size 1
            subtree_medium: PartitionSet(
                {split_m1, split_m2}, encoding=encoding
            ),  # size 2
            subtree_large: PartitionSet(
                {split_l1, split_l2, split_l3}, encoding=encoding
            ),  # size 3
        }

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # First selected should be smallest
        first = manager.get_next_subtree(set())
        assert first == subtree_small, f"Expected smallest subtree first, got {first}"


class TestDeterministicOrdering:
    """
    Property 8: Deterministic Ordering

    For any input, running the algorithm multiple times should produce
    identical output orderings.

    Feature: expand-path-grouping, Property 8: Deterministic Ordering
    Validates: Requirements 3.6
    """

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_deterministic_output(self, data):
        """
        Property: Same input produces same output order every time.
        """
        expand_splits_by_subtree, encoding = data

        def get_full_order(manager: PathGroupManager) -> List[Partition]:
            processed: Set[Partition] = set()
            order: List[Partition] = []
            while True:
                next_subtree = manager.get_next_subtree(processed)
                if next_subtree is None:
                    break
                order.append(next_subtree)
                processed.add(next_subtree)
            return order

        # Run twice with fresh managers
        manager1 = PathGroupManager(expand_splits_by_subtree, encoding)
        order1 = get_full_order(manager1)

        manager2 = PathGroupManager(expand_splits_by_subtree, encoding)
        order2 = get_full_order(manager2)

        assert order1 == order2, f"Non-deterministic ordering: {order1} != {order2}"

    def test_explicit_determinism(self):
        """
        Property: Explicit test for deterministic ordering with tie-breaking.
        """
        encoding = create_encoding(10)

        # Create subtrees with same path size (need tie-breaking)
        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)

        # Same size paths, no overlap
        split_a = create_partition(frozenset({2}), encoding)
        split_b = create_partition(frozenset({3}), encoding)

        expand_splits_by_subtree = {
            subtree_a: PartitionSet({split_a}, encoding=encoding),
            subtree_b: PartitionSet({split_b}, encoding=encoding),
        }

        # Run 5 times
        orders = []
        for _ in range(5):
            manager = PathGroupManager(expand_splits_by_subtree, encoding)
            processed: Set[Partition] = set()
            order: List[Partition] = []
            while True:
                next_subtree = manager.get_next_subtree(processed)
                if next_subtree is None:
                    break
                order.append(next_subtree)
                processed.add(next_subtree)
            orders.append(order)

        # All orders should be identical
        for i, order in enumerate(orders[1:], 1):
            assert order == orders[0], (
                f"Order {i} differs from order 0: {order} != {orders[0]}"
            )


class TestGroupCohesion:
    """
    Property 9: Group Cohesion

    For any two subtrees in the same path group, all subtrees in that group
    should be processed consecutively without interleaving with subtrees
    from other groups.

    Feature: expand-path-grouping, Property 9: Group Cohesion
    Validates: Requirements 6.3, 6.4
    """

    @given(expand_paths_data(min_subtrees=2, max_subtrees=6))
    @settings(max_examples=100)
    def test_group_members_processed_consecutively(self, data):
        """
        Property: All members of a group are processed before any member of the next group.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Get the full processing order
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        if not order:
            return  # Empty input

        # Track which groups we've seen and verify cohesion
        seen_groups: Set[int] = set()
        current_group: Optional[int] = None

        for subtree in order:
            group = manager.get_group(subtree)
            if group is None:
                continue

            if current_group is None:
                current_group = group
                seen_groups.add(group)
            elif group != current_group:
                # Switching to a new group - verify we haven't seen it before
                assert group not in seen_groups, (
                    f"Group {group} appears non-consecutively in order. "
                    f"Already processed groups: {seen_groups}"
                )
                seen_groups.add(group)
                current_group = group

    def test_explicit_group_cohesion(self):
        """
        Property: Explicit test for group cohesion with known data.
        """
        encoding = create_encoding(10)

        # Create two groups:
        # Group 1: subtree_a and subtree_b (overlapping paths)
        # Group 2: subtree_c (isolated)
        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)
        subtree_c = create_partition(frozenset({2}), encoding)

        # Shared split for group 1
        shared_split = create_partition(frozenset({3, 4}), encoding)
        split_a_only = create_partition(frozenset({5}), encoding)
        split_b_only = create_partition(frozenset({6}), encoding)
        split_c = create_partition(frozenset({7, 8, 9}), encoding)  # Larger path

        expand_splits_by_subtree = {
            subtree_a: PartitionSet({shared_split, split_a_only}, encoding=encoding),
            subtree_b: PartitionSet({shared_split, split_b_only}, encoding=encoding),
            subtree_c: PartitionSet({split_c}, encoding=encoding),
        }

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Verify a and b are in same group, c is in different group
        group_a = manager.get_group(subtree_a)
        group_b = manager.get_group(subtree_b)
        group_c = manager.get_group(subtree_c)

        assert group_a == group_b, "A and B should be in same group"
        assert group_c != group_a, "C should be in different group"

        # Get processing order
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        # Verify cohesion: a and b should be consecutive (either order)
        pos_a = order.index(subtree_a)
        pos_b = order.index(subtree_b)
        pos_c = order.index(subtree_c)

        # a and b should be adjacent
        assert abs(pos_a - pos_b) == 1, (
            f"A and B should be consecutive, but positions are {pos_a} and {pos_b}"
        )

        # c should be either before both or after both
        assert (pos_c < min(pos_a, pos_b)) or (pos_c > max(pos_a, pos_b)), (
            f"C should not be between A and B. Positions: A={pos_a}, B={pos_b}, C={pos_c}"
        )

    @given(expand_paths_data(min_subtrees=3, max_subtrees=6))
    @settings(max_examples=100)
    def test_no_interleaving(self, data):
        """
        Property: Once we start processing a group, we finish it before starting another.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Get the full processing order
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        if len(order) < 2:
            return

        # For each group, find first and last occurrence in order
        group_spans: Dict[int, Tuple[int, int]] = {}

        for pos, subtree in enumerate(order):
            group = manager.get_group(subtree)
            if group is None:
                continue

            if group not in group_spans:
                group_spans[group] = (pos, pos)
            else:
                first, _ = group_spans[group]
                group_spans[group] = (first, pos)

        # Verify no spans overlap (except at boundaries)
        groups = list(group_spans.keys())
        for i, group_i in enumerate(groups):
            for group_j in groups[i + 1 :]:
                first_i, last_i = group_spans[group_i]
                first_j, last_j = group_spans[group_j]

                # Spans should not overlap
                overlaps = not (last_i < first_j or last_j < first_i)
                if overlaps:
                    # If they overlap, one must be completely contained in the other
                    # (which shouldn't happen for different groups)
                    assert False, (
                        f"Groups {group_i} and {group_j} have overlapping spans: "
                        f"{group_spans[group_i]} and {group_spans[group_j]}"
                    )


class TestGroupOrdering:
    """
    Property 10: Group Ordering by Minimum Size

    For any two path groups G1 and G2 where G1 has a smaller minimum expand
    path size, all subtrees in G1 should be processed before any subtree in G2.

    Feature: expand-path-grouping, Property 10: Group Ordering by Minimum Size
    Validates: Requirements 6.1, 6.2
    """

    @given(expand_paths_data(min_subtrees=2, max_subtrees=6))
    @settings(max_examples=100)
    def test_smaller_groups_processed_first(self, data):
        """
        Property: Groups with smaller minimum path size are processed first.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        if manager.get_num_groups() < 2:
            return  # Need at least 2 groups to test ordering

        # Get the full processing order
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        if not order:
            return

        # Compute minimum path size for each group
        group_min_sizes: Dict[int, int] = {}
        for group_idx in range(manager.get_num_groups()):
            members = manager.get_group_members(group_idx)
            if members:
                min_size = min(manager.get_path_size(s) for s in members)
                group_min_sizes[group_idx] = min_size

        # Find first occurrence of each group in order
        group_first_pos: Dict[int, int] = {}
        for pos, subtree in enumerate(order):
            group = manager.get_group(subtree)
            if group is not None and group not in group_first_pos:
                group_first_pos[group] = pos

        # Verify: if group A has smaller min size than group B,
        # then A's first occurrence should be before B's first occurrence
        for group_a in group_first_pos:
            for group_b in group_first_pos:
                if group_a == group_b:
                    continue

                size_a = group_min_sizes.get(group_a, 0)
                size_b = group_min_sizes.get(group_b, 0)
                pos_a = group_first_pos[group_a]
                pos_b = group_first_pos[group_b]

                if size_a < size_b:
                    assert pos_a < pos_b, (
                        f"Group {group_a} (min size {size_a}) should be processed "
                        f"before group {group_b} (min size {size_b}), "
                        f"but positions are {pos_a} and {pos_b}"
                    )

    def test_explicit_group_ordering(self):
        """
        Property: Explicit test for group ordering with known data.
        """
        encoding = create_encoding(10)

        # Create three isolated subtrees with different path sizes
        # They'll form three singleton groups
        subtree_small = create_partition(frozenset({0}), encoding)
        subtree_medium = create_partition(frozenset({1}), encoding)
        subtree_large = create_partition(frozenset({2}), encoding)

        # Non-overlapping paths of different sizes
        split_s = create_partition(frozenset({3}), encoding)
        split_m1 = create_partition(frozenset({4}), encoding)
        split_m2 = create_partition(frozenset({5}), encoding)
        split_l1 = create_partition(frozenset({6}), encoding)
        split_l2 = create_partition(frozenset({7}), encoding)
        split_l3 = create_partition(frozenset({8}), encoding)

        expand_splits_by_subtree = {
            subtree_small: PartitionSet({split_s}, encoding=encoding),  # size 1
            subtree_medium: PartitionSet(
                {split_m1, split_m2}, encoding=encoding
            ),  # size 2
            subtree_large: PartitionSet(
                {split_l1, split_l2, split_l3}, encoding=encoding
            ),  # size 3
        }

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Should have 3 groups (all isolated)
        assert manager.get_num_groups() == 3, (
            f"Expected 3 groups, got {manager.get_num_groups()}"
        )

        # Get processing order
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        # Order should be: small, medium, large
        assert order == [subtree_small, subtree_medium, subtree_large], (
            f"Expected [small, medium, large] order, got {order}"
        )

    @given(expand_paths_data(min_subtrees=3, max_subtrees=6))
    @settings(max_examples=100)
    def test_group_order_respects_min_size(self, data):
        """
        Property: The sequence of groups in the output respects min-size ordering.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Get the full processing order
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        if not order:
            return

        # Extract the sequence of groups (in order of first appearance)
        group_sequence: List[int] = []
        seen_groups: Set[int] = set()

        for subtree in order:
            group = manager.get_group(subtree)
            if group is not None and group not in seen_groups:
                group_sequence.append(group)
                seen_groups.add(group)

        # Compute min sizes for each group in sequence
        min_sizes = []
        for group_idx in group_sequence:
            members = manager.get_group_members(group_idx)
            if members:
                min_size = min(manager.get_path_size(s) for s in members)
                min_sizes.append(min_size)

        # Verify min_sizes is non-decreasing
        for i in range(len(min_sizes) - 1):
            assert min_sizes[i] <= min_sizes[i + 1], (
                f"Group sequence min sizes should be non-decreasing: {min_sizes}"
            )


class TestCycleDetection:
    """
    Tests for cycle detection in containment graph.

    The containment graph should be acyclic (DAG) for valid tree paths.
    If a cycle is detected, the system should fall back to size-based ordering.

    Feature: expand-path-grouping, Cycle Detection
    Validates: Requirements 3.5
    """

    @given(expand_paths_data(min_subtrees=2, max_subtrees=5))
    @settings(max_examples=100)
    def test_no_cycles_in_valid_containment(self, data):
        """
        Property: Valid containment relationships should not form cycles.

        Since containment is based on proper subset relationships,
        cycles should be impossible with valid data.
        """
        expand_splits_by_subtree, encoding = data

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # The manager should have initialized successfully
        # (no cycle detected, or cycle handled gracefully)
        assert manager.enabled

        # Verify we can get a complete ordering
        processed: Set[Partition] = set()
        order: List[Partition] = []

        while True:
            next_subtree = manager.get_next_subtree(processed)
            if next_subtree is None:
                break
            order.append(next_subtree)
            processed.add(next_subtree)

        # All subtrees should be in the order
        assert len(order) == len(expand_splits_by_subtree), (
            f"Expected {len(expand_splits_by_subtree)} subtrees in order, got {len(order)}"
        )

    def test_cycle_detection_method_exists(self):
        """
        Verify that the _detect_cycle method exists and returns None for valid data.
        """
        encoding = create_encoding(10)

        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)

        # Simple containment: A ⊂ B
        split1 = create_partition(frozenset({2}), encoding)
        split2 = create_partition(frozenset({3}), encoding)

        expand_splits_by_subtree = {
            subtree_a: PartitionSet({split1}, encoding=encoding),
            subtree_b: PartitionSet({split1, split2}, encoding=encoding),
        }

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # _detect_cycle should return None for valid data
        cycle = manager._detect_cycle()
        assert cycle is None, f"Expected no cycle, got {cycle}"

    def test_graceful_handling_with_empty_input(self):
        """
        Verify graceful handling of empty input.
        """
        encoding = create_encoding(10)
        expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]] = {}

        manager = PathGroupManager(expand_splits_by_subtree, encoding)

        # Should handle empty input gracefully
        assert manager.get_num_groups() == 0
        assert manager.get_next_subtree(set()) is None

    def test_graceful_handling_when_disabled(self):
        """
        Verify graceful handling when manager is disabled.
        """
        encoding = create_encoding(10)

        subtree_a = create_partition(frozenset({0}), encoding)
        split1 = create_partition(frozenset({2}), encoding)

        expand_splits_by_subtree = {
            subtree_a: PartitionSet({split1}, encoding=encoding),
        }

        manager = PathGroupManager(expand_splits_by_subtree, encoding, enabled=False)

        # Should return None when disabled
        assert manager.get_next_subtree(set()) is None
        assert not manager.enabled


# ============================================================================
# Integration Tests with PivotSplitRegistry
# ============================================================================

from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
)


class TestSharedCollapsePriorityPreserved:
    """
    Property 11: Shared Collapse Priority Preserved

    For any subtree with shared collapse work, it should be selected before
    any subtree without shared collapse work (regardless of path grouping).

    Feature: expand-path-grouping, Property 11: Shared Collapse Priority Preserved
    Validates: Requirements 7.1
    """

    def test_shared_collapse_takes_priority_over_path_grouping(self):
        """
        Property: Subtrees with shared collapse work are selected first,
        even if path grouping would suggest a different order.
        """
        encoding = create_encoding(10)

        # Create subtrees
        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)
        subtree_c = create_partition(frozenset({2}), encoding)

        # Create splits
        shared_collapse = create_partition(frozenset({3, 4}), encoding)
        unique_collapse_a = create_partition(frozenset({5}), encoding)
        expand_a = create_partition(frozenset({6}), encoding)
        expand_b = create_partition(frozenset({7, 8}), encoding)  # Larger path
        expand_c = create_partition(frozenset({9}), encoding)  # Smallest path

        # Setup: subtree_a and subtree_b share a collapse split
        # subtree_c has the smallest expand path but no shared collapse
        all_collapse = PartitionSet(
            {shared_collapse, unique_collapse_a}, encoding=encoding
        )
        all_expand = PartitionSet({expand_a, expand_b, expand_c}, encoding=encoding)

        collapse_by_subtree = {
            subtree_a: PartitionSet(
                {shared_collapse, unique_collapse_a}, encoding=encoding
            ),
            subtree_b: PartitionSet({shared_collapse}, encoding=encoding),
            subtree_c: PartitionSet(encoding=encoding),  # No collapse
        }

        expand_by_subtree = {
            subtree_a: PartitionSet({expand_a}, encoding=encoding),
            subtree_b: PartitionSet({expand_b}, encoding=encoding),
            subtree_c: PartitionSet({expand_c}, encoding=encoding),
        }

        state = PivotSplitRegistry(
            all_collapse,
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            subtree_a,  # pivot_edge
            use_path_grouping=True,
        )

        # First subtree should be one with shared collapse (a or b), not c
        first = state.get_next_subtree()
        assert first in {subtree_a, subtree_b}, (
            f"Expected subtree with shared collapse (a or b), got {first}"
        )

    def test_path_grouping_used_when_no_shared_collapse(self):
        """
        Property: When no shared collapse work remains, path grouping is used.
        """
        encoding = create_encoding(10)

        # Create subtrees with no shared collapse
        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)

        # Create expand paths with containment: a's path ⊂ b's path
        split1 = create_partition(frozenset({2}), encoding)
        split2 = create_partition(frozenset({3}), encoding)

        all_collapse = PartitionSet(encoding=encoding)  # No collapse
        all_expand = PartitionSet({split1, split2}, encoding=encoding)

        collapse_by_subtree = {
            subtree_a: PartitionSet(encoding=encoding),
            subtree_b: PartitionSet(encoding=encoding),
        }

        expand_by_subtree = {
            subtree_a: PartitionSet({split1}, encoding=encoding),  # Smaller
            subtree_b: PartitionSet(
                {split1, split2}, encoding=encoding
            ),  # Larger, contains a
        }

        state = PivotSplitRegistry(
            all_collapse,
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            subtree_a,
            use_path_grouping=True,
        )

        # With path grouping, a should be selected first (smaller path, contained in b)
        first = state.get_next_subtree()
        assert first == subtree_a, (
            f"Expected subtree_a (smaller path) first, got {first}"
        )


class TestExpandLastPreserved:
    """
    Property 13: Expand-Last Preserved

    For any shared expand split, it should only be expanded when the
    last remaining user processes it.

    Feature: expand-path-grouping, Property 13: Expand-Last Preserved
    Validates: Requirements 7.4
    """

    def test_expand_last_returns_splits_for_last_user(self):
        """
        Property: Shared expand splits are only returned for the last user.
        """
        encoding = create_encoding(10)

        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)

        # Shared expand split
        shared_expand = create_partition(frozenset({2, 3}), encoding)
        unique_expand_a = create_partition(frozenset({4}), encoding)

        all_collapse = PartitionSet(encoding=encoding)
        all_expand = PartitionSet({shared_expand, unique_expand_a}, encoding=encoding)

        collapse_by_subtree = {
            subtree_a: PartitionSet(encoding=encoding),
            subtree_b: PartitionSet(encoding=encoding),
        }

        expand_by_subtree = {
            subtree_a: PartitionSet(
                {shared_expand, unique_expand_a}, encoding=encoding
            ),
            subtree_b: PartitionSet({shared_expand}, encoding=encoding),
        }

        state = PivotSplitRegistry(
            all_collapse,
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            subtree_a,
            use_path_grouping=True,
        )

        # For subtree_a (first user), shared_expand should NOT be in last-user splits
        # because subtree_b also uses it
        last_user_a = state.get_expand_splits_for_last_user(subtree_a)
        assert shared_expand not in last_user_a, (
            f"Shared expand should not be in last-user splits for first user"
        )
        assert unique_expand_a in last_user_a, (
            f"Unique expand should be in last-user splits"
        )

        # Process subtree_a
        state.mark_splits_as_processed(
            subtree_a,
            PartitionSet(encoding=encoding),  # No collapse
            PartitionSet({unique_expand_a}, encoding=encoding),
        )

        # Now subtree_b is the last user of shared_expand
        last_user_b = state.get_expand_splits_for_last_user(subtree_b)
        assert shared_expand in last_user_b, (
            "Shared expand should be in last-user splits for last user"
        )


class TestPathGroupingDisabled:
    """
    Tests for when path grouping is disabled.

    Feature: expand-path-grouping, Configuration
    Validates: Requirements 8.1, 8.2
    """

    def test_fallback_to_smallest_expand_when_disabled(self):
        """
        Property: When path grouping is disabled, fallback to smallest expand path.
        """
        encoding = create_encoding(10)

        subtree_a = create_partition(frozenset({0}), encoding)
        subtree_b = create_partition(frozenset({1}), encoding)

        # Different sized expand paths
        split1 = create_partition(frozenset({2}), encoding)
        split2 = create_partition(frozenset({3}), encoding)
        split3 = create_partition(frozenset({4}), encoding)

        all_collapse = PartitionSet(encoding=encoding)
        all_expand = PartitionSet({split1, split2, split3}, encoding=encoding)

        collapse_by_subtree = {
            subtree_a: PartitionSet(encoding=encoding),
            subtree_b: PartitionSet(encoding=encoding),
        }

        expand_by_subtree = {
            subtree_a: PartitionSet(
                {split1, split2, split3}, encoding=encoding
            ),  # size 3
            subtree_b: PartitionSet({split1}, encoding=encoding),  # size 1
        }

        state = PivotSplitRegistry(
            all_collapse,
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            subtree_a,
            use_path_grouping=False,  # Disabled
        )

        # Should select subtree_b (smallest expand path)
        first = state.get_next_subtree()
        assert first == subtree_b, (
            f"Expected subtree_b (smallest path) when grouping disabled, got {first}"
        )

    def test_path_group_manager_not_created_when_disabled(self):
        """
        Property: PathGroupManager is not created when disabled.
        """
        encoding = create_encoding(10)

        subtree_a = create_partition(frozenset({0}), encoding)
        split1 = create_partition(frozenset({2}), encoding)

        all_collapse = PartitionSet(encoding=encoding)
        all_expand = PartitionSet({split1}, encoding=encoding)

        collapse_by_subtree = {subtree_a: PartitionSet(encoding=encoding)}
        expand_by_subtree = {subtree_a: PartitionSet({split1}, encoding=encoding)}

        state = PivotSplitRegistry(
            all_collapse,
            all_expand,
            collapse_by_subtree,
            expand_by_subtree,
            subtree_a,
            use_path_grouping=False,
        )

        assert state._path_group_manager is None, (
            "PathGroupManager should not be created when disabled"
        )
