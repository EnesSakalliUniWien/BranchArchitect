# partition_set.py
from typing import (
    Set,
    Tuple,
    Optional,
    Dict,
    TypeVar,
    Generic,
    Iterable,
    cast,
    Iterator,
    List,
    Any,
    Union,
)

try:
    from typing import Self  # Python 3.11+
except Exception:  # pragma: no cover - fallback for Python < 3.11
    from typing_extensions import Self
from collections.abc import MutableSet
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_algorithms import (
    solve_minimum_set_cover,
    compute_geometric_intersection,
)

# Type variable for generic typing
T = TypeVar("T", bound="Partition")


class PartitionSet(Generic[T], MutableSet[T]):
    """
    A set of partitions with common encoding information.

    This class represents a mutable set of Partition objects that share
    the same encoding information. It supports set operations, comparisons,
    and other standard set operations.

    Attributes:
        _bitmask_set: The underlying set of bitmasks
        _bitmask_to_partition: Mapping from bitmask to Partition objects
        encoding: Mapping from string names to indices
        _reversed_encoding: Lazily computed mapping from indices to string names
        order: Optional ordering for the indices
        name: Name of this partition set
    """

    __slots__ = (
        "_bitmask_set",
        "_bitmask_to_partition",
        "encoding",
        "_reversed_encoding",
        "order",
        "name",
    )

    @property
    def reversed_encoding(self) -> Dict[int, str]:
        """Lazily compute and cache the reversed encoding."""
        if self._reversed_encoding is None:
            self._reversed_encoding = {v: k for k, v in self.encoding.items()}
        return self._reversed_encoding

    @classmethod
    def _from_iterable(cls, it):
        """Create a new PartitionSet from an iterable. Used by ABC set operations."""
        return cls(splits=set(it))

    def __init__(
        self,
        splits: Optional[Union[Set[T], Set[Partition]]] = None,
        encoding: Optional[Dict[str, int]] = None,
        name: str = "PartitionSet",
        order: Optional[tuple[str, ...]] = None,
    ) -> None:
        self._bitmask_set: set[int] = set()
        self._bitmask_to_partition: dict[int, Partition] = {}
        self.encoding: Dict[str, int] = encoding or {}
        self._reversed_encoding: Optional[Dict[int, str]] = None  # Lazily computed
        self.order: Optional[tuple[str, ...]] = (
            order
            if order is not None
            else (tuple(self.encoding.keys()) if self.encoding else None)
        )
        self.name = name

        if splits:
            for split in splits:
                self.add(split)

    def _element_to_bitmask_and_partition(
        self, element: Union[Partition, Tuple[int, ...], int]
    ) -> Tuple[int, Partition]:
        """
        Convert a Partition, tuple, or int element to its bitmask and Partition object.

        Args:
            element: A Partition, tuple of ints, or single int

        Returns:
            A tuple of (bitmask, partition_object)

        Raises:
            TypeError: If element is not a supported type
        """
        if isinstance(element, Partition):
            # Fast path: skip encoding check if same object or both empty
            if (
                self.encoding is not element.encoding
                and self.encoding
                and element.encoding
            ):
                # Only do expensive dict comparison if identity check fails
                if self.encoding != element.encoding:
                    self_keys = sorted(self.encoding.keys())
                    elem_keys = sorted(element.encoding.keys())
                    raise ValueError(
                        f"Cannot add Partition with different encoding to PartitionSet.\n"
                        f"PartitionSet encoding keys: {self_keys}\n"
                        f"Partition encoding keys: {elem_keys}\n"
                        f"PartitionSet encoding: {self.encoding}\n"
                        f"Partition encoding: {element.encoding}"
                    )
            return element.bitmask, element
        elif isinstance(element, tuple):
            p = Partition(element, self.encoding)
            return p.bitmask, p
        else:  # element is assumed to be int here
            p = Partition((element,), self.encoding)
            return p.bitmask, p

    def _create_new_from_bitmasks(
        self,
        bitmask_set: set[int],
        bitmask_to_partition: dict[int, Partition],
        suffix: str,
    ) -> Self:
        """Helper to create a new instance with shared metadata."""
        new_set = type(self).__new__(type(self))
        new_set._bitmask_set = bitmask_set
        new_set._bitmask_to_partition = bitmask_to_partition
        new_set.encoding = self.encoding
        new_set._reversed_encoding = self._reversed_encoding
        new_set.order = self.order
        new_set.name = f"{self.name}_{suffix}"
        return new_set

    @property
    def fast_partitions(self) -> Iterable[T]:
        """Return the partitions in this set without sorting. Useful for performance-sensitive loops."""
        return cast(Iterable[T], self._bitmask_to_partition.values())

    def __contains__(self, x: object) -> bool:
        # Fast path for Partition (most common case) - direct bitmask lookup
        if isinstance(x, Partition):
            return x.bitmask in self._bitmask_set
        elif isinstance(x, (tuple, int)):
            try:
                bitmask, _ = self._element_to_bitmask_and_partition(
                    cast(Union[Partition, Tuple[int, ...], int], x)
                )
                return bitmask in self._bitmask_set
            except ValueError:
                # If encoding doesn't match, element is not in this set
                raise
        return False

    def __iter__(self) -> Iterator[T]:
        # Yield Partition objects directly from internal storage.
        # Note: This is NOT sorted. If you need deterministic order,
        # you must sort the result explicitly (e.g. sorted(partition_set)).
        # This optimization avoids O(N log N) sorting on every iteration.
        return cast(Iterator[T], iter(self._bitmask_to_partition.values()))

    def __len__(self) -> int:
        return len(self._bitmask_set)

    def add(self, value: Union[T, Tuple[int, ...], int, Partition]) -> None:
        # If we have no encoding yet and value is a Partition with encoding, inherit it
        if not self.encoding and isinstance(value, Partition) and value.encoding:
            self.encoding = value.encoding
            self._reversed_encoding = None  # Invalidate cache
            self.order = tuple(self.encoding.keys())

        bitmask, partition = self._element_to_bitmask_and_partition(value)
        if bitmask not in self._bitmask_set:
            self._bitmask_set.add(bitmask)
            self._bitmask_to_partition[bitmask] = partition

    def update(
        self, others: Iterable[Union[T, Tuple[int, ...], int, Partition]]
    ) -> None:
        """Add multiple elements at once."""
        # Local references for speed
        bitmask_set = self._bitmask_set
        bitmask_to_partition = self._bitmask_to_partition

        for value in others:
            # Inherit encoding if empty
            if not self.encoding and isinstance(value, Partition) and value.encoding:
                self.encoding = value.encoding
                self._reversed_encoding = None  # Invalidate cache
                self.order = tuple(self.encoding.keys())

            bitmask, partition = self._element_to_bitmask_and_partition(value)
            if bitmask not in bitmask_set:
                bitmask_set.add(bitmask)
                bitmask_to_partition[bitmask] = partition

    def discard(self, value: T) -> None:
        bitmask, _ = self._element_to_bitmask_and_partition(value)
        if bitmask in self._bitmask_set:
            self._bitmask_set.discard(bitmask)
            self._bitmask_to_partition.pop(bitmask, None)

    def __hash__(self) -> int:
        """
        Return a hash of this partition set based on its content only.
        The hash is computed on a frozenset of the bitmasks of the partitions
        it contains, making it independent of the `order` attribute.
        """
        return hash(frozenset(self._bitmask_set))

    def minimal_elements(self) -> Self:
        """
        Return minimal elements under subset order (no element is a superset of another).
        """
        parts = list(self.fast_partitions)
        # Sort ascending by set size (cached), then by bitmask for determinism
        parts.sort(key=lambda p: (p.size, p.bitmask))
        kept: list[Partition] = []
        kept_masks: list[int] = []
        for s in parts:
            s_mask = s.bitmask
            is_minimal = True
            for km in kept_masks:
                # km ⊆ s  <=>  (km & ~s) == 0
                if (km & ~s_mask) == 0:
                    is_minimal = False
                    break
            if is_minimal:
                kept.append(s)
                kept_masks.append(s_mask)
        return type(self)(
            set(kept),
            encoding=self.encoding,
            name="minimal_elements",
            order=self.order,
        )

    def maximal_elements(self) -> Self:
        """
        Return maximal elements under subset order (no element is a subset of another).
        """
        parts = list(self.fast_partitions)
        # Sort descending by set size (cached), then by bitmask for determinism
        parts.sort(key=lambda p: (-p.size, p.bitmask))
        kept: list[Partition] = []
        kept_masks: list[int] = []
        for s in parts:
            s_mask = s.bitmask
            is_maximal = True
            for km in kept_masks:
                # s ⊆ km  <=>  (s & ~km) == 0
                if (s_mask & ~km) == 0:
                    is_maximal = False
                    break
            if is_maximal:
                kept.append(s)
                kept_masks.append(s_mask)
        return type(self)(
            set(kept),
            encoding=self.encoding,
            name="maximal_elements",
            order=self.order,
        )

    def bottoms(self, min_size: int = 1) -> Self:
        """
        Return the bottoms of this set (its minimal elements / antichain bottoms),
        optionally filtering by size.

        Args:
            min_size: Keep only bottoms whose cardinality (number of indices) >= min_size.
                      Use 1 (default) to return all minimal elements. Use 2 to exclude
                      singletons (e.g., require cherries or larger).

        Returns:
            A PartitionSet containing the bottoms (minimal elements), filtered by size.
        """
        mins = self.minimal_elements()
        if min_size <= 1:
            return mins
        filtered = {p for p in mins if len(p) >= min_size}
        return type(self)(
            splits=filtered,
            encoding=self.encoding,
            name="bottoms",
            order=self.order,
        )

    def bottoms_under(
        self, upper: Union[Partition, Tuple[int, ...], int], min_size: int = 1
    ) -> Self:
        """
        Return the bottoms (minimal elements) of the downset under ``upper`` within this set.

        Computes the downset D = { s in self | s ⊆ upper } and returns
        the minimal elements of D. Optionally filters by cardinality.

        Args:
            upper: A Partition (or tuple/int convertible to Partition) defining the upper bound.
            min_size: Keep only bottoms whose size >= min_size (use 2 to exclude singletons).

        Returns:
            A PartitionSet containing the bottoms of the antichain under ``upper``.
        """
        upper_mask, _ = self._element_to_bitmask_and_partition(upper)
        # Downset under upper: all elements s with s ⊆ upper
        elems = {p for p in self.fast_partitions if (p.bitmask & ~upper_mask) == 0}
        down = type(self)(
            splits=elems,
            encoding=self.encoding,
            name=f"{self.name}_downset",
            order=self.order,
        )
        mins = down.minimal_elements()
        if min_size <= 1:
            return mins
        filtered = {p for p in mins if len(p) >= min_size}
        return type(self)(
            splits=filtered,
            encoding=self.encoding,
            name="bottoms_under",
            order=self.order,
        )

    def maximals_under(self, upper: Union[Partition, Tuple[int, ...], int]) -> Self:
        """
        Return the maximal elements of the downset under ``upper`` within this set.

        Computes the downset D = { s in self | s ⊆ upper } and returns
        the maximal elements of D.

        Args:
            upper: A Partition (or tuple/int convertible to Partition) defining the upper bound.

        Returns:
            A PartitionSet containing the maximal elements of the downset.
        """
        upper_mask, _ = self._element_to_bitmask_and_partition(upper)

        # 1. Filter elements that are subsets of upper
        # s ⊆ upper <=> (s & ~upper) == 0
        candidates = [p for p in self.fast_partitions if (p.bitmask & ~upper_mask) == 0]

        # 2. Find maximals among candidates
        # Sort descending by size, then bitmask
        candidates.sort(key=lambda p: (-p.size, p.bitmask))

        kept: list[Partition] = []
        kept_masks: list[int] = []

        for s in candidates:
            s_mask = s.bitmask
            is_maximal = True
            for km in kept_masks:
                # s ⊆ km <=> (s & ~km) == 0
                if (s_mask & ~km) == 0:
                    is_maximal = False
                    break
            if is_maximal:
                kept.append(s)
                kept_masks.append(s_mask)

        return type(self)(
            set(kept),
            encoding=self.encoding,
            name="maximals_under",
            order=self.order,
        )

    def covers(self, partition: Union[Partition, Tuple[int, ...], int]) -> bool:
        """
        Check if any element in this PartitionSet covers (contains as subset) the given partition.

        MATHEMATICAL DEFINITION:
            Returns True iff ∃s ∈ self: partition ⊆ s
            where ⊆ is the subset relation on partitions (clades)

        BITWISE IMPLEMENTATION:
            partition ⊆ s ⟺ (partition.bitmask & s.bitmask) == partition.bitmask
            This checks if all taxa in partition are also in s.

        PHYLOGENETIC INTERPRETATION:
            Returns True if the clade represented by partition is nested within
            (descendant of) at least one clade in this PartitionSet.

        Args:
            partition: The partition (clade) to check for coverage.
                      Can be a Partition object, tuple of indices, or single int.

        Returns:
            True if partition is a subset of at least one element in this set
            False if partition is not covered by any element in this set

        Example:
            partition_set = PartitionSet({(0, 1), (2, 3, 4)})  # {(A1,X), (B1,B2,B3)}
            partition_set.covers((0,))  # True, because (A1) ⊆ (A1,X)
            partition_set.covers((0, 2))  # False, not subset of any element
        """
        partition_mask, _ = self._element_to_bitmask_and_partition(partition)
        # Check if partition is subset of any element - iterate over bitmasks directly
        # This avoids attribute access overhead compared to iterating over Partition objects
        return any(
            (partition_mask & mask) == partition_mask for mask in self._bitmask_set
        )

    def union(self, *others: Iterable[T]) -> Self:
        """Optimized union using direct bitmask operations."""
        # Start with copies of our own data structures
        result_bitmask_set = set(self._bitmask_set)
        result_bitmask_to_partition = dict(self._bitmask_to_partition)

        for other in others:
            if isinstance(other, PartitionSet):
                # Fast path: directly merge bitmask sets for PartitionSet
                result_bitmask_set |= other._bitmask_set
                result_bitmask_to_partition.update(other._bitmask_to_partition)
            else:
                # Slower path for generic iterables
                for elem in other:
                    bitmask = elem.bitmask
                    if bitmask not in result_bitmask_set:
                        result_bitmask_set.add(bitmask)
                        result_bitmask_to_partition[bitmask] = elem

        # Create new PartitionSet without re-processing partitions
        return self._create_new_from_bitmasks(
            result_bitmask_set, result_bitmask_to_partition, "_union"
        )

    def intersection(self, *others: Iterable[T]) -> Self:
        """Optimized intersection using direct bitmask operations."""
        result_bitmask_set = set(self._bitmask_set)

        for other in others:
            if isinstance(other, PartitionSet):
                # Fast path: directly intersect bitmask sets
                result_bitmask_set &= other._bitmask_set
            else:
                # Create bitmask set for other iterable
                other_bitmasks = {elem.bitmask for elem in other}
                result_bitmask_set &= other_bitmasks

        # Create new PartitionSet with only the intersecting partitions
        return self._create_new_from_bitmasks(
            result_bitmask_set,
            {b: self._bitmask_to_partition[b] for b in result_bitmask_set},
            "_intersection",
        )

    def difference(self, *others: Iterable[T]) -> Self:
        """Optimized difference using direct bitmask operations."""
        result_bitmask_set = set(self._bitmask_set)

        for other in others:
            if isinstance(other, PartitionSet):
                # Fast path: directly subtract bitmask sets
                result_bitmask_set -= other._bitmask_set
            else:
                # Create bitmask set for other iterable
                other_bitmasks = {elem.bitmask for elem in other}
                result_bitmask_set -= other_bitmasks

        # Create new PartitionSet with remaining partitions
        return self._create_new_from_bitmasks(
            result_bitmask_set,
            {b: self._bitmask_to_partition[b] for b in result_bitmask_set},
            "_difference",
        )

    def symmetric_difference(self, other: Iterable[T]) -> Self:
        """Optimized symmetric difference using direct bitmask operations."""
        if isinstance(other, PartitionSet):
            # Fast path for PartitionSet
            result_bitmask_set: set[int] = self._bitmask_set ^ other._bitmask_set

            # Merge partition mappings
            result_bitmask_to_partition: dict[int, Partition] = {}
            for b in result_bitmask_set:
                if b in self._bitmask_to_partition:
                    result_bitmask_to_partition[b] = self._bitmask_to_partition[b]
                else:
                    result_bitmask_to_partition[b] = other._bitmask_to_partition[b]
        else:
            # Handle generic iterable
            other_bitmask_to_partition: dict[int, Partition] = {}
            other_bitmasks: set[int] = set()
            for elem in other:
                other_bitmasks.add(elem.bitmask)
                other_bitmask_to_partition[elem.bitmask] = elem

            result_bitmask_set = self._bitmask_set ^ other_bitmasks

            # Merge partition mappings
            result_bitmask_to_partition = {}
            for b in result_bitmask_set:
                if b in self._bitmask_to_partition:
                    result_bitmask_to_partition[b] = self._bitmask_to_partition[b]
                else:
                    result_bitmask_to_partition[b] = other_bitmask_to_partition[b]

        # Create new PartitionSet
        return self._create_new_from_bitmasks(
            result_bitmask_set, result_bitmask_to_partition, "_symdiff"
        )

    def __or__(self, other: object) -> Self:
        """Implement the | operator (union)."""
        if isinstance(other, Iterable):
            return self.union(cast(Iterable[T], other))
        return NotImplemented

    def __and__(self, other: object) -> Self:
        """Implement the & operator (intersection)."""
        if isinstance(other, Iterable):
            return self.intersection(cast(Iterable[T], other))
        return NotImplemented

    def __sub__(self, other: object) -> Self:
        """Implement the - operator (difference)."""
        if isinstance(other, Iterable):
            return self.difference(cast(Iterable[T], other))
        return NotImplemented

    def __xor__(self, other: object) -> Self:
        """Implement the ^ operator (symmetric_difference)."""
        if isinstance(other, Iterable):
            return self.symmetric_difference(cast(Iterable[T], other))
        return NotImplemented

    def __str__(self) -> str:
        """Return a string representation of this partition set."""
        splits_list = sorted(self, key=lambda s: s)
        return "\n".join(str(s) for s in splits_list)

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return str(sorted(self, key=lambda s: s))

    def resolve_to_indices(self) -> List[List[int]]:
        """
        Convert the PartitionSet to a list of lists of indices for readability in logs.

        Returns:
            A list where each element is a list of ints corresponding to
            the indices from a Partition in this set.
        """
        result: List[List[int]] = []
        for s in sorted(self, key=lambda p: p.indices):
            # Use a list instead of a tuple so singletons render as [i]
            # instead of the tuple form (i,), improving diagnostic readability.
            result.append(list(s.resolve_to_indices()))
        return result

    def issubset(self, other: Iterable[T]) -> bool:
        """
        Return True if all elements in self are also in other.

        Args:
            other: Iterable to check against

        Returns:
            True if self is a subset of other, False otherwise
        """
        if isinstance(other, PartitionSet):
            return self._bitmask_set.issubset(other._bitmask_set)
        elif isinstance(other, (set, frozenset)):
            return self._bitmask_set.issubset({p.bitmask for p in other})
        # Fallback: every element of self must be present in `other`
        return all(elem in other for elem in self)

    def __le__(self, other: Iterable[T]) -> bool:
        """Return True if self <= other (issubset)."""
        return self.issubset(other)

    def __lt__(self, other: Iterable[T]) -> bool:
        """Return True if self < other (strict subset)."""
        return self.issubset(other) and self != other

    def geometric_intersection(self, other: "PartitionSet[T]") -> "PartitionSet[T]":
        """
        Compute the **Geometric Intersection** (All-Pairs Interaction).
        Synthesize new common substructures by intersecting every cluster.

        If P1 = {{A,B}}, P2 = {{B,C}}:
            P1 & P2 (Set Logic) -> {}
            geometric_intersection(P1, P2) -> {{B}}
        """
        result_bitmasks, result_partitions = compute_geometric_intersection(
            self._bitmask_to_partition, other._bitmask_to_partition, self.encoding
        )

        # Construct new PartitionSet manually to avoid re-validation
        return self._create_new_from_bitmasks(
            result_bitmasks, result_partitions, f"_geometric_intersection_{other.name}"
        )

    @classmethod
    def from_existing(
        cls,
        source: Self,
        elements: Optional[set[Any]] = None,
        name: Optional[str] = None,
    ) -> Self:
        if elements is None:
            from typing import cast

            elements = cast(set[T], set(source._bitmask_to_partition.values()))
        return cls(
            splits=elements,
            encoding=source.encoding,
            name=name or source.name,
            order=source.order,
        )

    def copy(self, name: Optional[str] = None) -> Self:
        return self._create_new_from_bitmasks(
            set(self._bitmask_set),
            dict(self._bitmask_to_partition),
            name or self.name,
        )

    def to_singleton_partition_sets(self) -> List["PartitionSet[T]"]:
        """
        Transform each partition in this PartitionSet into its own individual PartitionSet.

        Returns a list where each element is a PartitionSet containing exactly one
        partition from the original set. This is useful for processing partitions
        independently while preserving their encoding and metadata.

        MATHEMATICAL DEFINITION:
            Given PartitionSet S = {p₁, p₂, ..., pₙ}
            Returns [PartitionSet({p₁}), PartitionSet({p₂}), ..., PartitionSet({pₙ})]

        Returns:
            List[PartitionSet[T]]: A list of singleton PartitionSets, one for each
                                   partition in the original set.

        Example:
            >>> ps = PartitionSet({(0, 1), (2, 3), (4,)})
            >>> singletons = ps.to_singleton_partition_sets()
            >>> len(singletons)
            3
            >>> len(singletons[0])  # Each PartitionSet has exactly 1 element
            1
        """
        result: List[PartitionSet[T]] = []
        for partition in sorted(self, key=lambda p: p.bitmask):
            singleton = type(self)(
                splits={partition},
                encoding=self.encoding,
                name=f"{self.name}_singleton",
                order=self.order,
            )
            result.append(singleton)
        return result

    # ----------------------------------------------------------------------------
    # Minimum (cardinality) union cover utilities
    # ----------------------------------------------------------------------------
    def minimum_cover(self) -> "PartitionSet[T]":
        """
        Compute a minimum-cardinality union cover (minimum set cover) of this PartitionSet.

        Finds a subset C ⊆ self with the fewest elements such that
        union(C) == union(self). This uses an exact branch-and-bound search with
        greedy lower-bound pruning. It is deterministic for a fixed self.

        Returns:
            PartitionSet[T]: a new PartitionSet containing one minimum cover.
        """
        elems = solve_minimum_set_cover(self.fast_partitions)
        return type(self).from_existing(
            self, elements=elems, name=f"{self.name}_minimum_cover"
        )


def is_full_overlap(target: Partition, reference: Partition) -> bool:
    """
    Return True if the target partition is fully contained in the reference partition (i.e., all indices of target are in reference).
    Uses bitmask operations for O(1) performance.
    """
    return (target.bitmask & reference.bitmask) == target.bitmask
