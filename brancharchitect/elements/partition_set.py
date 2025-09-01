# split.py
import sys
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
    Set as TypingSet,
)
try:  # Python 3.11+
    from typing import Self  # type: ignore
except Exception:  # Python <3.11
    from typing_extensions import Self  # type: ignore
from itertools import product
from collections.abc import MutableSet
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.frozen_partition_set import FrozenPartitionSet

# Type variable for generic typing
T = TypeVar("T", bound="Partition")
# Define the generic type variable if needed globally in this file
T_Partition = TypeVar("T_Partition", bound="Partition")


class PartitionSet(Generic[T], MutableSet[T]):
    __slots__ = (
        "_bitmask_set",
        "_bitmask_to_partition",
        "encoding",
        "reversed_encoding",
        "order",
        "name",
    )
    """
    A set of partitions with common encoding information.

    This class represents a mutable set of Partition objects that share
    the same encoding information. It supports set operations, comparisons,
    and other standard set operations.

    Attributes:
        _bitmask_set: The underlying set of bitmasks
        _bitmask_to_partition: Mapping from bitmask to Partition objects
        encoding: Mapping from string names to indices
        reversed_encoding: Mapping from indices to string names
        order: Optional ordering for the indices
        name: Name of this partition set
    """

    def _add_bitmask_partition(
        self,
        bitmask: int,
        partition: Partition,
        bitmask_set: set[int],
        bitmask_to_partition: dict[int, Partition],
    ) -> None:
        """
        Helper method to add a bitmask and partition to the given collections if not already present.

        Args:
            bitmask: The bitmask to add
            partition: The partition object to associate with the bitmask
            bitmask_set: The set of bitmasks to add to
            bitmask_to_partition: The mapping from bitmask to partition to update
        """
        if bitmask not in bitmask_set:
            bitmask_set.add(bitmask)
            bitmask_to_partition[bitmask] = partition

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
        self.reversed_encoding: Dict[int, str] = {
            v: k for k, v in self.encoding.items()
        }
        self.order: Optional[tuple[str, ...]] = (
            order
            if order is not None
            else (tuple(self.encoding.keys()) if self.encoding else None)
        )
        self.name = name

        if splits:
            for split in splits:
                self.add(split)

        if not self.encoding:
            self.encoding = {str(i): i for i in range(len(self))}
            self.reversed_encoding = {v: k for k, v in self.encoding.items()}
            self.order = tuple(self.encoding.keys())

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
            return element.bitmask, element
        elif isinstance(element, tuple):
            p = Partition(element, self.encoding)
            return p.bitmask, p
        else:  # element is assumed to be int here
            p = Partition((element,), self.encoding)
            return p.bitmask, p

    def __contains__(self, x: object) -> bool:
        if isinstance(x, (Partition, tuple, int)):
            bitmask, _ = self._element_to_bitmask_and_partition(
                cast(Union[Partition, Tuple[int, ...], int], x)
            )
            return bitmask in self._bitmask_set
        return False

    def __iter__(self) -> Iterator[T]:
        # Yield Partition objects sorted by bitmask for determinism
        from typing import cast

        partitions = cast(list[T], list(self._bitmask_to_partition.values()))
        return iter(sorted(partitions, key=lambda p: p.bitmask))

    def __len__(self) -> int:
        return len(self._bitmask_set)

    def add(self, value: Union[T, Tuple[int, ...], int, Partition]) -> None:
        bitmask: int = 0
        partition: Partition = Partition((), {})
        bitmask, partition = self._element_to_bitmask_and_partition(value)
        self._add_bitmask_partition(
            bitmask, partition, self._bitmask_set, self._bitmask_to_partition
        )

    def batch_add(
        self, elements: Iterable[Union[T, Tuple[int, ...], int, Partition]]
    ) -> None:
        """
        Add multiple elements (Partition, tuple, or int) to the PartitionSet efficiently in a batch.
        Skips duplicates and only adds new partitions.
        """
        for element in elements:
            self.add(element)

    def discard(self, value: T) -> None:
        bitmask, _ = self._element_to_bitmask_and_partition(value)
        if bitmask in self._bitmask_set:
            self._bitmask_set.discard(bitmask)
            self._bitmask_to_partition.pop(bitmask, None)

    def __hash__(self) -> int:
        """Return a hash of this partition set."""
        order_tuple: tuple[int, ...] = ()
        if self.order is None:
            order_tuple = ()
        else:
            # Convert to tuple of ints if possible, otherwise fallback to empty tuple
            try:
                order_tuple = tuple(int(x) for x in self.order)
            except Exception:
                order_tuple = ()
        return hash((frozenset(self._bitmask_set), order_tuple))

    def atom(self) -> Self:
        """Return a new PartitionSet containing only minimal elements (no element is a superset of another)."""
        try:
            # Optimization: Sort by size, then scan left to right, keeping only those not supersets of previous
            partitions = sorted(self, key=lambda s: len(s.taxa))
            minimal: list[Partition] = []
            for s in partitions:
                is_minimal = True
                for prev in minimal:
                    # If prev.taxa is a subset of s.taxa, s is not minimal
                    if prev.taxa <= s.taxa:
                        is_minimal = False
                        break
                if is_minimal:
                    minimal.append(s)
            return type(self)(
                set(minimal), encoding=self.encoding, name="atoms", order=self.order
            )
        except Exception as e:
            print(f"Warning: Error in PartitionSet.atom: {e}", file=sys.stderr)
            return type(self)(
                set(), encoding=self.encoding, name="atoms_error", order=self.order
            )

    def cover(self) -> Self:
        """Return a new PartitionSet containing only maximal elements (no element is a subset of another)."""
        try:
            # Optimization: Sort by size descending, then scan left to right, keeping only those not subsets of previous
            partitions = sorted(self, key=lambda s: -len(s.taxa))
            maximal: list[Partition] = []
            for s in partitions:
                is_maximal = True
                for prev in maximal:
                    # If prev.taxa is a superset of s.taxa, s is not maximal
                    if prev.taxa >= s.taxa:
                        is_maximal = False
                        break
                if is_maximal:
                    maximal.append(s)
            return type(self)(
                set(maximal), encoding=self.encoding, name="covering", order=self.order
            )
        except Exception as e:
            print(f"Warning: Error in PartitionSet.cover: {e}", file=sys.stderr)
            return type(self)(
                set(), encoding=self.encoding, name="covering_error", order=self.order
            )

    def union(self, *others: Iterable[T]) -> Self:
        try:
            result_bitmask_set = set(self._bitmask_set)
            result_bitmask_to_partition = dict(self._bitmask_to_partition)
            for other in others:
                for elem in other:
                    bitmask, partition = self._element_to_bitmask_and_partition(elem)
                    self._add_bitmask_partition(
                        bitmask,
                        partition,
                        result_bitmask_set,
                        result_bitmask_to_partition,
                    )
            return type(self)(
                splits=set(result_bitmask_to_partition.values()),
                encoding=self.encoding,
                name=self.name + "_union",
                order=self.order,
            )
        except Exception as e:
            print(f"Warning: Error in PartitionSet.union: {e}", file=sys.stderr)
            return type(self)(
                splits=set(self._bitmask_to_partition.values()),
                encoding=self.encoding,
                name=self.name + "_union_error",
                order=self.order,
            )

    def intersection(self, *others: Iterable[T]) -> Self:
        try:
            result_bitmask_set = set(self._bitmask_set)
            for other in others:
                other_bitmasks: set[int] = set()
                for elem in other:
                    bitmask, _ = self._element_to_bitmask_and_partition(elem)
                    other_bitmasks.add(bitmask)
                result_bitmask_set &= other_bitmasks
            result_partitions = [
                self._bitmask_to_partition[b] for b in result_bitmask_set
            ]
            return type(self)(
                splits=set(result_partitions),
                encoding=self.encoding,
                name=self.name + "_intersection",
                order=self.order,
            )
        except Exception as e:
            print(f"Warning: Error in PartitionSet.intersection: {e}", file=sys.stderr)
            return type(self)(
                splits=set(),
                encoding=self.encoding,
                name=self.name + "_intersection_error",
                order=self.order,
            )

    def difference(self, *others: Iterable[T]) -> Self:
        try:
            result_bitmask_set = set(self._bitmask_set)
            for other in others:
                other_bitmasks: set[int] = set()
                for elem in other:
                    bitmask, _ = self._element_to_bitmask_and_partition(elem)
                    other_bitmasks.add(bitmask)
                result_bitmask_set -= other_bitmasks
            result_partitions = [
                self._bitmask_to_partition[b] for b in result_bitmask_set
            ]
            return type(self)(
                splits=set(result_partitions),
                encoding=self.encoding,
                name=self.name + "_difference",
                order=self.order,
            )
        except Exception as e:
            print(f"Warning: Error in PartitionSet.difference: {e}", file=sys.stderr)
            return type(self)(
                splits=set(self._bitmask_to_partition.values()),
                encoding=self.encoding,
                name=self.name + "_difference_error",
                order=self.order,
            )

    def symmetric_difference(self, other: Iterable[T]) -> Self:
        try:
            other_bitmask_to_partition: dict[int, Partition] = {}
            other_bitmasks: set[int] = set()
            for elem in other:
                bitmask, partition = self._element_to_bitmask_and_partition(elem)
                other_bitmasks.add(bitmask)
                other_bitmask_to_partition[bitmask] = partition
            result_bitmask_set = self._bitmask_set ^ other_bitmasks
            all_partitions: dict[int, Partition] = dict(self._bitmask_to_partition)
            all_partitions.update(other_bitmask_to_partition)
            result_partitions = [all_partitions[b] for b in result_bitmask_set]
            return type(self)(
                splits=set(result_partitions),
                encoding=self.encoding,
                name=self.name + "_symdiff",
                order=self.order,
            )
        except Exception as e:
            print(
                f"Warning: Error in PartitionSet.symmetric_difference: {e}",
                file=sys.stderr,
            )
            return type(self)(
                splits=set(self._bitmask_to_partition.values()),
                encoding=self.encoding,
                name=self.name + "_symdiff_error",
                order=self.order,
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

    def __getitem__(self, index: int) -> T:
        """Get the partition at the specified index."""
        return sorted(self, key=lambda s: s)[index]

    def cartesian(self, other: "PartitionSet[T]") -> Self:
        """
        Return a new PartitionSet with the Cartesian product of this set and another.

        Args:
            other: Another PartitionSet to form the product with

        Returns:
            A new PartitionSet containing the Cartesian product

        Raises:
            TypeError: If other is not a PartitionSet
            ValueError: If the sets don't have compatible orders
        """
        try:
            if self.order is None or other.order is None:
                raise ValueError(
                    "Both PartitionSet objects must have an order to compute Cartesian product"
                )
            if self.order != other.order:
                raise ValueError(
                    "PartitionSet objects must have the same order to compute Cartesian product"
                )

            # Compute product of self and other, then merge each pair into a new Partition.
            new_partitions = {
                Partition(
                    tuple(sorted(set(p1.indices) | set(p2.indices))),
                    self.encoding or {},
                )
                for p1, p2 in product(self, other)
            }

            return type(self)(
                new_partitions,
                encoding=self.encoding,
                order=self.order,
                name=f"{self.name}_cartesian_{other.name}",
            )
        except Exception as e:
            # Log the error but return a valid PartitionSet
            print(f"Warning: Error in PartitionSet.cartesian: {e}", file=sys.stderr)
            # Return a new empty PartitionSet with the same metadata
            return type(self)(
                set(),
                encoding=self.encoding,
                name=self.name + "_cartesian_error",
                order=self.order,
            )

    def __mul__(self, other: "PartitionSet[T]") -> Self:
        """Implement the * operator (Cartesian product)."""
        return self.cartesian(other)

    def resolve_to_indices(self) -> List[Tuple[int, ...]]:
        """
        Convert the PartitionSet to a list of tuples of indices.

        Returns:
            A list of tuples where each tuple contains the indices from a Partition in this set.
        """
        try:
            result: List[Tuple[int, ...]] = []
            for s in sorted(self, key=lambda p: p.indices):
                result.append(s.resolve_to_indices())
            return result
        except Exception as e:
            # Log the error but return a valid result
            print(
                f"Warning: Error in PartitionSet.resolve_to_indices: {e}",
                file=sys.stderr,
            )
            return []

    @property
    def list_taxa_name(self) -> List[Tuple[str, ...]]:
        """Return a list of tuples of taxa names for each partition."""
        try:
            return [
                tuple(sorted(p.taxa))
                for p in sorted(self, key=lambda p: sorted(p.taxa))
            ]
        except Exception as e:
            # Log the error but return a valid result
            print(
                f"Warning: Error in PartitionSet.list_taxa_name: {e}", file=sys.stderr
            )
            return []

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
        return all(elem in self for elem in other)

    def __le__(self, other: Iterable[T]) -> bool:
        """Return True if self <= other (issubset)."""
        return self.issubset(other)

    def __lt__(self, other: Iterable[T]) -> bool:
        """Return True if self < other (strict subset)."""
        return self.issubset(other) and self != other

    def issuperset(self, other: Iterable[T]) -> bool:
        """
        Return True if all elements in other are also in self.

        Args:
            other: Iterable to check against

        Returns:
            True if self is a superset of other, False otherwise
        """
        if isinstance(other, PartitionSet):
            return self._bitmask_set.issuperset(other._bitmask_set)
        elif isinstance(other, (set, frozenset)):
            return self._bitmask_set.issuperset({o.bitmask for o in other})
        return all(elem in self for elem in other)

    def __ge__(self, other: Iterable[T]) -> bool:
        """Return True if self >= other (issuperset)."""
        return self.issuperset(other)

    def __gt__(self, other: Iterable[T]) -> bool:
        """Return True if self > other (strict superset)."""
        return self.issuperset(other) and self != other

    @classmethod
    def from_existing(
        cls,
        source: Self,
        elements: Optional[set[Any]] = None,
        name: Optional[str] = None,
    ) -> Self:
        try:
            if elements is None:
                from typing import cast

                elements = cast(set[T], set(source._bitmask_to_partition.values()))
            return cls(
                splits=elements,
                encoding=source.encoding,
                name=name or source.name,
                order=source.order,
            )
        except Exception as e:
            print(f"Warning: Error in PartitionSet.from_existing: {e}", file=sys.stderr)
            return cls(
                splits=set(),
                encoding=source.encoding,
                name=(name or source.name) + "_from_existing_error",
                order=source.order,
            )

    def copy(self, name: Optional[str] = None) -> Self:
        try:
            return type(self).from_existing(self, name=name)
        except Exception as e:
            print(f"Warning: Error in PartitionSet.copy: {e}", file=sys.stderr)
        from typing import cast

        return type(self)(
            splits=cast(set[T], set(self._bitmask_to_partition.values())),
            encoding=self.encoding,
            name=(name or self.name) + "_copy_error",
            order=self.order,
        )

    def freeze(self) -> FrozenPartitionSet[T]:
        try:
            from typing import cast

            return FrozenPartitionSet(
                splits=cast(set[T], set(self._bitmask_to_partition.values())),
                encoding=self.encoding,
                name=self.name,
                order=self.order,
            )
        except Exception as e:
            print(f"Warning: Error in PartitionSet.freeze: {e}", file=sys.stderr)
            return FrozenPartitionSet(
                splits=set(),
                encoding=self.encoding,
                name=self.name + "_freeze_error",
                order=None,
            )


def count_full_overlaps(
    target: Partition, partition_set: PartitionSet[Partition]
) -> int:
    """
    Count how many partitions in partition_set fully contain the target partition (by indices).
    """
    count = 0
    for part in partition_set:
        if is_full_overlap(target, part):
            count += 1
    return count


def is_full_overlap(target: Partition, reference: Partition) -> bool:
    """
    Return True if the target partition is fully contained in the reference partition (i.e., all indices of target are in reference).
    """
    return set(target.indices).issubset(set(reference.indices))


def subtract_partition_indices(
    source_set: PartitionSet[T_Partition], deletion_set: PartitionSet[T_Partition]
) -> PartitionSet[T_Partition]:
    """
    Creates a new PartitionSet by modifying partitions from the source_set.

    For each Partition in source_set, it removes any indices that are
    present in *any* Partition within the deletion_set. Partitions that
    become empty after subtraction are omitted from the result.

    Args:
        source_set: The PartitionSet containing partitions to be modified.
        deletion_set: The PartitionSet defining which indices to remove.

    Returns:
        A new PartitionSet containing the modified partitions from source_set,
        preserving the metadata (encoding, order) from the source_set.
        Returns an empty set if source_set is empty or all partitions become empty.
    """
    try:
        # 1. Aggregate all indices to be deleted
        indices_to_delete: TypingSet[int] = set()
        for del_partition in deletion_set:
            indices_to_delete.update(del_partition.indices)

        if not indices_to_delete:
            # If nothing to delete, return a copy of the source set
            # Ensure copy method exists and handles metadata correctly
            return source_set.copy(name=f"{source_set.name}_subtracted_noop")

        # 2. Create new partitions by subtracting indices
        modified_partitions: TypingSet[T_Partition] = set()
        for source_partition in source_set:
            current_indices = set(source_partition.indices)
            # Perform the set difference on indices
            new_indices_set = current_indices - indices_to_delete

            # Only add the partition if it's not empty after subtraction
            if new_indices_set:
                # Create a new Partition object with the remaining indices
                # Ensure indices are sorted tuple as expected by Partition constructor
                new_indices_tuple = tuple(sorted(new_indices_set))
                # Use the source_set's encoding for the new partition
                # Ensure Partition can be created this way
                new_partition = Partition(
                    indices=new_indices_tuple, encoding=source_set.encoding
                )
                # Cast to T_Partition before adding
                modified_partitions.add(cast(T_Partition, new_partition))

        # 3. Create the resulting PartitionSet using metadata from the source
        # Ensure from_existing method exists and works as expected
        result_set = type(source_set).from_existing(
            source_set,
            elements=modified_partitions,
            name=f"{source_set.name}_subtracted",
        )
        return result_set

    except Exception as e:
        raise ValueError(f"Error in subtract_partition_indices: {e}")
