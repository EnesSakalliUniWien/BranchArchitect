# split.py
from typing import (
    Set,
    Tuple,
    Optional,
    Dict,
    TypeVar,
    Generic,
    Iterable,
    Iterator,
    List,
    TYPE_CHECKING,
)
from functools import total_ordering
from brancharchitect.elements.partition import Partition

if TYPE_CHECKING:
    from brancharchitect.elements.partition_set import PartitionSet

# Type variable for generic typing
T = TypeVar("T", bound="Partition")
# Define the generic type variable if needed globally in this file
T_Partition = TypeVar("T_Partition", bound="Partition")


@total_ordering
class FrozenPartitionSet(Generic[T]):
    """
    Immutable version of PartitionSet that can be used as dictionary keys.

    This class provides an immutable view of a set of Partitions with
    the same encoding. It supports set operations and comparisons.

    Attributes:
        _data: The underlying frozenset of Partitions
        _encoding: Mapping from string names to indices
        _reversed_encoding: Mapping from indices to string names
        _order: Optional ordering for the indices
        _name: Name of this partition set
    """

    _data: frozenset[T]  # type annotation for the class attribute

    def __init__(
        self,
        splits: Optional[Set[T]] = None,
        encoding: Optional[Dict[str, int]] = None,
        name: str = "FrozenPartitionSet",
        order: Optional[tuple[str, ...]] = None,
    ) -> None:
        """
        Initialize a new FrozenPartitionSet.

        Args:
            splits: Set of Partitions to include
            encoding: Mapping from string names to indices
            name: Name of this partition set
            order: Optional ordering for the indices
        """
        self._data = frozenset(splits) if splits else frozenset()
        self.taxa_encoding: Dict[str, int] = dict(encoding) if encoding else {}
        self._reversed_encoding: dict[int, str] = {
            v: k for k, v in self.taxa_encoding.items()
        }
        self._order = (
            order
            if order is not None
            else (tuple(self.taxa_encoding.values()) if self.taxa_encoding else None)
        )
        self._name: str = name

    def __contains__(self, x: object) -> bool:
        """Check if this partition set contains the given element."""
        return x in self._data

    def __iter__(self) -> Iterator[T]:
        """Iterate over the partitions in this set."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of partitions in this set."""
        return len(self._data)

    def __str__(self) -> str:
        """Return a string representation of this partition set."""
        splits_list = sorted(self._data, key=lambda s: s)
        return "\n".join(str(s) for s in splits_list)

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return str(sorted(self._data, key=lambda s: s))

    def __eq__(self, other: object) -> bool:
        """Check if this partition set equals another partition set."""
        if isinstance(other, FrozenPartitionSet):
            return self._data == other._data
        elif isinstance(other, PartitionSet):
            from typing import cast

            other_partitions = cast(Set[T], set(other._bitmask_to_partition.values()))
            return self._data == other_partitions
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Compare this partition set with another partition set."""
        if isinstance(other, FrozenPartitionSet):
            return sorted(self._data) < sorted(other._data)
        elif isinstance(other, PartitionSet):
            from typing import cast

            other_partitions = cast(list[T], list(other._bitmask_to_partition.values()))
            return sorted(self._data) < sorted(other_partitions)
        return NotImplemented

    def __hash__(self) -> int:
        """Return a hash of this partition set."""
        return hash((self._data, self._order))

    def issubset(self, other: Iterable[T]) -> bool:
        """Return True if all elements in self are also in other."""
        if isinstance(other, FrozenPartitionSet):
            return self._data.issubset(other._data)
        elif isinstance(other, PartitionSet):
            from typing import cast

            other_partitions = cast(Set[T], set(other._bitmask_to_partition.values()))
            return self._data.issubset(other_partitions)
        return self._data.issubset(other)

    def resolve_to_indices(self) -> List[Tuple[int, ...]]:
        """Return a list of tuples of indices for each partition."""
        return [s.indices for s in sorted(self._data, key=lambda p: sorted(p.indices))]
