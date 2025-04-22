# split.py
from typing import Tuple, FrozenSet, Dict, Iterator, List, Any
from dataclasses import field
from functools import total_ordering
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
@total_ordering
class Partition:
    """
    Represents a partition of elements identified by indices.

    A Partition is an immutable collection of indices with optional string lookups.
    It supports set operations and comparisons with other Partitions.

    Attributes:
        indices: A tuple of indices representing the elements in this partition
        lookup: Optional mapping from string names to indices
    """

    indices: Tuple[int, ...]
    lookup: Dict[str, int] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        """Ensures indices are stored as a sorted tuple."""
        if isinstance(self.indices, int):
            object.__setattr__(self, "indices", (self.indices,))
        else:
            object.__setattr__(self, "indices", tuple(sorted(self.indices)))

    def __iter__(self) -> Iterator[int]:
        """Iterate over the indices in this partition."""
        return iter(self.indices)

    def __len__(self) -> int:
        """Return the number of indices in this partition."""
        return len(self.indices)

    def __lt__(self, other: Any) -> bool:
        """Compare this partition with another partition or tuple."""
        if isinstance(other, Partition):
            return self.indices < other.indices
        elif isinstance(other, tuple):
            if all(isinstance(x, str) for x in other):
                try:
                    other_indices = tuple(sorted(self.lookup[x] for x in other))
                    return self.indices < other_indices
                except (KeyError, AttributeError):
                    return False
            return self.indices < tuple(sorted(other))
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        """Check if this partition equals another partition or tuple."""
        if isinstance(other, Partition):
            return self.indices == other.indices
        elif isinstance(other, tuple):
            if all(isinstance(x, str) for x in other):
                try:
                    other_indices = tuple(sorted(self.lookup[x] for x in other))
                    return self.indices == other_indices
                except (KeyError, AttributeError):
                    return False
            return self.indices == tuple(sorted(other))
        return NotImplemented

    def __getitem__(self, index: int) -> int:
        """Get the index at the specified position."""
        return self.indices[index]

    @property
    def taxa(self) -> FrozenSet[str]:
        """Get the taxa names corresponding to the indices in this partition."""
        return frozenset(self.reverse_lookup[i] for i in self.indices)

    def bipartition(self) -> str:
        """Return a string representation of this partition as a bipartition."""
        left = sorted((self.reverse_lookup[i] for i in self.indices), key=len)
        right = sorted(self.reverse_lookup[i] for i in self.complementary_indices())
        return f"{', '.join(left)} | {', '.join(right)}"

    def complementary_indices(self) -> Tuple[int, ...]:
        """Return the indices not in this partition but in the lookup."""
        full_set = set(self.reverse_lookup.keys())
        return tuple(sorted(full_set - set(self.indices)))

    def __str__(self) -> str:
        """Return a string representation of this partition."""
        try:
            taxa_names = sorted(
                self.reverse_lookup.get(i, str(i)) for i in self.indices
            )
            return f"({', '.join(taxa_names)})"
        except Exception:
            return f"{tuple(sorted(self.indices))}"

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"({self})"

    def __hash__(self) -> int:
        """Return a hash of this partition."""
        return hash(self.indices)

    @property
    def reverse_lookup(self) -> Dict[int, str]:
        """Get a reverse mapping from indices to taxa names."""
        return {v: k for k, v in self.lookup.items()}

    def __json__(self) -> List[int]:
        """Return a JSON-serializable representation of this partition."""
        return list(self.indices)

    def to_dict(self) -> Dict[str, Tuple[int, ...]]:
        """Return a dictionary representation of this partition."""
        return {"indices": self.indices}

    def copy(self) -> "Partition":
        """Return a copy of this partition."""
        return Partition(self.indices, self.lookup)

    def __iand__(self, other: Any) -> "Partition":
        """Implement the &= operator (in-place intersection)."""
        if isinstance(other, Partition):
            common_indices = set(self.indices) & set(other.indices)
            object.__setattr__(self, "indices", tuple(sorted(common_indices)))
            return self
        return NotImplemented

    def __and__(self, other: Any) -> "Partition":
        """Implement the & operator (intersection)."""
        if isinstance(other, Partition):
            common_indices = set(self.indices) & set(other.indices)
            return Partition(tuple(sorted(common_indices)), self.lookup)
        return NotImplemented

    def resolve_to_indices(self) -> Tuple[int, ...]:
        """Return the indices of this partition as a tuple."""
        return tuple(self.indices)
