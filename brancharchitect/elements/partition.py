# split.py
from typing import Tuple, FrozenSet, Dict, Iterator, List, Any, Optional
from functools import total_ordering


@total_ordering
class Partition:
    __slots__ = ("indices", "encoding", "bitmask", "_cached_reverse_encoding")

    def __init__(
        self, indices: Tuple[int, ...], encoding: Optional[Dict[str, int]] = None
    ):
        """
        Partition represents a subset of taxa as a tuple of integer indices.
        encoding: dict mapping taxon names (str) to indices (int).
        Indices are stored sorted and unique.
        """
        # Ensure input is iterable, then get unique elements, then sort.
        # This makes self.indices always represent a set of unique, sorted indices.
        _unique_indices_set = set(indices)
        self.indices: Tuple[int, ...] = tuple(sorted(list(_unique_indices_set)))

        self.encoding: Dict[str, int] = encoding or {}
        # Compute bitmask for fast hashing/comparison
        bitmask = 0
        # Iterate over the unique, sorted indices
        for idx in self.indices:
            bitmask |= 1 << idx
        self.bitmask: int = bitmask
        self._cached_reverse_encoding: Optional[Dict[int, str]] = None

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        # Accurately reflects the number of unique indices in the partition
        return len(self.indices)

    def _tuple_to_indices(self, other: Any) -> Tuple[int, ...] | None:
        """
        Convert a tuple of str or int to a canonical tuple of unique, sorted indices.
        Returns None if conversion fails.
        """
        if not isinstance(other, tuple):
            return None
        from typing import Any as _Any, Tuple as _Tuple, cast as _cast

        other_tuple = _cast(_Tuple[_Any, ...], other)
        processed_indices_list: Optional[List[int]] = None

        if all(isinstance(x, str) for x in other_tuple):
            try:
                # Convert names to indices, ensure uniqueness, then sort
                unique_indices_from_names = set(self.encoding[x] for x in other_tuple)
                processed_indices_list = sorted(list(unique_indices_from_names))
            except (
                KeyError,
                AttributeError,
            ):  # Catch if encoding is missing or names not found
                return None
        elif all(isinstance(x, int) for x in other_tuple):
            # Ensure uniqueness from input integers, then sort
            unique_input_indices = set(_cast(_Tuple[int, ...], other_tuple))
            processed_indices_list = sorted(list(unique_input_indices))

        if processed_indices_list is not None:
            return tuple(processed_indices_list)
        return None

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Partition):
            # Both self.indices and other.indices are unique & sorted
            return self.indices < other.indices

        # Convert 'other' to canonical unique, sorted tuple of indices for comparison
        other_indices_tuple = self._tuple_to_indices(other)
        if other_indices_tuple is not None:
            return self.indices < other_indices_tuple
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Partition):
            # Primary comparison using bitmask is efficient and correct for set equality
            return self.bitmask == other.bitmask

        # For comparison with other types (e.g., a raw tuple),
        # convert 'other' to its canonical unique, sorted tuple of indices.
        other_indices_tuple = self._tuple_to_indices(other)
        if other_indices_tuple is not None:
            # Compare canonical forms: self.indices (unique, sorted) vs other_indices_tuple (unique, sorted)
            # This could also be implemented by checking if their bitmasks would be identical,
            # but direct tuple comparison is fine here as both are canonical.
            if len(self.indices) != len(other_indices_tuple):  # Quick check
                return False
            return self.indices == other_indices_tuple
        return NotImplemented

    def __getitem__(self, index: int) -> int:
        return self.indices[index]

    @property
    def taxa(self) -> FrozenSet[str]:
        """
        Return the set of taxon names corresponding to the indices in this partition.
        """
        return frozenset(self.reverse_encoding[i] for i in self.indices)

    def bipartition(self) -> str:
        """
        Return a string representation of the bipartition (left | right) using taxon names.
        """
        left: List[str] = sorted(
            (self.reverse_encoding[i] for i in self.indices), key=len
        )
        right: List[str] = sorted(
            self.reverse_encoding[i] for i in self.complementary_indices()
        )
        return f"{', '.join(left)} | {', '.join(right)}"

    def complementary_indices(self) -> Tuple[int, ...]:
        """
        Return the indices not in this partition (complement set).
        """
        full_set = set(self.reverse_encoding.keys())
        return tuple(sorted(full_set - set(self.indices)))

    def __str__(self) -> str:
        try:
            taxa_names: List[str] = sorted(
                self.reverse_encoding.get(i, str(i)) for i in self.indices
            )
            return f"({', '.join(taxa_names)})"
        except Exception:
            return f"{tuple(sorted(self.indices))}"

    def __repr__(self) -> str:
        return f"({self})"

    def __hash__(self) -> int:
        return hash(self.bitmask)

    @property
    def reverse_encoding(self) -> Dict[int, str]:
        """
        Return a reverse mapping from index to taxon name.
        Caches the result for performance.
        """
        if self._cached_reverse_encoding is None:
            self._cached_reverse_encoding = {v: k for k, v in self.encoding.items()}
        return self._cached_reverse_encoding

    def __json__(self) -> List[int]:
        return list(self.indices)

    def to_dict(self) -> Dict[str, Tuple[int, ...]]:
        return {"indices": self.indices}

    def copy(self) -> "Partition":
        return Partition(self.indices, self.encoding)

    def __iand__(self, other: Any) -> "Partition":
        if isinstance(other, Partition):
            common_indices = set(self.indices) & set(other.indices)
            self.indices = tuple(sorted(common_indices))
            # Update bitmask as well
            bitmask = 0
            for idx in self.indices:
                bitmask |= 1 << idx
            self.bitmask = bitmask
            # Invalidate the cache since indices changed
            self._cached_reverse_encoding = None
            return self
        return NotImplemented

    def __and__(self, other: Any) -> "Partition":
        if isinstance(other, Partition):
            common_indices: set[int] = set(self.indices) & set(other.indices)
            return Partition(tuple(sorted(common_indices)), self.encoding)
        return NotImplemented

    def resolve_to_indices(self) -> Tuple[int, ...]:
        """
        Return the tuple of indices for this partition.
        """
        return tuple(self.indices)

    def is_compatible_with(self, other: "Partition") -> bool:
        """
        Check if this partition is compatible with another partition.

        Two partitions are compatible if at least one of the four intersections is empty:
        - A ∩ B
        - A ∩ B_complement
        - A_complement ∩ B
        - A_complement ∩ B_complement

        Where A and B are the two partitions, and complements are with respect
        to the universal set of all taxa in the encoding.

        Args:
            other: Another Partition to check compatibility with

        Returns:
            bool: True if partitions are compatible, False otherwise

        Raises:
            ValueError: If partitions have different encodings
        """

        # Ensure both partitions use the same encoding
        if self.encoding != other.encoding:
            raise ValueError(
                "Cannot check compatibility between partitions with different encodings"
            )

        # Get universal set from encoding
        if not self.encoding:
            # If no encoding, assume partitions contain all relevant indices
            all_indices = set(self.indices) | set(other.indices)
        else:
            all_indices = set(self.encoding.values())

        # Convert partitions to sets
        A = set(self.indices)
        B = set(other.indices)
        A_complement = all_indices - A
        B_complement = all_indices - B

        # Check if any of the four intersections is empty
        if not (A & B):
            return True
        if not (A & B_complement):
            return True
        if not (A_complement & B):
            return True
        if not (A_complement & B_complement):
            return True

        return False

    def check_compatibility_with_list(self, partitions: List["Partition"]) -> bool:
        """
        Check if this partition is compatible with all partitions in a list.

        Args:
            partitions: List of Partition objects to check compatibility with

        Returns:
            bool: True if this partition is compatible with ALL partitions in the list,
                  False if incompatible with any partition
        """
        return all(self.is_compatible_with(p) for p in partitions)
