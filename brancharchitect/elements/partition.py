# split.py
from __future__ import annotations
from typing import Tuple, FrozenSet, Dict, Iterator, List, Any, Optional
from functools import total_ordering


@total_ordering
class Partition:
    __slots__ = (
        "_indices",
        "encoding",
        "bitmask",
        "_cached_reverse_encoding",
        "_cached_size",
        "_cached_hash",
        "_cached_taxa",
    )

    @classmethod
    def from_bitmask(cls, bitmask: int, encoding: Dict[str, int]) -> Partition:
        """
        Create a Partition directly from a bitmask and encoding.
        This is significantly faster than the standard constructor as it avoids
        sorting and set operations by delaying index derivation.
        """
        obj = cls.__new__(cls)
        obj._indices = None  # Lazily computed
        obj.encoding = encoding
        obj.bitmask = bitmask
        obj._cached_size = bitmask.bit_count()
        obj._cached_hash = hash(bitmask)
        obj._cached_reverse_encoding = None
        obj._cached_taxa = None
        return obj

    @property
    def indices(self) -> Tuple[int, ...]:
        """Lazy derivation of sorted indices from bitmask.

        Uses an optimized algorithm that extracts set bits directly,
        which is much faster for sparse bitmasks (typical in phylogenetics).
        """
        if self._indices is None:
            indices_list = []
            temp_mask = self.bitmask
            # Use bit manipulation to extract set bits directly
            # This is O(k) where k is the number of set bits, not O(n) where n is bit_length
            while temp_mask:
                # Find the lowest set bit position using bit_length
                lowest_bit = temp_mask & -temp_mask
                idx = lowest_bit.bit_length() - 1
                indices_list.append(idx)
                # Clear the lowest set bit
                temp_mask &= temp_mask - 1
            self._indices = tuple(indices_list)
        return self._indices

    def __init__(
        self, indices: Tuple[int, ...], encoding: Optional[Dict[str, int]] = None
    ):
        """
        Partition represents a subset of taxa as a tuple of integer indices.

        **TERMINOLOGY NOTE**:
        In this codebase, a 'Partition' represents a **Taxon Cluster** (Clade) or a
        **Phylogenetic Split** (Bipartition). It corresponds to a single vertex in the
        Cluster Containment Lattice. It is **NOT** a mathematical partition of a set (collection of disjoint subsets).

        Aliases: `Cluster`, `Split`

        Args:
            indices: Tuple of integer indices.
            encoding: dict mapping taxon names (str) to indices (int).

        Indices are stored sorted and unique.
        Preconditions/validation:
        - All indices must be integers >= 0
        - If encoding is provided and non-empty, all indices must be present in encoding.values()
        """
        # Ensure input is iterable, then get unique elements, then sort.
        _unique_indices_set = set(indices)
        self._indices: Optional[Tuple[int, ...]] = tuple(sorted(_unique_indices_set))
        self.encoding: Dict[str, int] = encoding or {}

        # Compute bitmask and validate in single pass
        bitmask = 0
        for idx in self._indices:
            if idx < 0:
                raise ValueError(
                    f"Partition indices must be non-negative integers; got {self._indices}"
                )
            bitmask |= 1 << idx

        self.bitmask: int = bitmask
        self._cached_size: int = bitmask.bit_count()
        self._cached_hash: int = hash(bitmask)
        self._cached_reverse_encoding: Optional[Dict[int, str]] = None
        self._cached_taxa: Optional[FrozenSet[str]] = None

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        # Avoid triggering lazy indices derivation if only count is needed
        return self._cached_size

    def __bool__(self) -> bool:
        # Efficient truthiness check without length or indices
        return self.bitmask != 0

    @property
    def size(self) -> int:
        """Return the number of taxa in this partition (cached for performance)."""
        return self._cached_size

    @property
    def is_singleton(self) -> bool:
        """Return True if this partition contains exactly one index (atom)."""
        return self._cached_size == 1

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

        Result is cached for performance (avoids repeated decoding).
        """
        if self._cached_taxa is None:
            self._cached_taxa = frozenset(
                self.reverse_encoding[i] for i in self.indices
            )
        return self._cached_taxa

    def bipartition(self) -> str:
        """
        Return a string representation of the bipartition (left | right).

        WARNING: This method relies on the partition's `encoding` to define the
        universe of taxa for the complement. If the encoding is incomplete,
        the right-hand side of the bipartition may also be incomplete.
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
        Return the indices not in this partition (the complement set).

        WARNING: The universe for the complement is derived from the keys in the
        `encoding` dictionary. If the encoding is not complete (i.e., it does
        not contain all taxa in the analysis), this method will produce an
        incorrect or incomplete complement.
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
        return self._cached_hash

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

    def __and__(self, other: Any) -> "Partition":
        if isinstance(other, Partition):
            # Fast path: use bitmask intersection directly
            new_bitmask = self.bitmask & other.bitmask
            return Partition.from_bitmask(new_bitmask, self.encoding)
        return NotImplemented

    def intersection(self, *others: Any) -> "Partition":
        """
        Return the set intersection with one or more other partitions/partition-like inputs.

        Accepts other Partition instances or tuples of indices/names (resolved via encoding).
        Returns a new Partition with the same encoding.
        """
        # Start with current indices
        result_set = set(self.indices)

        for other in others:
            if isinstance(other, Partition):
                result_set &= set(other.indices)
            else:
                other_indices_tuple = self._tuple_to_indices(other)
                if other_indices_tuple is None:
                    return NotImplemented  # type: ignore[return-value]
                result_set &= set(other_indices_tuple)

            # Early exit if empty
            if not result_set:
                break

        return Partition(tuple(sorted(result_set)), self.encoding)

    def __sub__(self, other: Any) -> "Partition":
        """
        Set-difference between partitions or partition-like inputs.

        Returns a new Partition consisting of elements in self not in other.
        Accepts another Partition or a tuple of names/indices.
        """
        if isinstance(other, Partition):
            if self.encoding and other.encoding and self.encoding is not other.encoding:
                if self.encoding != other.encoding:
                    raise ValueError(
                        "Cannot subtract partitions with different encodings"
                    )
            # Fast path: use bitmask difference directly
            new_bitmask = self.bitmask & ~other.bitmask
            return Partition.from_bitmask(new_bitmask, self.encoding)
        else:
            other_indices_tuple = self._tuple_to_indices(other)
            if other_indices_tuple is None:
                return NotImplemented
            # Build bitmask for other indices
            other_bitmask = 0
            for idx in other_indices_tuple:
                other_bitmask |= 1 << idx
            new_bitmask = self.bitmask & ~other_bitmask
            return Partition.from_bitmask(new_bitmask, self.encoding)

    def resolve_to_indices(self) -> Tuple[int, ...]:
        """
        Return the tuple of indices for this partition.
        """
        return tuple(self.indices)

    def is_compatible_with(self, other: "Partition", all_indices: set[int]) -> bool:
        """
        Check if this partition is compatible with another partition.

        Two partitions are compatible if at least one of the four intersections is empty:
        - A ∩ B
        - A ∩ B_complement
        - A_complement ∩ B
        - A_complement ∩ B_complement

        Where A and B are the two partitions, and complements are with respect
        to the provided universal set of all taxa indices.

        Args:
            other: Another Partition to check compatibility with.
            all_indices: The set of all possible indices, defining the universe.

        Returns:
            bool: True if partitions are compatible, False otherwise.

        Raises:
            ValueError: If partitions have different encodings.
        """
        if self.encoding != other.encoding:
            raise ValueError(
                "Cannot check compatibility between partitions with different encodings"
            )

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

    def check_compatibility_with_list(
        self, partitions: List["Partition"], all_indices: set[int]
    ) -> bool:
        """
        Check if this partition is compatible with all partitions in a list.

        Args:
            partitions: List of Partition objects to check compatibility with.
            all_indices: The set of all possible indices for the compatibility check.

        Returns:
            bool: True if this partition is compatible with ALL partitions in the list,
                  False if incompatible with any partition.
        """
        return all(self.is_compatible_with(p, all_indices) for p in partitions)
