# split.py
from typing import Tuple, FrozenSet, Dict, Iterator, List, Any
from functools import total_ordering


@total_ordering
class Partition:
    __slots__ = ('indices', 'encoding', 'bitmask')

    def __init__(self, indices: Tuple[int, ...], encoding: Dict[str, int] = None):
        """
        Partition represents a subset of taxa as a tuple of integer indices.
        encoding: dict mapping taxon names (str) to indices (int).
        """
        self.indices = tuple(sorted(indices))
        self.encoding = encoding or {}
        # Compute bitmask for fast hashing/comparison
        bitmask = 0
        for idx in self.indices:
            bitmask |= 1 << idx
        self.bitmask = bitmask

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Partition):
            return self.indices < other.indices
        elif isinstance(other, tuple):
            if all(isinstance(x, str) for x in other):
                try:
                    other_indices = tuple(sorted(self.encoding[x] for x in other))
                    return self.indices < other_indices
                except (KeyError, AttributeError):
                    return False
            return self.indices < tuple(sorted(other))
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Partition):
            return self.bitmask == other.bitmask
        elif isinstance(other, tuple):
            if all(isinstance(x, str) for x in other):
                try:
                    other_indices = tuple(sorted(self.encoding[x] for x in other))
                    return self.indices == other_indices
                except (KeyError, AttributeError):
                    return False
            return self.indices == tuple(sorted(other))
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
        left = sorted((self.reverse_encoding[i] for i in self.indices), key=len)
        right = sorted(self.reverse_encoding[i] for i in self.complementary_indices())
        return f"{', '.join(left)} | {', '.join(right)}"

    def complementary_indices(self) -> Tuple[int, ...]:
        """
        Return the indices not in this partition (complement set).
        """
        full_set = set(self.reverse_encoding.keys())
        return tuple(sorted(full_set - set(self.indices)))

    def __str__(self) -> str:
        try:
            taxa_names = sorted(
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
        """
        return {v: k for k, v in self.encoding.items()}

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
            return self
        return NotImplemented

    def __and__(self, other: Any) -> "Partition":
        if isinstance(other, Partition):
            common_indices = set(self.indices) & set(other.indices)
            return Partition(tuple(sorted(common_indices)), self.encoding)
        return NotImplemented

    def resolve_to_indices(self) -> Tuple[int, ...]:
        """
        Return the tuple of indices for this partition.
        """
        return tuple(self.indices)