# split.py
from typing import Set, Tuple, Optional, FrozenSet, Dict
from dataclasses import dataclass, field
from functools import total_ordering
from itertools import product


@dataclass(frozen=True)  # Remove order=True
@total_ordering  # Add total_ordering decorator
class Partition:
    indices: Tuple[int, ...]
    lookup: Dict[str, int] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self):
        # Handle the case when indices is a single integer
        if isinstance(self.indices, int):
            object.__setattr__(
                self, "indices", (self.indices,)
            )  # Create a single-element tuple
        # For collections, ensure we have a sorted tuple
        else:
            # Sort indices during initialization
            object.__setattr__(self, "indices", tuple(sorted(self.indices)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def __lt__(self, other):
        if isinstance(other, Partition):
            return self.indices < other.indices
        elif isinstance(other, tuple):
            # Handle string-based splits
            if all(isinstance(x, str) for x in other):
                try:
                    other_indices = tuple(sorted(self.lookup[x] for x in other))
                    return self.indices < other_indices
                except (KeyError, AttributeError):
                    return False
            # Handle integer-based splits
            return self.indices < tuple(sorted(other))
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Partition):
            return self.indices == other.indices
        elif isinstance(other, tuple):
            # Handle string-based splits
            if all(isinstance(x, str) for x in other):
                try:
                    other_indices = tuple(sorted(self.lookup[x] for x in other))
                    return self.indices == other_indices
                except (KeyError, AttributeError):
                    return False
            # Handle integer-based splits
            return self.indices == tuple(sorted(other))
        return NotImplemented

    def __getitem__(self, index):
        return self.indices[index]

    @property
    def taxa(self) -> FrozenSet[str]:
        """Return the set of taxa included in this split."""
        return frozenset(self.reverse_lookup[i] for i in self.indices)

    def bipartition(self) -> str:
        """
        Return a string representing the bipartition.
        The left side is the taxa in this split; the right side is the complement.
        """
        left = sorted((self.reverse_lookup[i] for i in self.indices), key=len)
        right = sorted(self.reverse_lookup[i] for i in self.complementary_indices())
        return f"{', '.join(left)} | {', '.join(right)}"

    def complementary_indices(self) -> Tuple[int, ...]:
        """Return the indices not included in the split."""
        full_set = set(self.reverse_lookup.keys())
        return tuple(sorted(full_set - set(self.indices)))

    def __str__(self) -> str:
        """Return a string representation of the partition using taxa names when available."""
        try:
            # Convert indices to taxa names in a single comprehension
            taxa_names = sorted(
                self.reverse_lookup.get(i, str(i)) for i in self.indices
            )
            return f"({', '.join(taxa_names)})"
        except Exception:
            # Fallback to raw indices if lookup fails
            return f"{tuple(sorted(self.indices))}"

    def __repr__(self) -> str:
        return f"({self})"

    def __hash__(self):
        return hash(
            self.indices
        )  # Only hash the indices since order is used for lookup

    @property
    def reverse_lookup(self) -> Dict[str, int]:
        return {v: k for k, v in self.lookup.items()}


    def __json__(self):
        """Return just the indices for JSON serialization."""
        return list(self.indices)

    def to_dict(self):
        """Convert to a dict containing only indices."""
        return {"indices": self.indices}
    
    def copy(self):
        """Return a copy of the partition."""
        return Partition(self.indices, self.lookup)
    
    # Add this method to your Partition class
    def __iand__(self, other):
        """Implement the in-place intersection operator (&=)."""
        if isinstance(other, Partition):
            # Calculate the intersection of indices
            common_indices = set(self.indices) & set(other.indices)
            # Since the class is frozen, we need to use object.__setattr__
            object.__setattr__(self, "indices", tuple(sorted(common_indices)))
            return self
        return NotImplemented

    # You should also implement __and__ for the regular & operator
    def __and__(self, other):
        """Implement the intersection operator (&)."""
        if isinstance(other, Partition):
            # Calculate the intersection of indices
            common_indices = set(self.indices) & set(other.indices)
            # Return a new Partition with the common indices
            return Partition(tuple(sorted(common_indices)), self.lookup)
        return NotImplemented

class PartitionSet(set):
    """
    A set of IndexedSplit objects that carries an extra 'order' attribute.
    """

    def __init__(
        self,
        splits: Optional[Set[Partition]] = None,
        look_up: Optional[Dict[str, int]] = None,
        name: str = "PartitionSet",
    ):
        super().__init__(splits or set())
        # Force 'order' to be a tuple for consistency.
        self.look_up = look_up
        self.reversed_lookup = {v: k for k, v in look_up.items()} if look_up else {}
        self.order = tuple(look_up.values()) if look_up else None
        self.name = name

    def atom(self) -> "PartitionSet":
        # A split is minimal if no other split has a strictly smaller taxa set.
        minimal = {
            s
            for s in self
            if not any((other != s and other.taxa < s.taxa) for other in self)
        }
        return PartitionSet(minimal, name="atoms")

    def cover(self) -> "PartitionSet":
        # A split is maximal if no other split has a strictly larger taxa set.
        maximal = {
            s
            for s in self
            if not any((other != s and other.taxa > s.taxa) for other in self)
        }
        return PartitionSet(maximal, self.look_up, name="covering")

    # Overriding set operations
    def union(self, *others: Set) -> "PartitionSet":
        result = super().union(*others)
        return PartitionSet(result, name=self.name + "_union")

    def intersection(self, *others: Set) -> "PartitionSet":
        result = super().intersection(*others)
        return PartitionSet(result, name=self.name + "_intersection", look_up=self.look_up)

    def difference(self, *others: Set) -> "PartitionSet":
        result = super().difference(*others)
        return PartitionSet(result, name=self.name + "_difference", look_up=self.look_up)

    def symmetric_difference(self, other: Set) -> "PartitionSet":
        result = super().symmetric_difference(other)
        return PartitionSet(result, name=self.name + "_symdiff", look_up=self.look_up)

    # Also override operators for convenience.
    def __or__(self, other: Set) -> "PartitionSet":
        return self.union(other)

    def __and__(self, other: Set) -> "PartitionSet":
        return self.intersection(other)

    def __sub__(self, other: Set) -> "PartitionSet":
        return self.difference(other)

    def __xor__(self, other: Set) -> "PartitionSet":
        return self.symmetric_difference(other)

    def __str__(self) -> str:
        splits_list = sorted(self, key=lambda s: s)
        # return "\n".join(f"({s.bipartition()})" for s in splits_list)
        return "\n".join(str(s) for s in splits_list)

    def __repr__(self) -> str:
        return str(sorted(self, key=lambda s: s))

    def __hash__(self):
        # Make IndexedSplitSet hashable by using its contents and order
        return hash((frozenset(self), self.order))

    def cartesian(self, other: "PartitionSet") -> "PartitionSet":
        """
        Compute the Cartesian product of two IndexedSplitSets.

        Args:
            other (IndexedSplitSet): The other set to compute product with

        Returns:
            IndexedSplitSet: A new set containing pairs of splits as tuples
        """
        if not isinstance(other, PartitionSet):
            raise TypeError("Cartesian product only defined for IndexedSplitSet")

        # Create product using itertools.product
        cart_product = set(product(self, other))

        # Return new IndexedSplitSet with combined order if available
        combined_order = None
        if self.order is not None and other.order is not None:
            combined_order = self.order + other.order

        return PartitionSet(
            cart_product,
            order=combined_order,
            name=f"{self.name}_cartesian_{other.name}",
        )

    def __mul__(self, other: "PartitionSet") -> "PartitionSet":
        """Operator × for Cartesian product."""
        return self.cartesian(other)

    # Add this method to PartitionSet class
    def add(self, element):
        if not isinstance(element, Partition):
            raise TypeError(f"Can only add Partition objects, got {type(element)}")
        super().add(element)