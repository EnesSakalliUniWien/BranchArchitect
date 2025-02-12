from typing import Set, Tuple, Optional, FrozenSet

# Define SplitIndices as an immutable tuple.
class SplitIndices(tuple):
    # Remove __slots__ (or set it to an empty tuple)
    def __new__(cls, indices: Tuple[int, ...], order: Tuple[str, ...]):
        if not all(isinstance(i, int) and 0 <= i < len(order) for i in indices):
            raise ValueError("Invalid indices for the given order")
        # Optionally sort the indices to ensure consistency
        sorted_indices = tuple(sorted(indices))
        obj = super().__new__(cls, sorted_indices)
        # Instead of using __slots__, we simply attach order to the instance.
        object.__setattr__(obj, "order", order)
        return obj

    @property
    def taxa(self) -> FrozenSet[str]:
        return frozenset(self.order[i] for i in self)

    def complementary_indices(self) -> Tuple[int, ...]:
        full_set = set(range(len(self.order)))
        comp = full_set - set(self)
        return tuple(sorted(comp))

    def __str__(self) -> str:
        taxa_str = ", ".join(self.order[i] for i in self)
        return f"Split({taxa_str})"

    def __repr__(self) -> str:
        return f"{tuple(self)}"

class IndexedSplitSet(set):
    """
    A set of IndexedSplit objects that carries an extra 'order' attribute and extra methods
    for comparing and filtering splits. Because this class subclasses set, it is usable in all
    contexts where a plain set is expected.
    """
    def __init__(self, splits: Optional[Set[SplitIndices]] = None,
                 order: Optional[Tuple[str, ...]] = None,
                 name: str = "IndexedSplitSet"):
        # Initialize as a normal set.
        super().__init__(splits or set())
        self.order = order
        self.name = name

    def common_splits(self, other: "IndexedSplitSet") -> "IndexedSplitSet":
        if self.order != other.order:
            raise ValueError("Cannot compare splits with different orders")
        return IndexedSplitSet(self.intersection(other), order=self.order, name="CommonSplits")

    def unique_splits(self, other: "IndexedSplitSet") -> "IndexedSplitSet":
        if self.order != other.order:
            raise ValueError("Cannot compare splits with different orders")
        return IndexedSplitSet(self.difference(other), order=self.order, name="UniqueSplits")

    def minimal_splits(self) -> "IndexedSplitSet":
        # A split is minimal if no other split has a strictly smaller taxa set.
        minimal = {s for s in self
                   if not any((other != s and other.taxa < s.taxa) for other in self)}
        return IndexedSplitSet(minimal, order=self.order, name="MinimalSplits")

    def maximal_splits(self) -> "IndexedSplitSet":
        # A split is maximal if no other split has a strictly larger taxa set.
        maximal = {s for s in self
                   if not any((other != s and other.taxa > s.taxa) for other in self)}
        return IndexedSplitSet(maximal, order=self.order, name="MaximalSplits")

    def __str__(self) -> str:
        splits_list = sorted(self, key=lambda s: s)
        return f"{self.name}:\n" + "\n".join(str(s) for s in splits_list)

    def __repr__(self) -> str:
        return f"IndexedSplitSet(splits={set(self)}, order={self.order}, name='{self.name}')"
