from typing import Set, NewType, Tuple, List, FrozenSet
from dataclasses import dataclass

# Change from Set[int] to Tuple[int, ...] for immutability
SplitIndices = NewType("SplitIndices", Tuple[int, ...])


@dataclass(frozen=True)
class IndexedSplit:
    indices: SplitIndices
    order: Tuple[str]

    def __post_init__(self):
        if not isinstance(self.indices, tuple):
            object.__setattr__(self, "indices", tuple(self.indices))
        if not all(
            isinstance(i, int) and 0 <= i < len(self.order) for i in self.indices
        ):
            raise ValueError("Invalid indices for the given order")

    @property
    def taxa(self) -> FrozenSet[str]:
        return frozenset(self.order[i] for i in self.indices)

    def complementary_indices(self) -> SplitIndices:
        full_set = set(range(len(self.order)))
        complement = full_set - set(self.indices)
        return SplitIndices(tuple(sorted(complement)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, IndexedSplit):
            return NotImplemented
        return self.indices == other.indices and self.order == other.order

    def __hash__(self) -> int:
        return hash((self.indices, tuple(self.order)))

    def __str__(self) -> str:
        taxa_str = ", ".join(self.order[i] for i in self.indices)
        return f"Split({taxa_str})"

    def __repr__(self) -> str:
        return f"IndexedSplit(indices={self.indices}, order={self.order})"


class IndexedSplitSet:
    def __init__(
        self,
        splits: Set[IndexedSplit] = None,
        order: List[str] = None,
        name: str = "IndexedSplitSet",
    ):
        self.splits = splits
        self.order = order
        self.name = name

    @classmethod
    def from_node(cls, node: "Node") -> "IndexedSplitSet":
        # Extract all proper splits from the node
        node_splits = node.to_splits()
        # Convert each SplitIndices into an IndexedSplit
        indexed_splits = {IndexedSplit(s, tuple(node._order)) for s in node_splits}
        return cls(indexed_splits, tuple(node._order))

    def _validate_splits(self) -> None:
        for split in self.splits:
            if split.order != self.order:
                raise ValueError(f"Split {split} has different order than SplitSet")

    def unique_splits(self, other: "IndexedSplitSet") -> "IndexedSplitSet":
        if self.order != other.order:
            print(self.order)
            print(other.order)
            raise ValueError("Cannot compare splits with different orders")
            
        unique = self.splits - other.splits
        return IndexedSplitSet(unique, self.order, name="UniqueSplits")

    def common_splits(self, other: "IndexedSplitSet") -> "IndexedSplitSet":
        if self.order != other.order:
            raise ValueError("Cannot compare splits with different orders")
        common = self.splits & other.splits
        return IndexedSplitSet(common, self.order, name="CommonSplits")

    def minimal_splits(self) -> "IndexedSplitSet":
        minimal = set()
        for split in self.splits:
            is_minimal = True
            for other in self.splits:
                if other != split and other.taxa < split.taxa:
                    is_minimal = False
                    break
            if is_minimal:
                minimal.add(split)
        return IndexedSplitSet(minimal, self.order, name="MinimalSplits")

    def maximal_splits(self) -> "IndexedSplitSet":
        maximal = set()
        for split in self.splits:
            is_maximal = True
            for other in self.splits:
                if other != split and other.taxa > split.taxa:
                    is_maximal = False
                    break
            if is_maximal:
                maximal.add(split)
        return IndexedSplitSet(maximal, self.order, name="MaximalSplits")

    def vertical_print(self, as_taxa: bool = True):
        splits_list = list(self.splits)
        splits_list.sort(key=lambda s: sorted(s.taxa))

        if as_taxa:
            taxa_order = sorted({t for s in splits_list for t in s.taxa})
            matrix = []
            for taxon in taxa_order:
                row = [taxon if taxon in s.taxa else "" for s in splits_list]
                matrix.append(row)
            header = [f"Split {i+1}" for i in range(len(splits_list))]
        else:
            all_indices = sorted({i for s in splits_list for i in s.indices})
            matrix = []
            for idx in all_indices:
                row = [str(idx) if idx in s.indices else "" for s in splits_list]
                matrix.append(row)
            header = [f"Split {i+1}" for i in range(len(splits_list))]

        print(" | ".join(header))
        print("-" * (len(header) * 10))
        for row in matrix:
            print(" | ".join(element.ljust(8) for element in row))

    def vertical_print_with_tabulate(self, as_taxa: bool = True):
        from tabulate import tabulate

        splits_list = list(self.splits)
        splits_list.sort(key=lambda s: sorted(s.taxa))

        if as_taxa:
            taxa_order = sorted({t for s in splits_list for t in s.taxa})
            matrix = []
            header = [f"Split {i+1}" for i in range(len(splits_list))]
            for taxon in taxa_order:
                row = [taxon if taxon in s.taxa else "" for s in splits_list]
                matrix.append(row)
            print(tabulate(matrix, headers=header, tablefmt="fancy_grid"))
        else:
            all_indices = sorted({i for s in splits_list for i in s.indices})
            matrix = []
            header = [f"Split {i+1}" for i in range(len(splits_list))]
            for idx in all_indices:
                row = [str(idx) if idx in s.indices else "" for s in splits_list]
                matrix.append(row)
            print(tabulate(matrix, headers=header, tablefmt="fancy_grid"))

    def vertical_print_combined(self):
        splits_list = list(self.splits)
        splits_list.sort(key=lambda s: sorted(s.taxa))

        for i, split in enumerate(splits_list, start=1):
            indices_str = ",".join(map(str, split.indices))
            taxa_str = ",".join(sorted(split.taxa))
            print(f"Split {i}: Indices=[{indices_str}] | Taxa=[{taxa_str}]")

    def __str__(self) -> str:
        splits_str = "\n".join(
            str(split) for split in sorted(self.splits, key=lambda x: x.indices)
        )
        return f"{self.name}:\n{splits_str}"

    def __repr__(self) -> str:
        return f"IndexedSplitSet(splits={self.splits}, order={self.order}, name='{self.name}')"
