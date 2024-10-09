import json
from dataclasses import dataclass, field, asdict
from copy import deepcopy
from typing import Optional, Any, NewType, Tuple, Dict, List, Generator

SplitIndices = NewType("SplitIndices", Tuple[int, ...])
Splits = NewType("Splits", Dict[SplitIndices, float])


@dataclass()
class Node:
    children: List["Node"] = field(default_factory=list, compare=False)
    name: Optional[str] = field(default=None, compare=False)
    length: Optional[float] = field(default=None, compare=True)
    values: Dict[str, Any] = field(default_factory=dict, compare=True)
    split_indices: SplitIndices = field(default_factory=tuple, compare=True)
    leaf_name: Optional[str] = field(default=None, compare=False)
    _order: List[str] = field(default_factory=list, compare=False)

    def append_child(self, node: "Node") -> None:
        self.children.append(node)

    def __repr__(self) -> str:
        return f"Node('{self.name}')"

    def __hash__(self) -> int:
        return hash(self.split_indices)

    def deep_copy(self) -> "Node":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_newick(self, lengths: bool = True) -> str:
        return self._to_newick(lengths=lengths) + ";"

    def _to_newick(self, lengths: bool = True) -> str:
        length = ""
        if (self.length is not None) and lengths:
            length = f":{self.length}"

        meta = ""
        if self.values:
            meta = (
                "["
                + ",".join(f"{key}={value}" for key, value in self.values.items())
                + "]"
            )

        children_str = ""
        if self.children:
            children_str = (
                "("
                + ",".join(child._to_newick(lengths=lengths) for child in self.children)
                + ")"
            )

        return f'{children_str}{self.name or ""}{meta}{length}'

    def to_json(self) -> str:
        serialized_dict = self.to_dict()
        return json.dumps(serialized_dict, indent=4)

    def _initialize_split_indices(self, order: List[str]) -> None:
        # Reset split_indices
        self.split_indices = ()

        for child in self.children:
            child._initialize_split_indices(order)
        if not self.children:
            self.split_indices = (order.index(self.name),)
        else:
            indices = []
            for child in self.children:
                indices.extend(child.split_indices)
            self.split_indices = tuple(sorted(indices))

    def _fix_child_order(self) -> None:
        self.children.sort(key=lambda node: min(node.split_indices))
        for child in self.children:
            child._fix_child_order()

    def traverse(self) -> Generator["Node", None, None]:
        yield self
        for child in self.children:
            yield from child.traverse()

    def to_splits(self) -> Splits:
        """Returns the tree as a dict mapping each split indices to the length of the split"""
        splits = {node.split_indices: node.length for node in self.traverse()}
        return splits

    def _index(self, component: Tuple[str, ...]) -> SplitIndices:
        return tuple(sorted(self._order.index(name) for name in component))

    def get_leaves(self) -> List["Node"]:
        """Return all leaf nodes of the subtree rooted at this node."""
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def reorder_taxa(self, permutation: List[str]) -> None:
        """
        Reorder the taxa in the tree according to the given permutation.
        """
        # Create a mapping from taxa names to their new positions
        taxa_order = {name: idx for idx, name in enumerate(permutation)}

        # Recursive function to reorder children
        def _reorder(node: "Node") -> None:
            if node.children:
                # Sort children based on the minimum index of their descendant taxa
                node.children.sort(
                    key=lambda child: min(
                        taxa_order[leaf.name] for leaf in child.get_leaves()
                    )
                )
                for child in node.children:
                    _reorder(child)

        _reorder(self)
