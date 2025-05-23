import json
from enum import Enum
from statistics import mean
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict, List, Self, cast
from brancharchitect.partition_set import PartitionSet, Partition


class ReorderStrategy(Enum):
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MEDIAN = "median"


@dataclass()
class Node:
    children: List[Self] = field(default_factory=list, compare=False)
    length: Optional[float] = field(default=None, compare=True)
    values: Dict[str, Any] = field(default_factory=dict, compare=True)
    split_indices: Partition = field(
        default_factory=lambda: Partition(()), compare=True
    )
    parent: Optional[Self] = None
    _encoding: Dict[str, int] = field(default_factory=dict, compare=False)
    _split_index = None
    # Only keep caches that are actually used in the code below
    _cached_subtree_order: Optional[Tuple[str, ...]] = field(
        default=None, init=False, compare=False
    )
    _cached_subtree_cost: Optional[float] = field(
        default=None, init=False, compare=False
    )
    _cache_valid: bool = field(default=False, init=False, compare=False)
    _traverse_cache: Optional[List[Self]] = field(
        default=None, init=False, compare=False
    )
    _splits_cache: Optional[PartitionSet] = field(
        default=None, init=False, compare=False
    )

    def __init__(
        self,
        children=None,
        name=None,
        length=None,
        values=None,
        split_indices=(),
        _order=None,
        _encoding=None,
    ):
        self.children = children or []
        self.name = name
        self.length = length
        self.values = values or {}
        self.split_indices = split_indices
        self._order = _order or []
        self._split_index = None
        self._cached_subtree_order = None
        self._cached_subtree_cost = None
        self._cache_valid = False
        self._traverse_cache = None
        self._splits_cache = None

        if not self._order:
            self._order = self.get_current_order()
        if not _encoding:
            self._encoding = {name: i for i, name in enumerate(self._order)}

        if not self._encoding:
            raise ValueError("Encoding dictionary cannot be empty")

    @property
    def leaves(self) -> List[Self]:
        """
        Get all leaf nodes in the subtree rooted at this node.

        A leaf node is defined as a node with no children.

        Returns:
            List[Node]: List of all leaf nodes in this subtree.
        """
        if not self.children:
            return [self]  # This node is a leaf

        # Collect leaves from all children recursively
        result: List[Self] = []
        for child in self.children:
            # Use cast to tell the type checker these are compatible with Self
            result.extend(cast(List[Self], child.leaves))
        return result

    # ------------------------------------------------------------------------
    # Equality & hashing
    # ------------------------------------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.split_indices == other.split_indices

    def __hash__(self) -> int:
        return hash(frozenset(self.to_splits()))

    def __repr__(self) -> str:
        return f"Node('{self.name}')"

    def __str__(self):
        return str(tuple(sorted(self.get_current_order())))

    # ------------------------------------------------------------------------
    # Splits & related caching
    # ------------------------------------------------------------------------

    def to_splits(self, with_leaves=False) -> PartitionSet:
        if self._splits_cache is not None and not with_leaves:
            return self._splits_cache
        splits: PartitionSet = PartitionSet(encoding=self._encoding)
        for nd in self.traverse():
            if nd.children:
                splits.add(nd.split_indices)
            if with_leaves:
                splits.add(nd.split_indices)
        if not with_leaves:
            self._splits_cache = splits
        return splits

    def build_split_index(self):
        self._split_index = {}
        self._populate_split_index(self)

    def _populate_split_index(self, node):
        if node.split_indices is not None:
            self._split_index[node.split_indices] = node
        for ch in node.children:
            self._populate_split_index(ch)

    def find_node_by_split(self, target_split) -> Optional[Self]:
        """
        Find a node by its split indices (accepts tuple or Partition).
        """
        try:
            if self._split_index is None:
                self.build_split_index()
            if self._split_index is None:
                return None
            # Convert tuple to Partition if needed
            if not isinstance(target_split, Partition):
                target_split = Partition(tuple(target_split), self._encoding)
            return self._split_index.get(target_split)
        except Exception as e:
            raise ValueError(f"Error finding node by split: {e}")
 
    # ------------------------------------------------------------------------
    # append_child sets parent pointer (pointer-based approach)
    # ------------------------------------------------------------------------
    def append_child(self, node: Self) -> None:
        node.parent = self
        self.children.append(node)
        self.invalidate_caches(propagate_up=True)

    # ------------------------------------------------------------------------
    # deep_copy (unchanged, except we skip copying .parent)
    # ------------------------------------------------------------------------
    def deep_copy(self):
        new_node = Node(
            name=self.name,
            length=self.length,
            values=self.values.copy(),
            split_indices=self.split_indices,
            _order=tuple([name for name in self._order]),
        )
        new_node.children = [child.deep_copy() for child in self.children]
        for child in new_node.children:
            child.parent = new_node
        return new_node

    # ------------------------------------------------------------------------
    # split_indices initialization (unchanged)
    # ------------------------------------------------------------------------

    def _initialize_split_indices(self, encoding: Dict[str, int]) -> None:
        """Initialize split indices with better error handling and validation."""
        # Process children first
        for child in self.children:
            child._initialize_split_indices(encoding)

        try:
            if not self.children:
                self.split_indices = Partition((encoding[self.name],), encoding)
            else:
                # For internal nodes, collect child indices
                idxs: list[int] = []
                for ch in self.children:
                    idxs.extend(tuple(sorted(ch.split_indices)))
                self.split_indices = Partition(tuple(sorted(idxs)), encoding)

            # Rebuild split index after modification
            self.build_split_index()

        except Exception as e:
            raise ValueError(f"Failed to initialize split indices: {str(e)}")

    # ------------------------------------------------------------------------
    # traversal, fix_child_order, to_hierarchy, etc.
    # ------------------------------------------------------------------------

    def traverse(self) -> List[Self]:
        if self._traverse_cache is not None:
            return self._traverse_cache
        nodes = [self]
        for child in self.children:
            nodes.extend(child.traverse())
        self._traverse_cache = nodes
        return nodes

    def _index(self, component: Tuple[str, ...]) -> Partition:
        return Partition(
            tuple(sorted(self._order.index(name) for name in component)), self._encoding
        )

    def _fix_child_order(self) -> None:
        self.children.sort(key=lambda node: min(node.split_indices))
        for child in self.children:
            child._fix_child_order()

    def to_hierarchy(self) -> Dict[str, Any]:
        return {
            "name": self.name or "Internal",
            "children": (
                [c.to_hierarchy() for c in self.children] if self.children else []
            ),
            "values": self.values,
        }

    def swap_children(self):
        if len(self.children) >= 2:
            self.children[0], self.children[1] = self.children[1], self.children[0]
            self.invalidate_caches(propagate_up=True)

    def to_weighted_splits(self) -> Dict[Partition, float]:
        return {
            nd.split_indices: (nd.length if nd.length is not None else 0.0)
            for nd in self.traverse()
        }

    # ------------------------------------------------------------------------
    # reorder_taxa => if children changed => invalidate
    # ------------------------------------------------------------------------

    def reorder_taxa(
        self,
        permutation: List[str],
        strategy: "ReorderStrategy" = ReorderStrategy.MINIMUM,
    ) -> None:
        tree_taxa = {leaf.name for leaf in self.get_leaves()}
        if set(permutation) != tree_taxa:
            raise ValueError(
                "Permutation must include all taxa in the tree.", permutation, tree_taxa
            )

        _visual_order_indices = {name: idx for idx, name in enumerate(permutation)}

        sorting_strategies = {
            ReorderStrategy.AVERAGE: lambda leaves: mean(
                _visual_order_indices[leaf.name] for leaf in leaves
            ),
            ReorderStrategy.MAXIMUM: lambda leaves: max(
                _visual_order_indices[leaf.name] for leaf in leaves
            ),
            ReorderStrategy.MINIMUM: lambda leaves: min(
                _visual_order_indices[leaf.name] for leaf in leaves
            ),
            ReorderStrategy.MEDIAN: lambda leaves: sorted(
                _visual_order_indices[leaf.name] for leaf in leaves
            )[len(leaves) // 2],
        }

        def _reorder(node: Self) -> None:
            if node.children:
                for child in node.children:
                    _reorder(child)
                # Sort children using selected strategy
                strategy_fn = sorting_strategies[strategy]
                node.children.sort(key=lambda child: strategy_fn(child.get_leaves()))

        _reorder(self)
        self.invalidate_caches(propagate_up=True)

    def get_leaves(self) -> List[Self]:
        """
        Return all leaf nodes in the subtree rooted at this node.
        Always traverses the current tree structure (no memoization).
        """
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(cast(List[Self], child.get_leaves()))
        return leaves

    # ------------------------------------------------------------------------
    # (NEW) Cached get_current_order
    # ------------------------------------------------------------------------
    def get_current_order(self) -> Tuple[str, ...]:
        """
        Return the current order of taxa in the tree as a tuple.
        """
        return tuple(str(leaf.name) for leaf in self.get_leaves())

    def to_newick(self, lengths: bool = True) -> str:
        return self._to_newick(lengths=lengths) + ";"

    def _to_newick(self, lengths: bool = True) -> str:
        meta = ""
        if self.values:
            meta = "[" + ",".join(f"{k}={v}" for k, v in self.values.items()) + "]"

        if self.children:
            child_str = (
                "(" + ",".join(ch._to_newick(lengths) for ch in self.children) + ")"
            )
            if lengths:
                return f"{child_str}{self.name or ''}{meta}:{self.length}"
            else:
                return f"{child_str}{self.name or ''}{meta}"
        else:
            if lengths:
                return f"{self.name or ''}{meta}:{self.length}"
            else:
                return f"{self.name or ''}{meta}"

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self) -> Dict[str, Any]:
        # Always serialize split_indices as a list of ints for JSON compatibility
        if self.is_leaf():
            return {
                "name": self.name,
                "length": self.length,
                "split_indices": self.split_indices.resolve_to_indices(),
                "children": [],
            }
        else:
            return {
                "name": "",
                "length": self.length,
                "split_indices": self.split_indices.indices,
                "children": [child.to_dict() for child in self.children],
            }

    def get_root(self) -> Self:
        cur = self
        while cur.parent is not None:
            cur = cur.parent
        return cur

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_internal(self) -> bool:
        return bool(self.children)

    def delete_taxa(self, indices_to_delete: list[int]) -> Self:
        """Delete taxa and update indices/caches."""
        # First delete the taxa
        self._delete_taxa_internal(indices_to_delete)

        self._delete_superfluous_nodes()

        # Update order and reinitialize indices
        self._initialize_split_indices(self._encoding)

        # Clear all caches
        self._split_index = None  # Force rebuild of split index
        # Rebuild split index with new indices
        self.build_split_index()
        self.invalidate_caches(propagate_up=True)
        return self

    def _delete_taxa_internal(self, indices_to_delete: list[int]) -> Self:
        """Internal method for taxa deletion."""
        # Keep only children whose split indices contain elements not in indices_to_delete
        self.children = [
            child
            for child in self.children
            if any(idx not in indices_to_delete for idx in child.split_indices)
        ]

        # Update split indices for this node
        self.split_indices = Partition(
            tuple(idx for idx in self.split_indices if idx not in indices_to_delete),
            self._encoding,
        )

        # Recursively process children
        for child in self.children:
            child._delete_taxa_internal(indices_to_delete)

        return self

    def _delete_superfluous_nodes(self) -> Self:
        """Remove nodes with single children."""
        self.children = [
            self._get_end_child(cast(Self, child)) for child in self.children
        ]
        for child in self.children:
            child._delete_superfluous_nodes()
        return self

    def _get_end_child(self, node: Self) -> Self:
        """Get the furthest non-single-child descendant."""
        if len(node.children) != 1:
            return node
        return self._get_end_child(cast(Self, node.children[0]))

    def get_external_indices(self) -> list:
        """Get all leaf indices in traversal order."""
        indices = []
        for child in self.children:
            if child.children:
                indices.extend(child.get_external_indices())
            else:
                indices.append(child.name)
        return indices

    # ------------------------------------------------------------------------
    # (NEW) Subtree cache management
    # ------------------------------------------------------------------------
    def invalidate_caches(self, propagate_up: bool = True) -> None:
        """
        Invalidate cached subtree order and cost for this node and, optionally, all ancestors.
        """
        self._cached_subtree_order = None
        self._cached_subtree_cost = None
        self._cache_valid = False
        self._traverse_cache = None
        self._splits_cache = None
        if propagate_up and self.parent is not None:
            self.parent.invalidate_caches(propagate_up=True)

    def update_caches(self) -> None:
        """
        Update (recompute) cached subtree order and cost for this node.
        """
        self._cached_subtree_order = tuple(leaf.name for leaf in self.get_leaves())
        self._cached_subtree_cost = self.compute_subtree_cost()
        self._cache_valid = True

    def get_cached_subtree_cost(self) -> float:
        """
        Return cached subtree cost, updating if invalid.
        """
        if not self._cache_valid or self._cached_subtree_cost is None:
            self.update_caches()
        return self._cached_subtree_cost

    def compute_subtree_cost(self) -> float:
        """
        Compute the cost for the subtree rooted at this node.
        Placeholder: replace with actual cost/distance logic as needed.
        """
        # Example: sum of branch lengths in subtree
        cost = 0.0
        if self.length is not None:
            cost += self.length
        for child in self.children:
            cost += child.get_cached_subtree_cost()
        return cost

    def assign_internal_node_names(self):
        """
        Assigns a unique name to each internal node based on its descendant leaf names, sorted alphabetically and joined.
        Leaves retain their original names.
        """
        if not self.children:
            return self.name
        child_names = []
        for child in self.children:
            child_name = child.assign_internal_node_names()
            child_names.append(child_name)
        # Internal node name: join sorted unique descendant names
        self.name = ''.join(sorted(set(child_names)))
        return self.name
