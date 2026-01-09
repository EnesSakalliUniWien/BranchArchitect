from __future__ import annotations
import json
from enum import Enum
from statistics import mean
from typing import Optional, Any, Tuple, Dict, List

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from brancharchitect.elements.partition_set import PartitionSet, Partition


class ReorderStrategy(Enum):
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MEDIAN = "median"


class Node:
    """
    Tree node with optimized memory layout using __slots__.

    Using __slots__ provides:
    - ~20-30% faster attribute access
    - Reduced memory footprint (no per-instance __dict__)
    - Faster deep_copy operations
    """

    __slots__ = (
        "children",
        "parent",
        "name",
        "length",
        "values",
        "split_indices",
        "taxa_encoding",
        "depth",
        "list_index",
        "_split_index",
        "_cached_subtree_order",
        "_cached_subtree_cost",
        "_cache_valid",
        "_traverse_cache",
        "_splits_cache",
        "_splits_with_leaves_cache",
        "_leaves_cache",
    )

    # Type annotations (for static analysis, not runtime)
    children: List[Self]
    parent: Optional[Self]
    name: str
    length: Optional[float]
    values: Dict[str, Any]
    split_indices: Partition
    taxa_encoding: Dict[str, int]
    depth: Optional[int]
    list_index: Optional[int]
    _split_index: Optional[Dict[Partition, Self]]
    _cached_subtree_order: Optional[Tuple[str, ...]]
    _cached_subtree_cost: Optional[float]
    _cache_valid: bool
    _traverse_cache: Optional[List[Self]]
    _splits_cache: Optional[PartitionSet[Partition]]
    _splits_with_leaves_cache: Optional[PartitionSet[Partition]]
    _leaves_cache: Optional[List[Self]]

    def __init__(
        self,
        children: Optional[List[Self]] = None,
        name: str = "",
        length: float = 0.00001,
        values: Optional[Dict[str, Any]] = None,
        split_indices: Optional[Partition] = None,
        taxa_encoding: Optional[Dict[str, int]] = None,
        depth: Optional[int] = None,
    ):
        # Avoid mutable default arguments; create fresh containers
        self.children = list(children) if children is not None else []
        # Set parent pointers for all children
        for child in self.children:
            child.parent = self
        self.parent = None  # Initialize parent attribute
        self.name = name
        self.length = length
        self.values = dict(values) if values is not None else {}
        # Ensure split_indices is a Partition object
        if split_indices is None:
            self.split_indices = Partition((), taxa_encoding or {})
        elif isinstance(split_indices, tuple):
            self.split_indices = Partition(split_indices, taxa_encoding or {})
        else:
            self.split_indices = split_indices
        self._split_index = None
        self._cached_subtree_order = None
        self._cached_subtree_cost = None
        self._cache_valid = False
        self._traverse_cache = None
        self._splits_cache = None
        self._splits_with_leaves_cache = None
        self._leaves_cache = None
        self.list_index = None
        self.depth = depth

        # Encoding is the single source of truth for split_indices.
        # Derive encoding from current leaves only if not provided.
        if taxa_encoding is None:
            leaf_order = list(self.get_current_order())
            self.taxa_encoding = {name: i for i, name in enumerate(leaf_order)}
        else:
            self.taxa_encoding = taxa_encoding

        if not self.taxa_encoding:
            raise ValueError("Encoding dictionary cannot be empty")

        # Always ensure split_indices is a Partition (already handled above)

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
            # Directly extend with child.leaves since it already returns List[Self]
            result.extend(child.leaves)
        return result

    # ------------------------------------------------------------------------
    # Equality & hashing
    # ------------------------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        """
        Check if two trees are topologically identical.

        Compares the complete set of splits (internal tree structure),
        not just the root split. Two trees are equal if and only if they
        have the exact same set of internal splits.

        The comparison uses PartitionSet.__eq__ which compares the bitmask sets,
        ensuring accurate topological equality checking.

        Note: This is more expensive than comparing just root splits, but
        provides accurate topological equality. Use `split_indices` directly
        if you only need to check if trees have the same leaf set.
        """
        if not isinstance(other, Node):
            return NotImplemented

        # Quick check: if root splits differ, trees are definitely different
        if self.split_indices != other.split_indices:
            return False

        # Full topology check: compare all internal splits using PartitionSet equality
        # PartitionSet.__eq__ compares _bitmask_set internally, which is efficient
        splits_self = self.to_splits()
        splits_other = other.to_splits()

        # PartitionSet inherits from MutableSet, so == compares the sets
        return splits_self == splits_other

    def __hash__(self) -> int:
        """
        Hash based on split_indices to maintain consistency with __eq__.

        Note: Since __eq__ now compares full topology, ideally we would hash
        the full split set. However, computing to_splits() for every hash
        operation would be expensive. We keep the hash based on split_indices
        (root split) as a performance optimization, accepting that hash
        collisions are possible for trees with same leaves but different
        internal structure. This is acceptable because:
        1. Equality still works correctly (uses full topology)
        2. Hash collisions just mean slower lookups in dicts/sets, not incorrectness
        """
        return hash(self.split_indices)

    def __repr__(self) -> str:
        return f"Node('{self.name}')"

    def __str__(self):
        return str(tuple(sorted(self.get_current_order())))

    # ------------------------------------------------------------------------
    # Splits & related caching
    # ------------------------------------------------------------------------s

    def to_splits(self, with_leaves: bool = False) -> PartitionSet[Partition]:
        """
        Return the set of splits (PartitionSet) for the subtree rooted at this node.
        Uses a recursive accumulation strategy for O(N) performance.
        """
        if not with_leaves and self._splits_cache is not None:
            return self._splits_cache
        if with_leaves and self._splits_with_leaves_cache is not None:
            return self._splits_with_leaves_cache

        splits: PartitionSet[Partition] = PartitionSet(encoding=self.taxa_encoding)

        to_add: List[Partition] = []

        for nd in self.traverse():
            # An internal node always defines a split. A leaf node only does if with_leaves is True.
            if nd.children:
                # Ensure the split is not empty before adding
                if nd.split_indices:
                    to_add.append(nd.split_indices)
            elif with_leaves:
                if nd.split_indices:
                    to_add.append(nd.split_indices)

        splits.update(to_add)

        if not with_leaves:
            self._splits_cache = splits
        else:
            self._splits_with_leaves_cache = splits
        return splits

    def build_split_index(self):
        self._split_index = {}
        self._populate_split_index(self)

    def _populate_split_index(self, node: Self) -> None:
        if self._split_index is None:
            self._split_index = {}

        self._split_index[node.split_indices] = node
        for ch in node.children:
            self._populate_split_index(ch)

    def find_node_by_split(self, target_split: Any) -> Optional[Self]:
        """
        Find a node by its split indices (accepts tuple or Partition).

        Strict encoding policy:
        - If a Partition is provided and its encoding differs from this tree's
          taxa_encoding, a ValueError is raised. Callers must re-encode upstream.
        """
        try:
            if self._split_index is None:
                self.build_split_index()

            # Convert tuple to Partition if needed (only if it's not already a Partition)
            if not isinstance(target_split, Partition):
                target_split = Partition(tuple(target_split), self.taxa_encoding)
            else:
                if target_split.encoding != self.taxa_encoding:
                    raise ValueError(
                        "Cannot search for split with different encoding. "
                        "Provide a Partition using this tree's taxa_encoding."
                    )

            # At this point _split_index should not be None due to build_split_index
            if self._split_index is not None:
                return self._split_index.get(target_split)
            return None
        except Exception as e:
            raise ValueError(f"Error finding node by split: {e}")

    # ------------------------------------------------------------------------
    # append_child sets parent pointer (pointer-based approach)
    # ------------------------------------------------------------------------
    def append_child(self, node: Self) -> None:
        node.parent = self
        self.children.append(node)
        # Invalidate all caches, including splits cache, after tree modification
        self.invalidate_caches(propagate_up=True)

    # Shared empty dict for deep_copy optimization (avoids creating new empty dicts)
    _EMPTY_VALUES: Dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # deep_copy (optimized iterative version to avoid function call overhead)
    # ------------------------------------------------------------------------
    def _create_shallow_node_copy(self) -> Self:
        """Create a shallow copy of this node without copying children.

        Internal helper for deep_copy - creates node with all attributes
        but children list is empty.
        """
        new_node = object.__new__(type(self))
        new_node.name = self.name
        new_node.length = self.length if self.length is not None else 0.0
        new_node.values = self.values.copy() if self.values else Node._EMPTY_VALUES
        new_node.split_indices = self.split_indices
        new_node.taxa_encoding = self.taxa_encoding
        new_node.parent = None
        new_node.depth = None
        new_node.list_index = None
        new_node._split_index = None
        new_node._cached_subtree_order = None
        new_node._cached_subtree_cost = None
        new_node._cache_valid = False
        new_node._traverse_cache = None
        new_node._splits_cache = None
        new_node._splits_with_leaves_cache = None
        new_node._leaves_cache = None
        new_node.children = []
        return new_node

    def deep_copy(self) -> Self:
        """Create a deep copy of this subtree using iterative stack-based traversal.

        This iterative approach eliminates Python function call overhead,
        providing ~2-3x speedup for large trees compared to recursive version.
        """
        # Create root copy
        root_copy = self._create_shallow_node_copy()

        # Stack holds (original_node, copy_node) pairs to process
        stack: list[tuple[Self, Self]] = [(self, root_copy)]

        while stack:
            original, copy = stack.pop()

            # Process all children of current node
            for child in original.children:
                child_copy = child._create_shallow_node_copy()
                child_copy.parent = copy
                copy.children.append(child_copy)

                # Only add to stack if child has children to process
                if child.children:
                    stack.append((child, child_copy))

        return root_copy

    # ------------------------------------------------------------------------
    # split_indices initialization (unchanged)
    # ------------------------------------------------------------------------

    def _initialize_split_indices(self, encoding: Dict[str, int]) -> None:
        """Initialize split indices with better error handling and validation.

        Note: This is the internal recursive method. It does NOT call build_split_index()
        to avoid O(N²) complexity. The public initialize_split_indices() calls
        build_split_index() once at the end.
        """
        # Set the encoding on this node
        self.taxa_encoding = encoding

        # Process children first (post-order traversal)
        for child in self.children:
            child._initialize_split_indices(encoding)

        try:
            if not self.children:
                # Leaf node - must have a name in the encoding
                if self.name in encoding:
                    # Use from_bitmask for faster creation (avoids sorting/set operations)
                    idx = encoding[self.name]
                    self.split_indices = Partition.from_bitmask(1 << idx, encoding)
                else:
                    # Fallback: try matching stripped names (slower path)
                    stripped_name = self.name.strip()
                    found_idx = None
                    for key, idx in encoding.items():
                        if key.strip() == stripped_name:
                            found_idx = idx
                            break

                    if found_idx is not None:
                        # Use from_bitmask for faster creation
                        self.split_indices = Partition.from_bitmask(
                            1 << found_idx, encoding
                        )
                    else:
                        # Internal node that became a leaf after deletion
                        self.split_indices = Partition.from_bitmask(0, encoding)
            else:
                # For internal nodes, collect child indices using bitmasks for speed
                combined_mask = 0
                for ch in self.children:
                    combined_mask |= ch.split_indices.bitmask

                self.split_indices = Partition.from_bitmask(combined_mask, encoding)

        except Exception as e:
            raise ValueError(f"Failed to initialize split indices: {str(e)}")

    def initialize_split_indices(self, encoding: Dict[str, int]) -> None:
        """
        Public method to initialize split indices for the tree.

        This method recursively initializes split indices for all nodes in the tree,
        starting from the current node. It ensures that each node has a proper
        Partition object representing its split.

        Args:
            encoding: Dictionary mapping taxon names to their integer indices

        Raises:
            ValueError: If initialization fails due to invalid encoding or tree structure
        """
        self._initialize_split_indices(encoding)
        # Build split index ONCE at the root after all nodes are initialized
        # This avoids O(N²) complexity from calling it at every node
        self.build_split_index()
        # Invalidate all caches to ensure fresh state after initialization
        # This is important because tree construction may have set stale caches
        self.invalidate_caches(propagate_up=False, propagate_down=True)

    # ------------------------------------------------------------------------
    # traversal, fix_child_order, to_hierarchy, etc.
    # ------------------------------------------------------------------------

    def traverse(self) -> List[Self]:
        """
        Return a list of all nodes in the subtree rooted at this node (Pre-order).
        Uses an iterative stack approach to avoid O(N^2) list extensions and recursion depth issues.
        """
        if self._traverse_cache is not None:
            return self._traverse_cache

        nodes: List[Self] = []
        stack: List[Self] = [self]

        while stack:
            current = stack.pop()
            nodes.append(current)
            # Add children in reverse to maintain left-to-right visit order (pre-order)
            for child in reversed(current.children):
                stack.append(child)

        self._traverse_cache = nodes
        return nodes

    def names_to_partition(self, names: Tuple[str, ...]) -> Partition:
        """
        Convert a tuple of taxon names to a Partition using this tree's taxa_encoding.

        This is the preferred API over the legacy `_index` helper.
        """
        try:
            indices = tuple(sorted(self.taxa_encoding[name] for name in names))
        except KeyError as e:
            raise ValueError(
                f"Unknown taxon name '{e.args[0]}' for this tree's encoding"
            )
        return Partition(indices, self.taxa_encoding)

    def fix_child_order(self) -> None:
        self.children.sort(
            key=lambda node: min(node.split_indices)
            if node.split_indices
            else float("inf")
        )
        for child in self.children:
            child.fix_child_order()

    def to_hierarchy(self) -> Dict[str, Any]:
        return {
            "name": self.name or "Internal",
            "children": (
                [c.to_hierarchy() for c in self.children] if self.children else []
            ),
            "values": self.values,
        }

    def swap_children(self) -> None:
        """
        Reverses the order of all children in place.

        This method works for any number of children (2 or more) and is a key
        operation for testing alternative node orientations. It correctly
        invalidates all necessary caches after the modification.
        """
        if len(self.children) >= 2:
            # Use list.reverse() to handle any number of children, not just the first two.
            self.children.reverse()

            # Invalidate all caches, as the leaf order has changed.
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
        """
        Reorder the children of this subtree to match a desired leaf permutation.

        Stability guarantees:
        - If the current leaf order of a node already matches the desired permutation
          for that node, the node and all of its descendants are left untouched.
        - Subtrees that are already aligned keep their internal ordering, preventing
          churn in unaffected regions when only a sibling needs to move.
        """
        # 1. Validate permutation
        tree_taxa = {leaf.name for leaf in self.get_leaves()}
        if set(permutation) != tree_taxa:
            raise ValueError(
                "Permutation must include all taxa in the tree.", permutation, tree_taxa
            )

        # 2. Map taxon names to their desired target index
        target_indices = {name: i for i, name in enumerate(permutation)}

        # 3. Bottom-up calculation of sort keys (Dynamic Programming)
        # Key: node_id -> value depending on strategy
        node_keys: Dict[int, Any] = {}

        def compute_keys(node: Self) -> Any:
            if not node.children:
                # Leaf: return its target index
                idx = target_indices[node.name]
                if strategy == ReorderStrategy.MINIMUM:
                    val = idx
                elif strategy == ReorderStrategy.MAXIMUM:
                    val = idx
                elif strategy == ReorderStrategy.AVERAGE:
                    val = (idx, 1, idx)  # sum, count, min (tie-breaker)
                else:  # MEDIAN
                    val = [idx]
                node_keys[id(node)] = val
                return val

            # Internal: recurse on children first
            child_vals = [compute_keys(child) for child in node.children]

            if strategy == ReorderStrategy.MINIMUM:
                val = min(child_vals)
            elif strategy == ReorderStrategy.MAXIMUM:
                val = max(child_vals)
            elif strategy == ReorderStrategy.AVERAGE:
                total_sum = sum(v[0] for v in child_vals)
                total_count = sum(v[1] for v in child_vals)
                min_val = min(v[2] for v in child_vals)
                val = (total_sum, total_count, min_val)
            else:  # MEDIAN
                # For median, we must collect all indices.
                # This is O(N log N) or O(N^2) worst case, but unavoidable for exact median.
                val = []
                for v in child_vals:
                    val.extend(v)

            node_keys[id(node)] = val
            return val

        compute_keys(self)

        # 4. Define sort key extractor
        def get_sort_val(n: Self) -> Any:
            val = node_keys[id(n)]
            if strategy == ReorderStrategy.MINIMUM:
                return val
            elif strategy == ReorderStrategy.MAXIMUM:
                return val
            elif strategy == ReorderStrategy.AVERAGE:
                # Sort by average, break ties with min index
                return (val[0] / val[1], val[2])
            else:  # MEDIAN
                # Sort indices to find median
                val.sort()
                return val[len(val) // 2]

        # 5. Top-down reordering using the pre-computed keys
        def apply_reordering(node: Self) -> bool:
            if not node.children:
                return False

            changed = False
            # Recurse first (post-order) or last (pre-order)?
            # Sorting children doesn't affect children's internal order, so order doesn't matter much.
            # But let's do children first to be safe.
            for child in node.children:
                changed = apply_reordering(child) or changed

            # Sort children in-place
            # Check if sort is needed to avoid unnecessary writes/invalidation
            # Create a list of (key, child) tuples to avoid recomputing key during sort
            children_with_keys = [
                (get_sort_val(child), child) for child in node.children
            ]
            children_with_keys.sort(key=lambda x: x[0])

            sorted_children = [child for _, child in children_with_keys]

            # Use identity check because Node equality is topological (leaves are equal)
            if [id(c) for c in sorted_children] != [id(c) for c in node.children]:
                node.children = sorted_children
                changed = True

            return changed

        if apply_reordering(self):
            self.invalidate_caches(propagate_up=True)

    def get_leaves(self) -> List[Self]:
        """
        Return all leaf nodes in the subtree rooted at this node.
        Uses caching for performance - cache is invalidated when tree structure changes.
        """
        if self._leaves_cache is not None:
            return self._leaves_cache

        if not self.children:
            self._leaves_cache = [self]
            return self._leaves_cache

        leaves: List[Self] = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        self._leaves_cache = leaves
        return leaves

    # ------------------------------------------------------------------------
    # (NEW) Cached get_current_order
    # ------------------------------------------------------------------------
    def get_current_order(self) -> tuple[str, ...]:
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
                length_str = (
                    f"{float(self.length):.6f}"
                    if self.length is not None
                    else "0.000000"
                )
                return f"{child_str}{self.name or ''}{meta}:{length_str}"
            else:
                return f"{child_str}{self.name or ''}{meta}"
        else:
            if lengths:
                length_str = (
                    f"{float(self.length):.6f}"
                    if self.length is not None
                    else "0.000000"
                )
                return f"{self.name or ''}{meta}:{length_str}"
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
                "split_indices": list(self.split_indices.resolve_to_indices()),
                "children": [],
            }
        else:
            return {
                "name": "",
                "length": self.length,
                "split_indices": list(self.split_indices.indices),
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

    def remove_subtree(
        self,
        target: "Node",
        mode: str = "stable",  # "stable" or "shrink"
        preserve_lengths: bool = True,
    ) -> None:
        """
        Remove the subtree rooted at 'target' from this tree.

        Args:
            target: The node object to remove. Must be a descendant of self.
            mode: "stable" keeps original taxa_encoding (removed taxa bits become 0).
                  "shrink" rebuilds encoding map (indices shift, expensive).
            preserve_lengths: If True, merges branch lengths when compressing single-child nodes.

        Raises:
            ValueError: If target is root or not found in parent's children.
        """
        if target.parent is None:
            raise ValueError("Cannot remove root node")

        parent = target.parent
        try:
            parent.children.remove(target)
        except ValueError:
            raise ValueError(
                f"Target node {target.name} not found in parent's children list."
            )
        target.parent = None

        # Prune single-child chain growing upwards (Compress linear segments)
        curr = parent
        while curr.parent is not None:  # Stop at root
            if not curr.children:
                # Became a leaf (internal node with all parts removed)
                # Its split mask effectively becomes 0 in stable mode
                pass
            elif len(curr.children) == 1:
                # Compression needed
                # A -> B(curr) -> C(child)  ==>  A -> C
                child = curr.children[0]
                grandparent = curr.parent

                # Merge logic
                if preserve_lengths and curr.length is not None:
                    child_len = child.length if child.length is not None else 0.0
                    curr_len = curr.length if curr.length is not None else 0.0
                    child.length = child_len + curr_len

                # Pointer updates
                child.parent = grandparent
                if grandparent:
                    # Replace curr with child in grandparent's list
                    # Use index-based replacement for safety if duplicate nodes exist (unlikely in tree)
                    try:
                        idx = grandparent.children.index(curr)
                        grandparent.children[idx] = child
                    except ValueError:
                        # Fallback if list consistency issues
                        grandparent.children.remove(curr)
                        grandparent.children.append(child)

                curr = child  # Continue checking from this level (now attached to grandparent)

            curr = curr.parent  # Move up to next ancestor

        # Recompute splits (Stable Mode)
        if mode == "stable":
            # Walk up from the modification point
            cursor = parent
            while cursor:
                # Fast recalculation using existing encoding
                if cursor.is_leaf():
                    # If it was internal and became leaf, mask is 0
                    if cursor.name not in cursor.taxa_encoding:
                        cursor.split_indices = Partition.from_bitmask(
                            0, cursor.taxa_encoding
                        )
                else:
                    new_mask = 0
                    for child in cursor.children:
                        new_mask |= child.split_indices.bitmask
                    cursor.split_indices = Partition.from_bitmask(
                        new_mask, cursor.taxa_encoding
                    )
                cursor = cursor.parent
        elif mode == "shrink":
            # Full Rebuild
            remaining_leaves = sorted([l.name for l in self.get_root().get_leaves()])
            new_encoding = {name: i for i, name in enumerate(remaining_leaves)}
            self.get_root().initialize_split_indices(new_encoding)
        else:
            raise ValueError(f"Unknown pruning mode: {mode}")

        # Invalidate caches globally for safety
        self.get_root().invalidate_caches()

    def find_leaf_by_name(self, name: str) -> Optional["Node"]:
        """
        Find a leaf node by name using safe traversal (O(N) fallback).
        Does not rely on cached split indices which might be stale.
        """
        for node in self.traverse():
            if node.is_leaf() and node.name == name:
                return node
        return None

    def find_node_by_bitmask(self, bitmask: int) -> Optional["Node"]:
        """
        Find node with specific split bitmask.
        Safe to use during pruning if implemented via traversal or fresh index.
        """
        # Try cache first? No, explicit request to NOT depend on potentially stale index.
        # But we can try _split_index if valid?
        # Safe fallback logic:
        for node in self.traverse():
            if node.split_indices.bitmask == bitmask:
                return node
        return None

    def delete_taxa(self, indices_to_delete: list[int]) -> Self:
        """
        Delete taxa and update indices/caches.
        This will invalidate all caches, including the splits cache, to ensure correctness.
        """
        # Create deletion mask once for efficiency
        deletion_mask = 0
        for idx in indices_to_delete:
            deletion_mask |= 1 << idx

        # First delete the taxa
        self._delete_taxa_internal(deletion_mask)

        self._prune_single_child_nodes()

        # Update order and reinitialize indices
        self._initialize_split_indices(self.taxa_encoding)
        self.build_split_index()  # Rebuild index after deletion

        # Debug: Log the leaves after deletion
        try:
            from brancharchitect.logger.debug import jt_logger

            if not jt_logger.disabled:
                remaining_leaves = [leaf.name for leaf in self.get_leaves()]
                taxa_to_delete_names = [
                    name
                    for name, idx in self.taxa_encoding.items()
                    if idx in indices_to_delete
                ]
                jt_logger.info(
                    f"After deleting indices {taxa_to_delete_names}, remaining leaves: {remaining_leaves}"
                )
        except Exception:
            pass

        # Clear all caches
        self._split_index = None  # Force rebuild of split index
        # Rebuild split index with new indices
        self.build_split_index()
        self.invalidate_caches(propagate_up=True)
        return self

    def _delete_taxa_internal(self, deletion_mask: int) -> Self:
        """
        Internal method for taxa deletion. Optimized for performance by using bitmasks.
        """
        # Keep only children whose split indices contain elements not in indices_to_delete
        self.children = [
            child
            for child in self.children
            if (child.split_indices.bitmask & ~deletion_mask) != 0
        ]

        # Update split indices for this node using bitmask
        new_mask = self.split_indices.bitmask & ~deletion_mask
        self.split_indices = Partition.from_bitmask(new_mask, self.taxa_encoding)

        # Recursively process children
        for child in self.children:
            child._delete_taxa_internal(deletion_mask)
        return self

        return self

    def _prune_single_child_nodes(self) -> Self:
        """Remove internal nodes with exactly one child by connecting their child directly to the parent."""
        # Replace each child with the deepest descendant that does not have exactly one child
        new_children = [self._get_end_child(child) for child in self.children]
        # Reattach and fix parent pointers
        self.children = new_children
        for child in self.children:
            child.parent = self
            child._prune_single_child_nodes()
        return self

    def _get_end_child(self, node: Self) -> Self:
        """Get the furthest non-single-child descendant."""
        if len(node.children) != 1:
            return node
        return self._get_end_child(node.children[0])

    # ------------------------------------------------------------------------
    # (NEW) Subtree cache management
    # ------------------------------------------------------------------------
    def invalidate_caches(
        self, propagate_up: bool = True, propagate_down: bool = True
    ) -> None:
        """
        Invalidate all caches for this node, including splits cache, subtree order, and cost.
        This should be called after any tree modification to ensure cache consistency.
        If propagate_up is True, also invalidate caches for all ancestors.
        If propagate_down is True, also invalidate caches for all descendants.
        """
        self._cached_subtree_order = None
        self._cached_subtree_cost = None
        self._cache_valid = False
        self._traverse_cache = None
        self._splits_cache = None
        self._splits_with_leaves_cache = None
        self._split_index = None  # Clear split index to force rebuild
        self._leaves_cache = None  # Clear leaves cache

        # Propagate down to children
        if propagate_down:
            for child in self.children:
                child.invalidate_caches(propagate_up=False, propagate_down=True)

        # Propagate up to parents
        if propagate_up and self.parent is not None:
            self.parent.invalidate_caches(propagate_up=True, propagate_down=False)

    def update_caches(self) -> None:
        """
        Update (recompute) cached subtree order and cost for this node.
        """
        self._cached_subtree_order = tuple(str(leaf.name) for leaf in self.get_leaves())
        self._cached_subtree_cost = self.compute_subtree_cost()
        self._cache_valid = True

    def get_cached_subtree_cost(self) -> float:
        """
        Return cached subtree cost, updating if invalid.
        """
        if not self._cache_valid or self._cached_subtree_cost is None:
            self.update_caches()
        return (
            self._cached_subtree_cost if self._cached_subtree_cost is not None else 0.0
        )

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

    def assign_internal_node_names(self):  # -> None | Any | str | LiteralString:
        """
        Assigns a unique name to each internal node based on its descendant leaf names, sorted alphabetically and joined.
        Leaves retain their original names.
        """
        if not self.children:
            return self.name
        child_names: List[str] = []
        for child in self.children:
            child_name: str | None = child.assign_internal_node_names()
            if child_name:
                child_names.append(child_name)
        # Internal node name: join sorted unique descendant names
        self.name: str = "".join(sorted(set(child_names)))
        return self.name

    def find_lowest_common_ancestor(self, other: "Node") -> Optional["Node"]:
        """
        Find the lowest common ancestor (LCA) of this node and another node.

        Args:
            other: The other node to find LCA with

        Returns:
            Node representing the LCA, or None if no common ancestor exists
        """
        if self is other:
            return self

        # Get paths to root for both nodes
        self_ancestors: set["Node"] = set()
        current = self
        while current is not None:
            self_ancestors.add(current)
            current = current.parent

        # Find first common ancestor in other's path to root
        current = other
        while current is not None:
            if current in self_ancestors:
                return current
            current = current.parent

        return None

    def path_to_ancestor(self, ancestor: "Node") -> List["Node"]:
        """
        Get the path from this node up to (but excluding) the specified ancestor.

        Args:
            ancestor: The target ancestor node

        Returns:
            List[Node] from self up to (excluding) ancestor, empty if ancestor not found
        """
        path: List["Node"] = []
        current = self

        while current is not None and current is not ancestor:
            path.append(current)
            current = current.parent

        # Return path only if we found the ancestor
        return path if current is ancestor else []

    def path_from_ancestor(self, descendant: "Node") -> List["Node"]:
        """
        Get the path from this node down to the specified descendant.

        Args:
            descendant: The target descendant node

        Returns:
            List[Node] from self down to descendant (excluding self), empty if descendant not found
        """
        if self is descendant:
            return []

        path: List["Node"] = []
        current = descendant
        while current is not None and current is not self:
            path.append(current)
            current = current.parent

        # If we reached None, self is not an ancestor of descendant
        if current is None:
            return []

        # path is currently [descendant, ..., child_of_self]. Reverse it.
        return path[::-1]

    def find_path_between_splits(
        self, split1: Partition, split2: Partition
    ) -> List["Node"]:
        """
        Find the path between two nodes identified by their split indices.

        Args:
            split1: Partition representing the first node
            split2: Partition representing the second node

        Returns:
            List[Node] representing the complete path from split1 to split2,
            empty list if either split not found or no path exists
        """
        # Find nodes corresponding to the splits
        node1: Self | None = self.find_node_by_split(split1)
        node2: Self | None = self.find_node_by_split(split2)

        if node1 is None or node2 is None:
            return []

        # Handle same node case
        if node1 is node2:
            return [node1]

        # Find LCA
        lca = node1.find_lowest_common_ancestor(node2)
        if lca is None:
            return []

        # Build complete path: node1 -> LCA -> node2
        path_to_lca = node1.path_to_ancestor(lca)
        path_from_lca = lca.path_from_ancestor(node2)

        # Combine paths: upward + LCA + downward
        complete_path = path_to_lca + [lca] + path_from_lca

        return complete_path
