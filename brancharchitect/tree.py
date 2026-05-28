from __future__ import annotations
import json
from enum import Enum
from typing import Optional, Any, Tuple, Dict, List, TypeAlias, cast

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from brancharchitect.elements.partition_set import PartitionSet, Partition

_SUPPORT_METADATA_KEYS = (
    "support",
    "bootstrap",
    "bootstrap_support",
    "bs",
    "ufboot",
    "UFBoot",
    "SH-aLRT",
    "sh_alrt",
    "alrt",
    "S",
)

_BOOTSTRAP_REPLICATE_FREQUENCY_SUPPORT_KINDS = {
    "bootstrap_replicate_split_frequency",
    "bootstrap_replicate_subtree_frequency",
}

_BOOTSTRAP_REPLICATE_FREQUENCY_LABELS = {
    "bootstrap_replicate_split_frequency": "Bootstrap Split Frequency",
    "bootstrap_replicate_subtree_frequency": "Bootstrap Subtree Frequency",
}
ReorderValue: TypeAlias = int | tuple[int, int, int] | list[int]


def _annotation_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "string"


def _annotation_field(
    path: List[str],
    label: str,
    value: Any,
    role: str,
    unit: Optional[str] = None,
    analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    field: Dict[str, Any] = {
        "path": path,
        "label": label,
        "value": value,
        "value_type": _annotation_value_type(value),
        "role": role,
    }
    if unit is not None:
        field["unit"] = unit
    if analysis is not None:
        field["analysis"] = analysis
    return field


def _field_key(path: List[str]) -> str:
    return ".".join(path)


def _single_value_support_annotation(
    support_kind: str,
    value: float,
) -> Dict[str, Any]:
    if support_kind == "bootstrap":
        path = ["support", "bootstrap", "value"]
        return _annotation_field(
            path,
            "Bootstrap",
            value,
            "branch_support",
            unit="percent",
            analysis={
                "type": "tree_inference",
                "method": "bootstrap",
            },
        )
    if support_kind == "ufboot":
        path = ["support", "iqtree", "ufboot"]
        return _annotation_field(
            path,
            "UFBoot",
            value,
            "branch_support",
            unit="percent",
            analysis={
                "type": "tree_inference",
                "method": "iqtree",
                "mode": "ufboot",
            },
        )
    if support_kind == "sh_alrt":
        path = ["support", "iqtree", "sh_alrt"]
        return _annotation_field(
            path,
            "SH-aLRT",
            value,
            "branch_support",
            unit="percent",
            analysis={
                "type": "tree_inference",
                "method": "iqtree",
                "mode": "sh_alrt",
            },
        )
    raise ValueError(f"Unsupported single-value branch support kind: {support_kind}")


def _bootstrap_replicate_frequency_annotation_fields(
    node: "Node",
    support_kind: str,
    value: float,
) -> Dict[str, Dict[str, Any]]:
    analysis = {
        "type": "rogue_taxa",
        "method": support_kind,
    }
    fields: Dict[str, Dict[str, Any]] = {}
    path = ["support", "bootstrap_rogue", "frequency"]
    fields[_field_key(path)] = _annotation_field(
        path,
        _BOOTSTRAP_REPLICATE_FREQUENCY_LABELS[support_kind],
        value,
        "branch_support",
        unit="percent",
        analysis=analysis,
    )
    for key, label in (
        ("replicate_count", "Replicate Count"),
        ("replicate_total", "Replicate Total"),
    ):
        metadata_value = _to_float_or_none(node.values.get(key))
        if metadata_value is not None:
            path = ["support", "bootstrap_rogue", key]
            fields[_field_key(path)] = _annotation_field(
                path,
                label,
                metadata_value,
                "branch_support_context",
                analysis=analysis,
            )
    return fields


def _to_float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def build_branch_annotation_fields(node: "Node") -> Dict[str, Dict[str, Any]]:
    """Build hierarchical branch annotation fields from labels and metadata."""
    fields: Dict[str, Dict[str, Any]] = {}
    if node.is_leaf():
        return fields

    internal_label = (node.name or "").strip()
    support_kind = str(node.values.get("support_kind", ""))
    if internal_label:
        raw_path = ["label", "raw_internal"]
        fields[_field_key(raw_path)] = _annotation_field(
            raw_path,
            "Raw Internal Label",
            internal_label,
            "source_annotation",
        )
        parts = internal_label.split("/")
        values = [_to_float_or_none(part) for part in parts]
        if all(value is not None for value in values):
            numeric_values = [float(value) for value in values if value is not None]
            if len(numeric_values) == 1:
                value = numeric_values[0]
                support_kind = support_kind or "bootstrap"
                if support_kind in _BOOTSTRAP_REPLICATE_FREQUENCY_SUPPORT_KINDS:
                    fields.update(
                        _bootstrap_replicate_frequency_annotation_fields(
                            node,
                            support_kind,
                            value,
                        )
                    )
                    return fields

                field = _single_value_support_annotation(support_kind, value)
                fields[_field_key(field["path"])] = field
                return fields
            if len(numeric_values) >= 2:
                analysis = {
                    "type": "tree_inference",
                    "method": "iqtree",
                    "mode": "sh_alrt_ufboot",
                }
                sh_path = ["support", "iqtree", "sh_alrt"]
                fields[_field_key(sh_path)] = _annotation_field(
                    sh_path,
                    "SH-aLRT",
                    numeric_values[0],
                    "branch_support",
                    unit="percent",
                    analysis=analysis,
                )
                uf_path = ["support", "iqtree", "ufboot"]
                fields[_field_key(uf_path)] = _annotation_field(
                    uf_path,
                    "UFBoot",
                    numeric_values[-1],
                    "branch_support",
                    unit="percent",
                    analysis=analysis,
                )
                return fields

    if support_kind in _BOOTSTRAP_REPLICATE_FREQUENCY_SUPPORT_KINDS:
        bootstrap_frequency = _to_float_or_none(node.values.get("bootstrap_frequency"))
        if bootstrap_frequency is not None:
            fields.update(
                _bootstrap_replicate_frequency_annotation_fields(
                    node,
                    support_kind,
                    bootstrap_frequency,
                )
            )
            return fields

    for key in _SUPPORT_METADATA_KEYS:
        if key not in node.values:
            continue
        support_value = _to_float_or_none(node.values[key])
        if support_value is None:
            continue
        path = ["support", "metadata", key]
        fields[_field_key(path)] = _annotation_field(
            path,
            key,
            support_value,
            "branch_support",
            unit="percent",
            analysis={
                "type": "tree_annotation",
                "method": "metadata_support",
            },
        )
        break

    for key, value in node.values.items():
        if key in _SUPPORT_METADATA_KEYS:
            continue
        if key in {
            "support_kind",
            "bootstrap_frequency",
            "replicate_count",
            "replicate_total",
        }:
            continue
        if isinstance(value, (str, int, float, bool, list)):
            path = ["metadata", str(key)]
            fields[_field_key(path)] = _annotation_field(
                path,
                str(key),
                value,
                "metadata",
            )

    return fields


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
    _split_index: Optional[Dict[Partition, "Node"]]
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

    def __str__(self) -> str:
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

    def build_split_index(self) -> None:
        self._split_index = {}
        stack = [self]
        while stack:
            node = stack.pop()
            self._split_index[node.split_indices] = node
            stack.extend(node.children)

    def _populate_split_index(self, node: Self) -> None:
        if self._split_index is None:
            self._split_index = {}
        stack = [node]
        while stack:
            current = stack.pop()
            self._split_index[current.split_indices] = current
            stack.extend(current.children)

    def find_node_by_split(self, target_split: Any) -> Optional["Node"]:
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

    # ------------------------------------------------------------------------
    # deep_copy (optimized iterative version to avoid function call overhead)
    # ------------------------------------------------------------------------
    def deep_copy(self, *, build_split_index: bool = True) -> Self:
        """Create a deep copy of this subtree using iterative stack-based traversal.

        This iterative approach eliminates Python function call overhead,
        providing ~2-3x speedup for large trees compared to recursive version.
        When build_split_index is False, the copied tree keeps the usual lazy
        lookup behavior and builds the root split index on first lookup.
        """
        object_new = object.__new__

        # Create root copy
        root_copy = object_new(type(self))
        root_copy.name = self.name
        root_copy.length = self.length if self.length is not None else 0.0
        root_copy.values = self.values.copy() if self.values else {}
        root_copy.split_indices = self.split_indices
        root_copy.taxa_encoding = self.taxa_encoding
        root_copy.parent = None
        root_copy.depth = None
        root_copy.list_index = None
        root_copy._traverse_cache = None
        root_copy._splits_cache = None
        root_copy._splits_with_leaves_cache = None
        root_copy._leaves_cache = None
        root_copy.children = []
        root_split_index: Dict[Partition, "Node"] | None = (
            {root_copy.split_indices: root_copy} if build_split_index else None
        )
        root_copy._split_index = root_split_index

        # Stack holds (original_node, copy_node) pairs to process
        stack: list[tuple[Self, Self]] = [(self, root_copy)]

        while stack:
            original, copy = stack.pop()

            # Process all children of current node
            for child in original.children:
                child_copy = object_new(type(child))
                child_copy.name = child.name
                child_copy.length = child.length if child.length is not None else 0.0
                child_copy.values = child.values.copy() if child.values else {}
                child_copy.split_indices = child.split_indices
                child_copy.taxa_encoding = child.taxa_encoding
                child_copy.parent = copy
                child_copy.depth = None
                child_copy.list_index = None
                child_copy._split_index = None
                child_copy._traverse_cache = None
                child_copy._splits_cache = None
                child_copy._splits_with_leaves_cache = None
                child_copy._leaves_cache = None
                child_copy.children = []
                copy.children.append(child_copy)
                if root_split_index is not None:
                    root_split_index[child_copy.split_indices] = child_copy

                # Only add to stack if child has children to process
                if child.children:
                    stack.append((child, child_copy))

        return root_copy

    # ------------------------------------------------------------------------
    # split_indices initialization (unchanged)
    # ------------------------------------------------------------------------

    def _initialize_split_indices(self, encoding: Dict[str, int]) -> None:
        """Initialize split indices using iterative post-order traversal."""
        stripped_encoding = {key.strip(): idx for key, idx in encoding.items()}
        stack: list[tuple["Node", bool]] = [(self, False)]

        try:
            while stack:
                node, visited = stack.pop()

                if not visited:
                    node.taxa_encoding = encoding
                    stack.append((node, True))
                    for child in reversed(node.children):
                        stack.append((child, False))
                    continue

                if not node.children:
                    if node.name in encoding:
                        idx = encoding[node.name]
                        node.split_indices = Partition.from_bitmask(1 << idx, encoding)
                        continue

                    found_idx = stripped_encoding.get(node.name.strip())
                    if found_idx is not None:
                        node.split_indices = Partition.from_bitmask(
                            1 << found_idx, encoding
                        )
                    else:
                        node.split_indices = Partition.from_bitmask(0, encoding)
                    continue

                combined_mask = 0
                for child in node.children:
                    combined_mask |= child.split_indices.bitmask
                node.split_indices = Partition.from_bitmask(combined_mask, encoding)
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
        # Invalidate all caches to ensure fresh state after initialization
        # This is important because tree construction may have set stale caches
        self.invalidate_caches(propagate_up=False, propagate_down=True)
        # Build split index ONCE at the root after all nodes are initialized
        # and stale caches have been cleared.
        self.build_split_index()

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
        nodes: list["Node"] = []
        stack: list["Node"] = [self]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)

        changed = False
        for node in reversed(nodes):
            original_order = [id(child) for child in node.children]
            node.children.sort(
                key=lambda child: (
                    min(child.split_indices) if child.split_indices else float("inf")
                )
            )
            changed = changed or original_order != [
                id(child) for child in node.children
            ]

        if changed:
            self.invalidate_caches(propagate_up=True, propagate_down=True)

    def collapse_unary_internal_nodes(self, preserve_lengths: bool = True) -> Self:
        """
        Collapse internal one-child nodes into their child.

        Unary internal nodes do not carry phylogenetic topology, but they do create
        duplicate split keys and visible extra edges in rendered trees. When
        preserving lengths, the removed node's branch length is added to its child
        so root-to-leaf distances are unchanged.
        """
        root = self.get_root()
        changed = False

        for node in reversed(root.traverse()):
            if node.parent is None or len(node.children) != 1:
                continue

            child = node.children[0]
            parent = node.parent

            if preserve_lengths:
                child.length = (child.length or 0.0) + (node.length or 0.0)

            try:
                index = parent.children.index(node)
            except ValueError as exc:
                raise ValueError(
                    "Unary node parent/child links are inconsistent"
                ) from exc

            parent.children[index] = child
            child.parent = parent
            node.parent = None
            node.children = []
            changed = True

        if changed:
            root.initialize_split_indices(root.taxa_encoding)

        return root

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

        if tuple(permutation) == self.get_current_order():
            return

        # 2. Map taxon names to their desired target index
        target_indices = {name: i for i, name in enumerate(permutation)}

        # 3. Bottom-up calculation of sort keys (Dynamic Programming)
        # Key: node_id -> value depending on strategy
        node_keys: Dict[int, ReorderValue] = {}

        def compute_keys(node: Self) -> ReorderValue:
            val: ReorderValue
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
                val = min(cast(int, child_val) for child_val in child_vals)
            elif strategy == ReorderStrategy.MAXIMUM:
                val = max(cast(int, child_val) for child_val in child_vals)
            elif strategy == ReorderStrategy.AVERAGE:
                total_sum = sum(
                    cast(tuple[int, int, int], child_val)[0] for child_val in child_vals
                )
                total_count = sum(
                    cast(tuple[int, int, int], child_val)[1] for child_val in child_vals
                )
                min_val = min(
                    cast(tuple[int, int, int], child_val)[2] for child_val in child_vals
                )
                val = (total_sum, total_count, min_val)
            else:  # MEDIAN
                # For median, we must collect all indices.
                # This is O(N log N) or O(N^2) worst case, but unavoidable for exact median.
                median_values: list[int] = []
                for child_val in child_vals:
                    median_values.extend(cast(list[int], child_val))
                val = median_values

            node_keys[id(node)] = val
            return val

        compute_keys(self)

        # 4. Define sort key extractor
        def get_sort_val(
            n: Self,
        ) -> int | float | tuple[float, int] | tuple[int, int, int] | list[int]:
            val = node_keys[id(n)]
            if strategy == ReorderStrategy.MINIMUM:
                return val
            elif strategy == ReorderStrategy.MAXIMUM:
                return val
            elif strategy == ReorderStrategy.AVERAGE:
                # Sort by average, break ties with min index
                tuple_val = cast(tuple[int, int, int], val)
                return (tuple_val[0] / tuple_val[1], tuple_val[2])
            else:  # MEDIAN
                # Sort indices to find median
                median_vals = cast(list[int], val)
                median_vals.sort()
                return median_vals[len(median_vals) // 2]

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

        leaves = [node for node in self.traverse() if not node.children]
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
        def node_to_dict(node: Self) -> Dict[str, Any]:
            if node.is_leaf():
                split_indices = list(node.split_indices.resolve_to_indices())
                name = node.name
            else:
                split_indices = list(node.split_indices.indices)
                name = ""

            node_dict: Dict[str, Any] = {
                "name": name,
                "length": node.length,
                "split_indices": split_indices,
                "children": [],
            }
            annotation_fields = build_branch_annotation_fields(node)
            if annotation_fields:
                node_dict["annotations"] = {
                    "fields": annotation_fields,
                }
            return node_dict

        root_dict = node_to_dict(self)
        stack: list[tuple["Node", Dict[str, Any]]] = [(self, root_dict)]

        while stack:
            node, serialized = stack.pop()
            child_dicts = [node_to_dict(child) for child in node.children]
            serialized["children"] = child_dicts
            for child, child_dict in reversed(list(zip(node.children, child_dicts))):
                stack.append((child, child_dict))

        return root_dict

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
        # Track a node guaranteed to survive compression for later use
        surviving_node = parent
        curr: Optional["Node"] = parent

        # Compress single-child nodes going up the tree (stops at root since root has no parent)
        while curr is not None and curr.parent is not None:
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
                    curr_len = curr.length
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

                # If curr was our surviving_node, update to child (which is now in the tree)
                if curr is surviving_node:
                    surviving_node = child

                curr = child  # Continue checking from this level (now attached to grandparent)
                continue  # Skip the curr = curr.parent at the end

            curr = curr.parent  # Move up to next ancestor

        # Handle root becoming single-child (special case - can't compress root normally)
        # Get the root node
        root = surviving_node.get_root()
        if len(root.children) == 1:
            # Root has single child - promote grandchildren to be root's children
            single_child = root.children[0]
            # Transfer all grandchildren to root
            root.children = single_child.children
            for grandchild in root.children:
                grandchild.parent = root
                # Optionally merge branch lengths
                if preserve_lengths and single_child.length is not None:
                    gc_len = grandchild.length if grandchild.length is not None else 0.0
                    grandchild.length = gc_len + single_child.length
            # Disconnect the single child
            single_child.parent = None
            single_child.children = []

        # Recompute splits (Stable Mode)
        # Use surviving_node to walk up (guaranteed to be in the tree)
        if mode == "stable":
            # Walk up from the surviving node
            cursor: Optional["Node"] = surviving_node
            while cursor is not None:
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
            # Full Rebuild - use surviving_node to get root
            remaining_leaves = sorted(
                [leaf.name for leaf in surviving_node.get_root().get_leaves()]
            )
            new_encoding = {name: i for i, name in enumerate(remaining_leaves)}
            surviving_node.get_root().initialize_split_indices(new_encoding)
        else:
            raise ValueError(f"Unknown pruning mode: {mode}")

        # Invalidate caches globally for safety - use surviving_node to get root
        surviving_node.get_root().invalidate_caches()

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
        self.invalidate_caches(propagate_up=True)
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
        Invalidate all traversal and split caches for this node.
        This should be called after any tree modification to ensure cache consistency.
        If propagate_up is True, also invalidate caches for all ancestors.
        If propagate_down is True, also invalidate caches for all descendants.
        """

        def clear(node: Self) -> None:
            node._traverse_cache = None
            node._splits_cache = None
            node._splits_with_leaves_cache = None
            node._split_index = None
            node._leaves_cache = None

        clear(self)

        if propagate_down:
            stack = list(self.children)
            while stack:
                node = stack.pop()
                clear(node)
                stack.extend(node.children)

        if propagate_up:
            parent = self.parent
            while parent is not None:
                clear(parent)
                parent = parent.parent

    def assign_internal_node_names(self) -> str:
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
        current: Optional["Node"] = self
        while current is not None:
            self_ancestors.add(current)
            current = current.parent

        # Find first common ancestor in other's path to root
        other_current: Optional["Node"] = other
        while other_current is not None:
            if other_current in self_ancestors:
                return other_current
            other_current = other_current.parent

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
        current: Optional["Node"] = self

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
        current: Optional["Node"] = descendant
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
        node1: Optional["Node"] = self.find_node_by_split(split1)
        node2: Optional["Node"] = self.find_node_by_split(split2)

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
