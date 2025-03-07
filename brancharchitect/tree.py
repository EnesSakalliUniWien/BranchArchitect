import json
from enum import Enum
from statistics import mean
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict, List, Generator
from brancharchitect.split import PartitionSet, Partition


class ReorderStrategy(Enum):
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MEDIAN = "median"


@dataclass()
class Node:
    children: List["Node"] = field(default_factory=list, compare=False)
    length: Optional[float] = field(default=None, compare=True)
    values: Dict[str, Any] = field(default_factory=dict, compare=True)
    split_indices: Partition = field(default_factory=tuple, compare=True)
    parent: Optional["Node"] = None
    _visual_order_indices: List[str] = field(default=None, compare=True)
    _encoding: Dict[str, int] = field(default_factory=list, compare=False)
    _cached_splits_without_leaves: Optional[PartitionSet] = field(
        default=None, init=False, compare=False
    )
    _cached_splits_with_leaves: Optional[PartitionSet] = field(
        default=None, init=False, compare=False
    )
    _split_index = None
    _list_inxdex = None

    # ------------------------------------------------------------------------
    # (NEW) Cache for get_current_order
    # ------------------------------------------------------------------------
    _cached_current_order: Optional[Tuple[str, ...]] = field(
        default=None, init=False, compare=False
    )  # (NEW)

    def __init__(
        self,
        children=None,
        name=None,
        length=None,
        values=None,
        split_indices=(),
        _visual_order_indices=None,
        _order=None,
        _encoding=None,
    ):
        self.children = children or []
        self.name = name
        self.length = length
        self.values = values or {}
        self.split_indices = split_indices
        self._visual_order_indices = _visual_order_indices
        self._order = _order or []
        self._cached_splits_without_leaves = None
        self._cached_splits_with_leaves = None
        self._split_index = None
        self._list_inxdex = None
        self._cached_current_order = None

        if not self._order:
            self._order = self.get_current_order()
        if not _encoding:
            self._encoding = {name: i for i, name in enumerate(self._order)}

        if not self._encoding:
            raise ValueError("Encoding dictionary cannot be empty")
        
    @property
    def leaves(self) -> List["Node"]:
        """
        Get all leaf nodes in the subtree rooted at this node.

        A leaf node is defined as a node with no children.

        Returns:
            List[Node]: List of all leaf nodes in this subtree.
        """
        if not self.children:
            return [self]  # This node is a leaf

        # Collect leaves from all children recursively
        result = []
        for child in self.children:
            result.extend(child.leaves)
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

    def to_splits(
        self, with_leaves=False
    ) -> PartitionSet[Partition]:
        splits = PartitionSet(look_up=self._encoding)
        for nd in self.traverse():
            if nd.children:
                splits.add(nd.split_indices)
            if with_leaves:
                splits.add(nd.split_indices)
        return splits

    def build_split_index(self):
        self._split_index = {}
        self._populate_split_index(self)

    def _populate_split_index(self, node):
        if node.split_indices is not None:
            self._split_index[node.split_indices] = node
        for ch in node.children:
            self._populate_split_index(ch)

    def find_node_by_split(self, target_split: Partition) -> Optional["Node"]:
        """Find a node by its split indices with more robust error handling."""
        try:
            if self._split_index is None:
                self.build_split_index()
            return self._split_index.get(target_split)
        except Exception as e:
            raise ValueError(f"Error finding node by split: {e}")

    # ------------------------------------------------------------------------
    # append_child sets parent pointer (pointer-based approach)
    # ------------------------------------------------------------------------
    def append_child(self, node: "Node") -> None:
        node.parent = self
        self.children.append(node)

    # ------------------------------------------------------------------------
    # deep_copy (unchanged, except we skip copying .parent)
    # ------------------------------------------------------------------------
    def deep_copy(self):
        new_node = Node(
            name=self.name,
            length=self.length,
            values=self.values.copy(),
            split_indices=self.split_indices,
            _visual_order_indices=(
                self._visual_order_indices.copy()
                if self._visual_order_indices
                else None
            ),
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
        self.split_indices = None
        
        # Validate encoding
        if not encoding:
            raise ValueError("Encoding dictionary cannot be empty")
            
        # Process children first
        for child in self.children:
            
            child._initialize_split_indices(encoding)
        
        try:
            if not self.children:
                # For leaf nodes, ensure the name exists in encoding
                if self.name not in encoding:
                    raise KeyError(f"Leaf name '{self.name}' not found in encoding")
                
                self.split_indices = Partition((encoding[self.name],), lookup=encoding)

            else:

                # For internal nodes, collect child indices
                idxs = []
                for ch in self.children:
                    if not hasattr(ch, 'split_indices') or ch.split_indices is None:
                        raise ValueError(f"Child node {ch.name} has no split indices")
                    idxs.extend(tuple(sorted(ch.split_indices)))
                if not idxs:
                    raise ValueError(f"No valid split indices found for node {self.name}")
                self.split_indices = Partition(tuple(sorted(idxs)), lookup=encoding)
                
            # Rebuild split index after modification
            self.build_split_index()
            
        except Exception as e:
            raise ValueError(f"Failed to initialize split indices: {str(e)}")

    # ------------------------------------------------------------------------
    # traversal, fix_child_order, to_hierarchy, etc.
    # ------------------------------------------------------------------------

    def traverse(self) -> Generator["Node", None, None]:
        yield self
        for ch in self.children:
            yield from ch.traverse()

    def _index(self, component: Tuple[str, ...]) -> Partition:
        return tuple(sorted(self._order.index(name) for name in component))

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

    def to_weighted_splits(self) -> Dict[Partition, float]:
        return {nd.split_indices: nd.length for nd in self.traverse()}

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

        self._visual_order_indices = {name: idx for idx, name in enumerate(permutation)}

        sorting_strategies = {
            ReorderStrategy.AVERAGE: lambda leaves: mean(
                self._visual_order_indices[leaf.name] for leaf in leaves
            ),
            ReorderStrategy.MAXIMUM: lambda leaves: max(
                self._visual_order_indices[leaf.name] for leaf in leaves
            ),
            ReorderStrategy.MINIMUM: lambda leaves: min(
                self._visual_order_indices[leaf.name] for leaf in leaves
            ),
            ReorderStrategy.MEDIAN: lambda leaves: sorted(
                self._visual_order_indices[leaf.name] for leaf in leaves
            )[len(leaves) // 2],
        }

        def _reorder(node: "Node") -> None:
            if node.children:
                for child in node.children:
                    _reorder(child)
                # Sort children using selected strategy
                strategy_fn = sorting_strategies[strategy]
                node.children.sort(key=lambda child: strategy_fn(child.get_leaves()))

        _reorder(self)

    def get_leaves(self) -> List["Node"]:
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
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
        if self.is_leaf():
            return {
                "name": self.name,
                "length": self.length,
                "split_indices": self.split_indices,
                "children": [],
            }
        else:
            return {
                "name": "",
                "length": self.length,
                "split_indices": self.split_indices,
                "children": [child.to_dict() for child in self.children],
            }

    def get_root(self) -> "Node":
        cur = self
        while cur.parent is not None:
            cur = cur.parent
        return cur

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_internal(self) -> bool:
        return bool(self.children)

    # ----------- Tree Operations -----------
    def midpoint_root(self) -> "Node":
        """
        Correct midpoint rooting implementation that creates new nodes when needed.
        Follows established phylogenetic algorithms for proper midpoint rooting.
        """
        # Create a deep copy to avoid modifying original tree
        working_tree = self.get_root().deep_copy()
        original_root = working_tree.get_root()
        leaves = original_root.get_leaves()

        if len(leaves) < 2:
            return original_root

        # 1. Find the tree diameter (longest path between any two leaves)
        L1, _ = _farthest_leaf_pointer(leaves[0])
        L2, diam_length = _farthest_leaf_pointer(L1)

        # 2. Find path between diameter endpoints
        path_nodes, path_dists = _build_pointer_path(L1, L2)
        half_distance = diam_length / 2
        accumulated = 0.0
        new_root = None

        # 3. Locate midpoint position
        for i in range(len(path_dists)):
            edge_length = path_dists[i]
            parent_node = path_nodes[i]
            child_node = path_nodes[i + 1]

            if accumulated + edge_length >= half_distance:
                # Calculate exact split point
                dist_from_parent = half_distance - accumulated
                dist_from_child = edge_length - dist_from_parent

                # 4. Create new node if needed
                if dist_from_parent > 0 and dist_from_child > 0:
                    # Split the edge
                    midpoint = Node(name="Midpoint", length=dist_from_parent)

                    # Update parent-child relationships
                    if child_node.parent == parent_node:  # Forward direction
                        # Remove child from parent
                        parent_node.children.remove(child_node)
                        # Add midpoint to parent
                        parent_node.children.append(midpoint)
                        midpoint.parent = parent_node
                        # Set up child node
                        child_node.parent = midpoint
                        child_node.length = dist_from_child
                        midpoint.children.append(child_node)
                    else:  # Reverse direction
                        # Remove parent from child
                        child_node.children.remove(parent_node)
                        # Add midpoint to child
                        child_node.children.append(midpoint)
                        midpoint.parent = child_node
                        # Set up parent node
                        parent_node.parent = midpoint
                        parent_node.length = dist_from_child
                        midpoint.children.append(parent_node)

                    new_root = midpoint
                else:
                    # Midpoint exactly at an existing node
                    new_root = parent_node if dist_from_parent == 0 else child_node

                break

            accumulated += edge_length

        # 5. Reroot the tree
        final_root = _flip_upward(new_root)

        # 6. Maintain tree properties
        final_root._order = original_root._order.copy()
        # Correctly call _initialize_split_indices without extra arguments
        final_root._initialize_split_indices()
        final_root._fix_child_order()

        return final_root

    def delete_taxa(self, indices_to_delete: list[int]) -> "Node":
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
        return self

    def _delete_taxa_internal(self, indices_to_delete: list[int]) -> "Node":
        """Internal method for taxa deletion."""
        # Keep only children whose split indices contain elements not in indices_to_delete
        self.children = [
            child
            for child in self.children
            if any(idx not in indices_to_delete for idx in child.split_indices)
        ]

        # Update split indices for this node
        self.split_indices = tuple(
            idx for idx in self.split_indices if idx not in indices_to_delete
        )

        # Recursively process children
        for child in self.children:
            child._delete_taxa_internal(indices_to_delete)

        return self

    def _delete_superfluous_nodes(self) -> "Node":
        """Remove nodes with single children."""
        self.children = [self._get_end_child(child) for child in self.children]
        for child in self.children:
            child._delete_superfluous_nodes()
        return self

    def _get_end_child(self, node: "Node") -> "Node":
        """Get the furthest non-single-child descendant."""
        if len(node.children) != 1:
            return node
        return self._get_end_child(node.children[0])

    def get_external_indices(self) -> list:
        """Get all leaf indices in traversal order."""
        indices = []
        for child in self.children:
            if child.children:
                indices.extend(child.get_external_indices())
            else:
                indices.append(child.name)
        return indices


# ---------------------------
# Midpoint Rooting Helpers
# ---------------------------
def _farthest_leaf_pointer(start: Node) -> Tuple[Node, float]:
    """
    Find the farthest leaf node from a starting node in a tree and its cumulative distance.

    This function traverses the tree bidirectionally (upward to parent and downward to children)
    using BFS and returns the leaf node with the maximum cumulative distance from the start node.

    Args:
        start (Node): The starting node in the tree. Each node must have:
            - `parent`: Parent node (None if root).
            - `children`: List of child nodes.
            - `length`: Edge distance to its parent (None or 0.0 if undefined).
            - `is_leaf()`: Method returning True if the node has no children.

    Returns:
        Tuple[Node, float]: The farthest leaf node and its distance from the start.

    Notes:
        - A "leaf" is a node with no children.
        - Traversal includes both parent and children directions.
        - Edge distances are determined dynamically:
            - When moving **to a parent**, the current node's `length` is used.
            - When moving **to a child**, the child's `length` (distance to its parent) is used.
        - If `length` is None or missing, it defaults to 0.0.

    Examples:
        Consider the following tree:
        ```
                A (root)
               / \\
            2.0  3.0
             /     \\
            B       C
             \\
              4.0
               \\
                D (leaf)
        ```
        - **Scenario 1**: Start at node `B`:
          - Traversal paths: `B → A (distance 2.0)`, `B → D (distance 4.0)`, `A → C (distance 3.0)`.
          - Farthest leaf is `C` with total distance `2.0 + 3.0 = 5.0`.
          - Returns: `(C, 5.0)`.

        - **Scenario 2**: Start at node `A`:
          - Traversal paths: `A → B (distance 2.0)`, `A → C (distance 3.0)`, `B → D (distance 4.0)`.
          - Farthest leaf is `D` with total distance `2.0 + 4.0 = 6.0`.
          - Returns: `(D, 6.0)`.
    """
    visited = set()
    queue = deque([(start, 0.0)])
    farthest, max_dist = start, 0.0

    while queue:
        node, dist = queue.popleft()
        if node.is_leaf() and dist > max_dist:
            farthest, max_dist = node, dist

        # Traverse parent and children
        for neighbor in [node.parent] + node.children:
            if neighbor and neighbor not in visited:
                visited.add(neighbor)
                edge_len = neighbor.length if neighbor.parent == node else node.length
                queue.append((neighbor, dist + (edge_len or 0.0)))

    return farthest, max_dist


def _build_pointer_path(L1: Node, L2: Node) -> Tuple[List[Node], List[float]]:
    """
    Find the shortest path between two nodes in a bifurcating tree and return the node sequence and edge distances.

    This function uses BFS to trace the path from `L1` to `L2`, exploring both parent (upward) and child (downward)
    directions. It returns the path as a list of nodes and the corresponding edge distances between consecutive nodes.

    Args:
        L1 (Node): The starting node. Must belong to the same bifurcating tree as `L2`.
        L2 (Node): The target node to find a path to.

    Returns:
        Tuple[List[Node], List[float]]: 
            - `path`: Ordered list of nodes from `L1` to `L2`.
            - `dists`: Edge distances between consecutive nodes in `path`.

    Notes:
        - **Tree Structure**: The tree is bifurcating (each internal node has exactly two children) and has four leaves.
        - **Edge Lengths**:
            - When moving **to a child**, the distance is the child's `length` (distance from the current node to the child).
            - When moving **to a parent**, the distance is the current node's `length` (distance from the parent to the current node).
            - Defaults to `0.0` if `length` is `None` or undefined.
        - **Shortest Path**: BFS guarantees the shortest path in terms of **number of edges**, not cumulative distance.

    Examples:
        Consider this bifurcating tree with four leaves (`D`, `E`, `F`, `G`):
        ```
                A (root)
               / \\
        2.0   B     C  3.0
             / \\   / \\
        1.0 D   E F   G 2.0
        ```
        - **Edge Lengths**:
          - `B.length = 2.0` (distance from `A` to `B`).
          - `C.length = 3.0` (distance from `A` to `C`).
          - `D.length = 1.0` (distance from `B` to `D`), `E.length = 1.5` (distance from `B` to `E`).
          - `F.length = 1.0` (distance from `C` to `F`), `G.length = 2.0` (distance from `C` to `G`).

        - **Scenario 1**: `L1 = B`, `L2 = F`
          - Path: `B → A → C → F`
          - Distances: `B`'s length (2.0), `C`'s length (3.0), `F`'s length (1.0)
          - Returns: `([B, A, C, F], [2.0, 3.0, 1.0])`

        - **Scenario 2**: `L1 = D`, `L2 = G`
          - Path: `D → B → A → C → G`
          - Distances: `D`'s length (1.0), `B`'s length (2.0), `C`'s length (3.0), `G`'s length (2.0)
          - Returns: `([D, B, A, C, G], [1.0, 2.0, 3.0, 2.0])`
    """
    parent_map = {}
    queue = deque([L1])
    visited = set([L1])

    while queue:
        node = queue.popleft()
        if node == L2:
            break

        # Explore parent and children
        for neighbor in [node.parent] + node.children:
            if neighbor and neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = (node, neighbor.length or node.length or 0.0)
                queue.append(neighbor)

    # Reconstruct path
    path, dists = [L2], []
    while path[-1] != L1:
        parent, length = parent_map[path[-1]]
        path.append(parent)
        dists.append(length)

    return path[::-1], dists[::-1]


def _flip_upward(new_root: Node) -> Node:
    """Proper implementation of tree reorientation with parent-child preservation"""
    current = new_root
    prev_node = None
    prev_length = 0.0

    while current.parent:
        parent = current.parent
        old_length = current.length or 0.0

        # Remove circular reference
        if parent in current.children:
            current.children.remove(parent)

        # Remove current from parent's children
        if current in parent.children:
            parent.children.remove(current)

        # Reverse relationship
        current.parent = prev_node
        current.length = prev_length

        if prev_node is not None:
            prev_node.children.append(current)

        # Move up the tree
        prev_node = current
        prev_length = old_length
        current = parent

    # Handle original root
    if current is not None:
        current.parent = prev_node
        current.length = prev_length
        if prev_node is not None and current not in prev_node.children:
            prev_node.children.append(current)

    return new_root
