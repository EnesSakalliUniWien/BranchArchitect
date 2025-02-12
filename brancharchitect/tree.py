import json
import itertools
from enum import Enum
from statistics import mean
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict, List, Generator
from brancharchitect.split import IndexedSplitSet, SplitIndices


class ReorderStrategy(Enum):
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MEDIAN = "median"


@dataclass()
class Node:
    children: List["Node"] = field(default_factory=list, compare=False)
    name: Optional[str] = field(default=None, compare=False)
    length: Optional[float] = field(default=None, compare=True)
    values: Dict[str, Any] = field(default_factory=dict, compare=True)
    split_indices: SplitIndices = field(default_factory=tuple, compare=True)
    leaf_name: Optional[str] = field(default=None, compare=False)
    parent: Optional["Node"] = None
    _visual_order_indices: List[str] = field(default=None, compare=True)
    _order: List[str] = field(default_factory=list, compare=False)
    _cached_splits: Optional[set] = field(default=None, init=False, compare=False)
    _split_index = None
    _list_index = None

    # ------------------------------------------------------------------------
    # (NEW) Cache for get_current_order
    # ------------------------------------------------------------------------
    _cached_current_order: Optional[Tuple[str, ...]] = field(
        default=None, init=False, compare=False
    )  # (NEW)

    # ------------------------------------------------------------------------
    # Equality & hashing (unchanged)
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
        return str(self.get_current_order())

    # ------------------------------------------------------------------------
    # Save/restore & ID assignment (unchanged)
    # ------------------------------------------------------------------------
    def save_state(self):
        state = {}
        state["children_order"] = [child.deep_copy() for child in self.children]
        return state

    def restore_state(self, state):
        self.children = state["children_order"]

    def assign_node_ids(self):
        counter = itertools.count()
        for node in self.traverse():
            node.node_id = next(counter)

    # ------------------------------------------------------------------------
    # Splits & related caching (unchanged)
    # ------------------------------------------------------------------------
    def to_splits(self, with_leaves=False) -> set[SplitIndices]:
        if self._cached_splits is not None:
            return self._cached_splits
        splits = set()
        for nd in self.traverse():
            if nd.children:
                splits.add(nd.split_indices)
            else:
                if(with_leaves):
                    splits.add(nd.split_indices)
        self._cached_splits = splits
        return splits

    def is_internal(self) -> bool:
        return bool(self.children)

    def build_split_index(self):
        self._split_index = {}
        self._populate_split_index(self)

    def _populate_split_index(self, node):
        if node.split_indices is not None:
            self._split_index[node.split_indices] = node
        for ch in node.children:
            self._populate_split_index(ch)

    def find_node_by_split(self, target_split) -> Optional["Node"]:
        if hasattr(target_split, "indices"):
            target_split = tuple(
                target_split.indices
            )  # use indices attribute for conversion
        if self._split_index is None:
            self.build_split_index()
        return self._split_index.get(target_split)

    # ------------------------------------------------------------------------
    # append_child sets parent pointer (pointer-based approach)
    # ------------------------------------------------------------------------
    def append_child(self, node: "Node") -> None:
        node.parent = self
        self.children.append(node)
        self.invalidate_split_cache()

    # ------------------------------------------------------------------------
    # deep_copy (unchanged, except we skip copying .parent)
    # ------------------------------------------------------------------------
    def deep_copy(self):
        new_node = Node(
            name=self.name,
            length=self.length,
            values=self.values.copy(),
            split_indices=self.split_indices,
            leaf_name=self.leaf_name,
            _visual_order_indices=(
                self._visual_order_indices.copy()
                if self._visual_order_indices
                else None
            ),
            _order=self._order.copy(),
        )
        # Deep copy children and set their parent to the new_node
        new_node.children = [child.deep_copy() for child in self.children]
        for child in new_node.children:
            child.parent = new_node  # Set parent correctly
        # Copy cached splits and split index if present
        if self._cached_splits is not None:
            new_node._cached_splits = self._cached_splits.copy()
        if self._split_index is not None:
            new_node._split_index = self._split_index.copy()
        return new_node

    # ------------------------------------------------------------------------
    # split_indices initialization (unchanged)
    # ------------------------------------------------------------------------
    def _initialize_split_indices(self, order: List[str]) -> None:
        self.split_indices = ()
        for child in self.children:
            child._initialize_split_indices(order)
        if not self.children:
            self.split_indices = (order.index(self.name),)
            pass
        else:
            idxs = []
            for ch in self.children:
                idxs.extend(ch.split_indices)
            self.split_indices = SplitIndices(tuple(sorted(idxs)), tuple(order))

    # ------------------------------------------------------------------------
    # traversal, fix_child_order, to_hierarchy, etc. (unchanged)
    # ------------------------------------------------------------------------
    def traverse(self) -> Generator["Node", None, None]:
        yield self
        for ch in self.children:
            yield from ch.traverse()

    def _index(self, component: Tuple[str, ...]) -> SplitIndices:
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

    def to_weighted_splits(self) -> Dict[SplitIndices, float]:
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

    def to_indexed_split_set(self, with_leaves: bool = False) -> IndexedSplitSet:
        # Suppose self.to_splits() now returns a set of raw tuple splits.
        node_splits = self.to_splits(with_leaves=with_leaves)
        # Create a set of SplitIndices objects.
        split_indices_set = {SplitIndices(s, tuple(self._order)) for s in node_splits}
        return IndexedSplitSet(split_indices_set, order=tuple(self._order))

    def to_apted_format(self) -> List[Any]:
        label = f"{self.name or ''}|{self.split_indices}"
        if not self.children:
            return [label]
        return [label] + [c.to_apted_format() for c in self.children]

    # ------------------------------------------------------------------------
    # (MODIFIED) Now also reset _cached_current_order
    # ------------------------------------------------------------------------
    def invalidate_split_cache(self):
        self._cached_splits = None
        for c in self.children:
            c.invalidate_split_cache()

    def invalidate_current_order_cache(self):
        self._cached_current_order = None  # (NEW) Invalidate get_current_order cache
        for c in self.children:
            c.invalidate_current_order_cache()

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
            return f"{child_str}{self.name or ''}{meta}:{self.length}"
        else:
            return f"{self.name or ''}{meta}:{self.length}"

    def _to_list(self) -> Any:
        if self.children:
            return [c._to_list() for c in self.children]
        else:
            return self.name

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
        final_root._initialize_split_indices(final_root._order)
        final_root._fix_child_order()

        return final_root


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
