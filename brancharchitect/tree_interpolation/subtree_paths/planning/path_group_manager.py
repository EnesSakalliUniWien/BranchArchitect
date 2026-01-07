"""
Path Group Manager for subtree ordering during tree interpolation.

This module provides path-based grouping and topological sorting of subtrees
based on their expand path relationships. Subtrees with overlapping or contained
expand paths are grouped together and processed in an order that respects
structural dependencies.

Key concepts:
- Path overlap: Two subtrees share at least one split in their expand paths
- Path containment: One subtree's expand path is a proper subset of another's
- Path group: Connected component of subtrees based on path overlap
- Topological ordering: Contained paths processed before containing paths
"""

from __future__ import annotations

import heapq
import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet

logger = logging.getLogger(__name__)


class PathGroupManager:
    """
    Manages path-based grouping and topological ordering of subtrees.

    This class analyzes expand path relationships between subtrees and provides
    an ordering that:
    1. Groups related subtrees together (based on path overlap)
    2. Processes smaller paths before larger containing paths (topological order)
    3. Always selects the smallest available path among ready candidates

    Example:
        manager = PathGroupManager(expand_splits_by_subtree, encoding)
        while True:
            next_subtree = manager.get_next_subtree(processed_set)
            if next_subtree is None:
                break
            # Process subtree...
            processed_set.add(next_subtree)
    """

    def __init__(
        self,
        expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]],
        encoding: Dict[str, int],
        enabled: bool = True,
    ) -> None:
        """
        Initialize path group manager.

        Args:
            expand_splits_by_subtree: Mapping of subtree -> expand path splits
            encoding: Taxa encoding dictionary
            enabled: Whether path-based grouping is enabled
        """
        self.encoding = encoding
        self.enabled = enabled
        self._expand_paths: Dict[Partition, PartitionSet[Partition]] = (
            expand_splits_by_subtree
        )

        # Computed on initialization
        self._overlap_graph: Dict[Partition, Set[Partition]] = {}
        self._containment_edges: Set[Tuple[Partition, Partition]] = set()
        self._groups: List[Set[Partition]] = []
        self._subtree_to_group: Dict[Partition, int] = {}

        # Topological sort state
        self._in_degree: Dict[Partition, int] = {}
        self._ready_queue: List[Tuple[int, int, Partition]] = []  # Min-heap

        # Track current group being processed
        self._current_group_index: int = 0
        self._processed_in_current_group: Set[Partition] = set()

        if enabled and expand_splits_by_subtree:
            self._compute_relationships()
            self._form_groups()
            self._initialize_topological_state()

    # ========================================================================
    # Relationship Detection
    # ========================================================================

    def _compute_relationships(self) -> None:
        """
        Compute pairwise overlap and containment relationships.

        For each pair of subtrees, determines:
        - Whether their expand paths overlap (share any splits)
        - Whether one path is contained in the other (proper subset)
        """
        subtrees = list(self._expand_paths.keys())

        # Initialize overlap graph for all subtrees
        for subtree in subtrees:
            self._overlap_graph[subtree] = set()

        # Check all pairs
        for i, subtree_a in enumerate(subtrees):
            path_a = self._expand_paths[subtree_a]
            path_a_set = set(path_a) if path_a else set()

            for subtree_b in subtrees[i + 1 :]:
                path_b = self._expand_paths[subtree_b]
                path_b_set = set(path_b) if path_b else set()

                # Check overlap (non-empty intersection)
                intersection = path_a_set & path_b_set
                if intersection:
                    self._overlap_graph[subtree_a].add(subtree_b)
                    self._overlap_graph[subtree_b].add(subtree_a)

                # Check containment (proper subset)
                # A is contained in B if A ⊂ B (proper subset)
                if path_a_set and path_b_set:
                    if path_a_set < path_b_set:  # A is proper subset of B
                        self._containment_edges.add((subtree_a, subtree_b))
                    elif path_b_set < path_a_set:  # B is proper subset of A
                        self._containment_edges.add((subtree_b, subtree_a))

    def has_overlap(self, subtree_a: Partition, subtree_b: Partition) -> bool:
        """
        Check if two subtrees have overlapping expand paths.

        Args:
            subtree_a: First subtree
            subtree_b: Second subtree

        Returns:
            True if paths share at least one split
        """
        return subtree_b in self._overlap_graph.get(subtree_a, set())

    def has_containment(self, contained: Partition, container: Partition) -> bool:
        """
        Check if contained's path is a proper subset of container's path.

        Args:
            contained: Subtree with potentially smaller path
            container: Subtree with potentially larger path

        Returns:
            True if contained's path is a proper subset of container's path
        """
        return (contained, container) in self._containment_edges

    def get_overlapping_subtrees(self, subtree: Partition) -> FrozenSet[Partition]:
        """
        Get all subtrees that have overlapping paths with the given subtree.

        Args:
            subtree: The subtree to query

        Returns:
            Frozen set of subtrees with overlapping paths
        """
        return frozenset(self._overlap_graph.get(subtree, set()))

    def get_containment_edges(self) -> FrozenSet[Tuple[Partition, Partition]]:
        """
        Get all containment relationships.

        Returns:
            Frozen set of (contained, container) tuples
        """
        return frozenset(self._containment_edges)

    # ========================================================================
    # Group Formation
    # ========================================================================

    def _form_groups(self) -> None:
        """
        Form connected components using union-find algorithm.

        Groups subtrees based on path overlap using transitive closure.
        If A overlaps B and B overlaps C, then A, B, C are in the same group.
        """
        if not self._expand_paths:
            return

        # Union-find with path compression
        parent: Dict[Partition, Partition] = {s: s for s in self._expand_paths}
        rank: Dict[Partition, int] = {s: 0 for s in self._expand_paths}

        def find(x: Partition) -> Partition:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x: Partition, y: Partition) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Union overlapping subtrees
        for subtree, neighbors in self._overlap_graph.items():
            for neighbor in neighbors:
                union(subtree, neighbor)

        # Collect groups by root
        groups_dict: Dict[Partition, Set[Partition]] = {}
        for subtree in self._expand_paths:
            root = find(subtree)
            if root not in groups_dict:
                groups_dict[root] = set()
            groups_dict[root].add(subtree)

        # Sort groups by minimum path size (smallest first)
        self._groups = sorted(
            groups_dict.values(),
            key=lambda g: (
                min(len(self._expand_paths.get(s, set())) for s in g),
                # Tie-breaker: lexicographic ordering of smallest subtree indices
                min(str(sorted(list(s.indices))) for s in g),
            ),
        )

        # Build subtree-to-group mapping
        for idx, group in enumerate(self._groups):
            for subtree in group:
                self._subtree_to_group[subtree] = idx

    def get_group(self, subtree: Partition) -> Optional[int]:
        """
        Get the group index for a subtree.

        Args:
            subtree: The subtree to query

        Returns:
            Group index, or None if subtree not tracked
        """
        return self._subtree_to_group.get(subtree)

    def get_group_members(self, group_index: int) -> FrozenSet[Partition]:
        """
        Get all subtrees in a group.

        Args:
            group_index: The group index

        Returns:
            Frozen set of subtrees in the group
        """
        if 0 <= group_index < len(self._groups):
            return frozenset(self._groups[group_index])
        return frozenset()

    def get_num_groups(self) -> int:
        """Get the number of path groups."""
        return len(self._groups)

    # ========================================================================
    # Topological Sorting
    # ========================================================================

    def _initialize_topological_state(self) -> None:
        """
        Initialize in-degree counts and ready queue for topological sort.

        Uses Kahn's algorithm with a priority queue to always select
        the smallest ready subtree.
        """
        if not self._groups:
            return

        # Check for cycles first
        cycle = self._detect_cycle()
        if cycle:
            logger.warning(
                "Cycle detected in containment graph: %s. "
                "Falling back to size-based ordering.",
                [str(sorted(list(s.indices))) for s in cycle],
            )
            # Clear containment edges to fall back to size-based ordering
            self._containment_edges = set()

        # Compute in-degree for all subtrees based on containment
        for subtree in self._expand_paths:
            self._in_degree[subtree] = 0

        for contained, container in self._containment_edges:
            # container depends on contained being processed first
            self._in_degree[container] += 1

        # Initialize ready queue for first group
        self._initialize_group_queue(0)

    def _initialize_group_queue(self, group_index: int) -> None:
        """
        Initialize ready queue for a specific group.

        Args:
            group_index: Index of the group to initialize
        """
        if group_index >= len(self._groups):
            return

        self._current_group_index = group_index
        self._processed_in_current_group = set()
        self._ready_queue = []

        group = self._groups[group_index]
        for subtree in group:
            if self._in_degree.get(subtree, 0) == 0:
                # Use (path_size, lexicographic_key, subtree) for heap ordering
                # Positive size in min-heap implies Shortest Path First
                path_size = len(self._expand_paths.get(subtree, set()))
                # Use bitmask for deterministic tie-breaking instead of string conversion
                tie_breaker = subtree.bitmask
                heapq.heappush(self._ready_queue, (path_size, tie_breaker, subtree))

    def _detect_cycle(self) -> Optional[List[Partition]]:
        """
        Detect cycles in containment graph using DFS.

        Returns:
            List of subtrees forming a cycle, or None if no cycle
        """
        if not self._containment_edges:
            return None

        # Build adjacency list for containment (contained -> containers)
        adj: Dict[Partition, List[Partition]] = {s: [] for s in self._expand_paths}
        for contained, container in self._containment_edges:
            adj[contained].append(container)

        visited: Set[Partition] = set()
        rec_stack: Set[Partition] = set()

        def dfs(node: Partition, path: List[Partition]) -> Optional[List[Partition]]:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj.get(node, []):
                if neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor) if neighbor in path else 0
                    return path[cycle_start:] + [neighbor]
                if neighbor not in visited:
                    result = dfs(neighbor, path + [neighbor])
                    if result:
                        return result

            rec_stack.remove(node)
            return None

        for subtree in self._expand_paths:
            if subtree not in visited:
                cycle = dfs(subtree, [subtree])
                if cycle:
                    return cycle

        return None

    # ========================================================================
    # Subtree Selection
    # ========================================================================

    def get_next_subtree(self, processed: Set[Partition]) -> Optional[Partition]:
        """
        Get next subtree to process using topological ordering.

        Selects the smallest ready subtree (zero in-degree) from the current
        group. When the current group is exhausted, moves to the next group.

        Args:
            processed: Set of already processed subtrees

        Returns:
            Next subtree to process, or None if all done
        """
        if not self.enabled or not self._groups:
            return None

        # Remove already-processed subtrees from ready queue
        while self._ready_queue:
            _, _, candidate = self._ready_queue[0]
            if candidate in processed:
                heapq.heappop(self._ready_queue)
            else:
                break

        # If current group exhausted, move to next
        while not self._ready_queue and self._current_group_index < len(self._groups):
            self._current_group_index += 1
            if self._current_group_index < len(self._groups):
                self._initialize_group_queue(self._current_group_index)
                # Remove already-processed from new queue
                while self._ready_queue:
                    _, _, candidate = self._ready_queue[0]
                    if candidate in processed:
                        heapq.heappop(self._ready_queue)
                    else:
                        break

        if not self._ready_queue:
            return None

        # Pop smallest ready subtree
        _, _, subtree = heapq.heappop(self._ready_queue)

        # Update in-degrees for dependents (containers that depend on this subtree)
        for contained, container in self._containment_edges:
            if contained == subtree:
                self._in_degree[container] -= 1
                if self._in_degree[container] == 0:
                    # Container is now ready - add to queue if in current group
                    if (
                        self._subtree_to_group.get(container)
                        == self._current_group_index
                    ):
                        if container not in processed:
                            path_size = len(self._expand_paths.get(container, set()))
                            # Use bitmask for deterministic tie-breaking
                            tie_breaker = container.bitmask
                            heapq.heappush(
                                self._ready_queue, (path_size, tie_breaker, container)
                            )

        return subtree

    def get_path_size(self, subtree: Partition) -> int:
        """
        Get the expand path size for a subtree.

        Args:
            subtree: The subtree to query

        Returns:
            Number of splits in the expand path
        """
        return len(self._expand_paths.get(subtree, set()))

    # ========================================================================
    # Debug and Inspection
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PathGroupManager("
            f"subtrees={len(self._expand_paths)}, "
            f"groups={len(self._groups)}, "
            f"containment_edges={len(self._containment_edges)}, "
            f"enabled={self.enabled})"
        )

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about path relationships.

        Returns:
            Dictionary with counts of subtrees, groups, overlaps, containments
        """
        total_overlaps = (
            sum(len(neighbors) for neighbors in self._overlap_graph.values()) // 2
        )

        return {
            "total_subtrees": len(self._expand_paths),
            "total_groups": len(self._groups),
            "total_overlaps": total_overlaps,
            "total_containments": len(self._containment_edges),
            "singleton_groups": sum(1 for g in self._groups if len(g) == 1),
        }
