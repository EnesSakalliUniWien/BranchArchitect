from brancharchitect.elements.partition_set import PartitionSet, Partition
from typing import List, Dict
from dataclasses import dataclass, field
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.types import TopToBottom


@dataclass
class PivotEdgeSubproblem:
    """
    Represents a pivot edge subproblem in the phylogenetic lattice structure.

    Each subproblem corresponds to a shared split (pivot) between two trees and contains
    the frontier information needed to compute reticulation events between them.
    """

    pivot_split: Partition

    tree1_node: Node

    tree2_node: Node

    # Per-side per-child frontiers (lists of PartitionSets of maximal shared elements)
    tree1_child_frontiers: Dict[Partition, TopToBottom]

    tree2_child_frontiers: Dict[Partition, TopToBottom]

    # Across-trees intersection in child subtrees
    child_subtree_splits_across_trees: PartitionSet[Partition]

    visits: int = field(default=0, init=False)

    encoding: dict[str, int]

    # Track excluded partitions (solutions that have been removed)
    # These are filtered out when building conflict matrices
    excluded_partitions: PartitionSet[Partition] = field(
        default_factory=lambda: PartitionSet()
    )

    # Deprecated single-side helpers removed: is_divergent/is_intermediate/is_collapsed
    # Edge type computation is inlined in `relationship` to avoid unused helpers.

    def remove_solutions_from_covers(self, solutions: List[PartitionSet[Partition]]):
        """
        Remove solved partitions from both covers (shared_top_splits) and
        bottom-to-frontier mappings to maintain data structure consistency.

        Also stores these partitions in excluded_partitions so they are filtered
        out in subsequent iterations without modifying the original trees.

        When a partition is removed from covers, it must also be removed from:
        1. The frontier sets (values of bottom_to_frontiers)
        2. The bottom_to_frontiers dictionary keys (if the partition is a bottom)
        3. Added to excluded_partitions for filtering in future iterations
        """
        for partitionset in solutions:
            for partition in partitionset:
                # Track this partition as excluded for future iterations
                self.excluded_partitions.add(partition)

                # Process tree1 child frontiers
                for top_to_bottom in self.tree1_child_frontiers.values():
                    # Remove from covers (tops)
                    if partition in top_to_bottom.shared_top_splits:
                        top_to_bottom.shared_top_splits.discard(partition)

                    # Remove from frontier sets (values)
                    for frontier_set in top_to_bottom.bottom_to_frontiers.values():
                        frontier_set.discard(partition)

                    # Remove from bottom keys if partition is a bottom
                    if partition in top_to_bottom.bottom_to_frontiers:
                        del top_to_bottom.bottom_to_frontiers[partition]

                # Process tree2 child frontiers
                for top_to_bottom in self.tree2_child_frontiers.values():
                    # Remove from covers (tops)
                    if partition in top_to_bottom.shared_top_splits:
                        top_to_bottom.shared_top_splits.discard(partition)

                    # Remove from frontier sets (values)
                    for frontier_set in top_to_bottom.bottom_to_frontiers.values():
                        frontier_set.discard(partition)

                    # Remove from bottom keys if partition is a bottom
                    if partition in top_to_bottom.bottom_to_frontiers:
                        del top_to_bottom.bottom_to_frontiers[partition]

    def has_remaining_conflicts(self) -> bool:
        """
        Check if this pivot subproblem still has any conflicts to resolve.

        Returns False if all conflicts have been resolved (via excluded_partitions),
        meaning we can skip this pivot in future iterations.
        """
        # Check if there are any non-excluded partitions in the frontier data structures
        for top_to_bottom in self.tree1_child_frontiers.values():
            if top_to_bottom.shared_top_splits:
                return True
            if top_to_bottom.bottom_to_frontiers:
                return True

        for top_to_bottom in self.tree2_child_frontiers.values():
            if top_to_bottom.shared_top_splits:
                return True
            if top_to_bottom.bottom_to_frontiers:
                return True

        return False

    @property
    def relationship(self) -> str:
        """
        Dynamically compute the relationship type based on both trees.

        Returns:
            A string describing the overall relationship: "divergent", "collapsed",
            "mixed", or "intermediate"
        """
        left_type = determine_covering_relation(
            get_child_splits(self.tree1_node),
            self.child_subtree_splits_across_trees,
            self.tree1_node,
        )
        right_type = determine_covering_relation(
            get_child_splits(self.tree2_node),
            self.child_subtree_splits_across_trees,
            self.tree2_node,
        )

        # If both sides have the same type, use that
        if left_type == right_type:
            return left_type

        # If one side is divergent and the other is collapsed, call it "mixed"
        if set([left_type, right_type]) == set(["divergent", "collapsed"]):
            return "mixed"

        # If one side is intermediate and the other is either divergent or collapsed
        return "intermediate"


def determine_covering_relation(
    child_splits: PartitionSet[Partition], common: PartitionSet[Partition], node: Node
) -> str:
    """
    Determine the covering relation between the descendant splits (D) of a node and the common splits (C).

    Let A = D ∩ C.

    In lattice-theoretic terms:
      - If A is empty, then the node's descendant splits do not overlap with the common split,
        meaning the node diverges completely from the common element ("divergent").
      - If child_splits ⊆ C, then all descendant splits are already included in the common split,
        so the node does not extend beyond the common element ("collapsed").
      - Otherwise, the node exhibits a partial extension relative to the common split ("intermediate").

    Returns:
        A string indicating the covering relation type: "divergent", "collapsed", or "intermediate".
    """
    A: PartitionSet[Partition] = child_splits & common
    if not A:
        return "divergent"
    if child_splits.issubset(common):
        return "collapsed"
    return "intermediate"


def get_child_splits(node: "Node") -> PartitionSet[Partition]:
    """
    Compute the set of child splits for a node n.

    Definition:
      If n has children { c₁, c₂, …, cₖ }, then
          D(n) = { s(c) : s(c) = c.split_indices for each c ∈ children(n) }.
    """
    return PartitionSet(
        {child.split_indices for child in node.children}, encoding=node.taxa_encoding
    )
