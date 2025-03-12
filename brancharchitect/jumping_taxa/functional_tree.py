# functional_tree.py
import sys
from brancharchitect.tree import Node
from brancharchitect.split import PartitionSet
from typing import Any, Dict, Optional, List
from dataclasses import dataclass


# Replace the reconfigure line with this version-compatible code:
try:
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+ approach
except AttributeError:
    # For older Python versions
    import os

    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)


@dataclass(frozen=True)
class HasseEdge:

    split: Any

    left_node: Node
    right_node: Node

    left_cover: List[PartitionSet]
    right_cover: List[PartitionSet]

    child_meet: PartitionSet

    def is_divergent(self, tree: int = 1) -> bool:
        """
        Returns True if the covering relation for the specified side is "divergent",
        meaning no child split overlaps with the common split.
        """
        if tree == 1:
            return (
                determine_covering_relation(
                    get_child_splits(self.left_node), self.child_meet, self.left_node
                )
                == "divergent"
            )
        else:
            return (
                determine_covering_relation(
                    get_child_splits(self.right_node), self.child_meet, self.right_node
                )
                == "divergent"
            )

    def is_intermediate(self, tree: int = 1) -> bool:
        """
        Returns True if the covering relation for the specified side is "intermediate",
        meaning some but not all child splits overlap with the common split.
        """
        if tree == 1:
            return (
                determine_covering_relation(
                    get_child_splits(self.left_node), self.child_meet, self.left_node
                )
                == "intermediate"
            )
        else:
            return (
                determine_covering_relation(
                    get_child_splits(self.right_node), self.child_meet, self.right_node
                )
                == "intermediate"
            )

    def is_collapsed(self, tree: int = 1) -> bool:
        """
        Returns True if the covering relation for the specified side is "collapsed",
        meaning every child split is common.
        """
        if tree == 1:
            return (
                determine_covering_relation(
                    get_child_splits(self.left_node), self.child_meet, self.left_node
                )
                == "collapsed"
            )
        else:
            return (
                determine_covering_relation(
                    get_child_splits(self.right_node), self.child_meet, self.right_node
                )
                == "collapsed"
            )

    def get_edge_types(self) -> tuple:
        """
        Returns the covering relation type for the specified side.
        """
        return tuple(
            (
                determine_covering_relation(
                    get_child_splits(self.left_node), self.child_meet, self.left_node
                ),
                determine_covering_relation(
                    get_child_splits(self.right_node), self.child_meet, self.right_node
                ),
            ),
        )


def determine_covering_relation(
    child_splits: PartitionSet, common: PartitionSet, node: Node
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
    A = child_splits & common
    if not A:
        return "divergent"
    if child_splits.issubset(common):
        return "collapsed"
    return "intermediate"


def get_child_splits(node: "Node") -> PartitionSet:
    """
    Compute the set of child splits for a node n.

    Definition:
      If n has children { c₁, c₂, …, cₖ }, then
          D(n) = { s(c) : s(c) = c.split_indices for each c ∈ children(n) }.
    """
    return PartitionSet({child.split_indices for child in node.children})


def compare_tree_splits(tree1: "Node", tree2: "Node") -> Optional[Dict[str, HasseEdge]]:
    """Compute detailed split information for two trees."""
    # Ensure both trees have their indices built

    # Get splits with validation
    S1 = tree1.to_splits()
    S2 = tree2.to_splits()

    # Get common splits and verify they exist in both trees
    U = S1 | S2
    s_edges = {}

    for s in U:
        if len(s) == 1:
            continue
        if s in S1 and s in S2:

            n_left = tree1.find_node_by_split(s)
            n_right = tree2.find_node_by_split(s)

            node_meet = n_left.to_splits(with_leaves=True) & n_right.to_splits(
                with_leaves=True
            )

            D1 = get_child_splits(n_left)
            D2 = get_child_splits(n_right)

            direct_child_meet = D1 & D2

            # Process further if at least one node has a nontrivial mixture of child splits.
            if D1 != direct_child_meet != D2:

                left_covers = compute_cover_elements(
                    n_left, D1, PartitionSet(node_meet)
                )

                right_covers = compute_cover_elements(
                    n_right, D2, PartitionSet(node_meet - {s})
                )

                s_edges[s] = HasseEdge(
                    split=s,
                    left_cover=left_covers,
                    right_cover=right_covers,
                    child_meet=direct_child_meet,
                    left_node=n_left,
                    right_node=n_right,
                )
    return s_edges


def compute_cover_elements(
    node: Node, child_splits: PartitionSet, common_excluding: PartitionSet
) -> List[PartitionSet]:
    """
    For each child split of a node, compute its covering element.

    Cover(child) = (child.to_splits(with_leaves=True) ∩ common_excluding).cover()

    Returns a PartitionSet of cover elements.
    """
    cover_list: List[PartitionSet] = []
    for split in child_splits:
        child_node = node.find_node_by_split(split)
        if child_node is not None:
            node_splits = child_node.to_splits(with_leaves=True)
            cover_candidate = node_splits & common_excluding
            covering_element = cover_candidate.cover()
            cover_list.append(covering_element)
        else:
            raise ValueError(f"Split {split} not found in tree {node}")
    # Add safety check for arms
    if not cover_list:
        raise ValueError(f"Arms not found for split {node.split_indices} ")
    return cover_list
