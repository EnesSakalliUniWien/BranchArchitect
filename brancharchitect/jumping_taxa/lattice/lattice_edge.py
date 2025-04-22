from brancharchitect.partition_set import PartitionSet, Partition
from typing import List
from dataclasses import dataclass, field
from brancharchitect.tree import Node


@dataclass
class LatticeEdge:

    split: Partition

    left_node: Node
    
    right_node: Node

    left_cover: List[PartitionSet]
    
    right_cover: List[PartitionSet]
    
    left_unique_atoms: List[PartitionSet]
    
    right_unique_atoms: List[PartitionSet]

    left_unique_covet: List[PartitionSet]
    
    right_unique_covet: List[PartitionSet]


    child_meet: PartitionSet
    
    visits: int = field(default=0, init=False)  # added visits counter
    
    look_up: dict
    
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

    def get_edge_types(self) -> tuple[str, str]:
        """
        Returns the covering relation types for left and right nodes as a tuple.
        """
        left_type = determine_covering_relation(
            get_child_splits(self.left_node), 
            self.child_meet, 
            self.left_node
        )
        right_type = determine_covering_relation(
            get_child_splits(self.right_node), 
            self.child_meet, 
            self.right_node
        )
        return (left_type, right_type)

    
    def remove_solutions_from_covers(self, solutions: List[PartitionSet]):
        
         for partitionset in solutions:
             for partition in partitionset:
                 for i in range(len(self.left_cover)):
                     if partition in self.left_cover[i]:
                         self.left_cover[i].remove(partition)
                 for i in range(len(self.right_cover)):
                     if partition in self.right_cover[i]:
                         self.right_cover[i].remove(partition)


    @property
    def relationship(self) -> str:
        """
        Dynamically compute the relationship type based on both trees.
        
        Returns:
            A string describing the overall relationship: "divergent", "collapsed", 
            "mixed", or "intermediate"
        """
        left_type, right_type = self.get_edge_types()
        
        # If both sides have the same type, use that
        if left_type == right_type:
            return left_type
        
        # If one side is divergent and the other is collapsed, call it "mixed"
        if set([left_type, right_type]) == set(["divergent", "collapsed"]):
            return "mixed"
        
        # If one side is intermediate and the other is either divergent or collapsed
        return "intermediate"


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
    return PartitionSet({child.split_indices for child in node.children}, look_up=node._encoding)
