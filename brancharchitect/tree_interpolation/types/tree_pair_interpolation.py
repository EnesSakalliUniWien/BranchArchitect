"""
Data types and classes for tree interpolation.

This module contains the data structures used throughout the tree interpolation
process, including result containers and intermediate data representations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node


@dataclass
class TreePairInterpolation:
    """Interpolation results for a single tree pair."""

    trees: List[Node]
    active_changing_split_tracking: Optional[List[Optional[Partition]]] = None
    jumping_subtree_solutions: Optional[Dict[Partition, List[List[Partition]]]] = None

    def __post_init__(self):
        """Initialize mutable fields to avoid sharing defaults between instances."""
        if self.active_changing_split_tracking is None:
            self.active_changing_split_tracking = []
        if self.jumping_subtree_solutions is None:
            self.jumping_subtree_solutions = {}
