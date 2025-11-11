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
    current_pivot_edge_tracking: Optional[List[Optional[Partition]]] = None
    jumping_subtree_solutions: Optional[Dict[Partition, List[Partition]]] = None

    def __post_init__(self):
        """Initialize mutable fields to avoid sharing defaults between instances."""
        if self.current_pivot_edge_tracking is None:
            self.current_pivot_edge_tracking = []
        if self.jumping_subtree_solutions is None:
            self.jumping_subtree_solutions = {}

    @property
    def current_pivt_edge_trackeing(self) -> List[Optional[Partition]]:
        """Backward-compatible accessor for typoed name."""
        return self.current_pivot_edge_tracking or []

    @current_pivt_edge_trackeing.setter
    def current_pivt_edge_trackeing(
        self, value: List[Optional[Partition]]
    ) -> None:
        self.current_pivot_edge_tracking = value
