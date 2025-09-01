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
    names: List[str]
    mapping_one: Dict[Partition, Partition]
    mapping_two: Dict[Partition, Partition]
    s_edge_tracking: Optional[List[Optional[Partition]]] = None
    subtree_tracking: Optional[List[Optional[Partition]]] = None
    lattice_edge_solutions: Optional[Dict[Partition, List[List[Partition]]]] = None
    s_edge_distances: Optional[Dict[Partition, Dict[str, float]]] = None

    def __post_init__(self):
        """Initialize mutable fields to avoid sharing defaults between instances."""
        if self.s_edge_tracking is None:
            self.s_edge_tracking = []
        if self.subtree_tracking is None:
            self.subtree_tracking = []
        if self.lattice_edge_solutions is None:
            self.lattice_edge_solutions = {}
        if self.s_edge_distances is None:
            self.s_edge_distances = {}
