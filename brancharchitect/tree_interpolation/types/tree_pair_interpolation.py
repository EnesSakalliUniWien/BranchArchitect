"""
Data types and classes for tree interpolation.

This module contains the data structures used throughout the tree interpolation
process, including result containers and intermediate data representations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from .interpolation_movement import SprMoveEvent


@dataclass
class TreePairInterpolation:
    """Interpolation results for a single tree pair."""

    trees: List[Node]
    active_pivot_edges: List[Optional[Partition]] = field(default_factory=list)
    current_subtree_highlights: List[Optional[List[Partition]]] = field(
        default_factory=list
    )
    spr_move_events: List[SprMoveEvent] = field(default_factory=list)
    jumping_subtree_solutions: Dict[Partition, List[Partition]] = field(
        default_factory=dict
    )
