"""
Data types and classes for tree interpolation.

This module contains the data structures used throughout the tree interpolation
process, including result containers and intermediate data representations.
"""

from __future__ import annotations
from typing import List, Dict, TypedDict
from brancharchitect.elements.partition import Partition


class PairData(TypedDict):
    """A structured dictionary for per-pair interpolation data."""

    mapping_one: Dict[Partition, Partition]
    mapping_two: Dict[Partition, Partition]
    s_edge_length: int
    jumping_subtree_solutions: Dict[Partition, List[List[Partition]]]
