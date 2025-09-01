"""Core type definitions for phylogenetic analysis."""

from typing import List
from dataclasses import dataclass


@dataclass
class DistanceMetrics:
    """Distance metrics for phylogenetic tree analysis."""

    rfd_list: List[float]
    """Robinson-Foulds distances between consecutive trees."""

    wrfd_list: List[float]
    """Weighted Robinson-Foulds distances between consecutive trees."""
