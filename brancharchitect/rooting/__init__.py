"""
Rooting module for phylogenetic trees.

This module provides various rerooting algorithms organized into focused submodules:
- core_rooting: Basic rerooting operations and tree manipulation
- optimization_rooting: Advanced optimization and matching algorithms
- rooting: Unified interface importing from the above modules
"""

# Import main functions explicitly for better maintainability
from .rooting import *

# Make sure we expose the same interface
__all__ = [
    "find_best_matching_node",
    "simple_reroot",
    "reroot_at_node",
    "find_farthest_leaves",
    "path_between",
    "midpoint_root",
    "find_best_matching_node_jaccard",
    "reroot_by_jaccard_similarity",
    "build_global_correspondence_map",
    "find_optimal_root_candidates",
    "reroot_to_compared_tree",
]
