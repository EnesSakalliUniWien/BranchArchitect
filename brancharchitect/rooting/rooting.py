"""
Robust rerooting implementation for phylogenetic trees.

This module provides a unified interface to various rerooting strategies by
importing and exposing functions from specialized modules:
- Core rerooting operations (core_rooting.py)
- Advanced optimization algorithms (optimization_rooting.py)

This maintains backward compatibility while organizing code into focused modules.

Author: BranchArchitect Team
"""

# Import all public functions from specialized modules for backward compatibility
from .core_rooting import (
    # Helper functions - Note: _flip_upward is private and used internally
    # Core rerooting operations
    find_best_matching_node,
    simple_reroot,
    reroot_at_node,
    # Midpoint rooting
    find_farthest_leaves,
    path_between,
    midpoint_root,
)

from .jaccard_similarity import (
    # Node matching and correspondence
    find_best_matching_node_jaccard,
    # Jaccard similarity-based matching
    reroot_by_jaccard_similarity,
    # Enhanced global optimization
)

from .global_optimization import (
    # Global correspondence mapping
    build_global_correspondence_map,
    # Find optimal root candidates
)

from .optimization_rooting import (
    # Reroot to compared tree
    reroot_to_compared_tree,
)

from .root_selection import (
    # Find optimal root candidates
    find_optimal_root_candidates,
)

# Re-export for backward compatibility
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
