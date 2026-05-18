"""
Subtree path-based interpolation module.

This module contains all components related to subtree path-based tree interpolation,
including path planning, state management, execution, and ordering strategies.
"""

from .execution import (
    build_subtree_interpolation_frames,
    execute_active_split_transition_sequence,
    execute_pivot_edge_interpolation,
    reorder_tree_toward_destination,
)
from .planning import (
    build_pivot_subtree_transition_paths,
    build_pivot_transition_plan,
    PivotTransitionState,
    log_final_plans,
)
from .validation import assert_tree_topology_matches_destination
from .analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
    find_incompatible_splits,
)

__all__ = [
    # Main interpolation functions
    "execute_active_split_transition_sequence",
    "execute_pivot_edge_interpolation",
    "build_subtree_interpolation_frames",
    # Path planning and state management
    "build_pivot_transition_plan",
    "PivotTransitionState",
    "build_pivot_subtree_transition_paths",
    "assert_tree_topology_matches_destination",
    # Path segment utilities
    "get_unique_splits_for_current_pivot_edge_subtree",
    "find_incompatible_splits",
    # Partial ordering strategies
    "reorder_tree_toward_destination",
    # Utilities
    "log_final_plans",
]
