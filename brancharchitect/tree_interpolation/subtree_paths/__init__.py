"""
Subtree path-based interpolation module.

This module contains all components related to subtree path-based tree interpolation,
including path planning, state management, execution, and ordering strategies.
"""

from .orchestrator import create_interpolation_for_active_split_sequence
from .planning import (
    build_edge_plan,
    InterpolationState,
    log_final_plans,
)
from .execution.step_executor import apply_stepwise_plan_for_edge
from .execution import (
    build_microsteps_for_selection,
    reorder_tree_toward_destination,
)
from .paths import calculate_subtree_paths
from .analysis import (
    get_unique_splits_for_active_changing_edge_subtree,
    find_incompatible_splits,
    get_shared_split_paths,
)

__all__ = [
    # Main interpolation functions
    "create_interpolation_for_active_split_sequence",
    "apply_stepwise_plan_for_edge",
    "build_microsteps_for_selection",
    # Path planning and state management
    "build_edge_plan",
    "InterpolationState",
    "calculate_subtree_paths",
    # Path segment utilities
    "get_unique_splits_for_active_changing_edge_subtree",
    "find_incompatible_splits",
    "get_shared_split_paths",
    # Partial ordering strategies
    "reorder_tree_toward_destination",
    # Utilities
    "log_final_plans",
]
