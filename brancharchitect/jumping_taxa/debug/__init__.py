"""Debugging utilities for jumping taxa algorithms."""

from brancharchitect.core.combined_logger import Logger

# Create logger instance first - before ANY other imports
jt_logger = Logger("JumpingTaxa")
jt_logger.disabled = False


# Utility functions
def format_set(s: set) -> str:
    """Format set for display."""
    if not s:
        return "∅"
    return "{" + ", ".join(str(x) for x in sorted(s)) + "}"


# Export core functionality first
__all__ = ["jt_logger", "format_set"]

# Only import submodules AFTER core functionality is established
from .output import (  # noqa: E402
    generate_debug_html,
    write_debug_output,
    log_tree_splits,
)  # noqa: E402
from .error_handling import (  # noqa: E402
    log_stacktrace,
    log_detailed_error,
    debug_algorithm_execution,
)  # noqa: E402

# Update exports with submodule functionality
__all__ += [
    "generate_debug_html",
    "write_debug_output",
    "log_tree_splits",
    "log_stacktrace",
    "log_detailed_error",
    "debug_algorithm_execution",
    "log_arms_definition",
    "log_matrix_state",
    "log_meet_result",
    "log_classification_analysis",
    "log_tree_comparison",
    "log_process_direction",
    "log_bidirectional_analysis",
]
