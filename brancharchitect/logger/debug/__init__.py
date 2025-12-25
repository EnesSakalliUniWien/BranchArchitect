"""Debugging utilities for jumping taxa algorithms."""

from brancharchitect.logger import jt_logger, format_set
from brancharchitect.logger.debug.output import (
    generate_debug_html,
    write_debug_output,
    log_tree_splits,
)
from brancharchitect.logger.debug.error_handling import (
    log_stacktrace,
    log_detailed_error,
    debug_algorithm_execution,
)

__all__ = [
    "jt_logger",
    "format_set",
    "generate_debug_html",
    "write_debug_output",
    "log_tree_splits",
    "log_stacktrace",
    "log_detailed_error",
    "debug_algorithm_execution",
]
