"""
Logging utilities for interpolation path plan generation.

This module provides modular logging functions for debugging and validating
tree interpolation plans.
"""

import logging
from typing import Dict, List, Any, Set, Optional
from brancharchitect.elements.partition import Partition

logger = logging.getLogger(__name__)


def log_initial_analysis(
    highest_occurrence: Optional[Partition],
    collapse_shared: Dict[Partition, Dict[str, Any]],
    least_active: Optional[Partition], 
    expand_shared: Dict[Partition, Dict[str, Any]]
) -> None:
    """
    Log initial analysis of shared paths.
    
    Args:
        highest_occurrence: Subtree with most shared collapse paths
        collapse_shared: Shared collapse paths between subtrees
        least_active: Subtree with least shared expand paths
        expand_shared: Shared expand paths between subtrees
    """
    logger.debug(
        "Highest occurrence in collapse shared paths: %s",
        highest_occurrence
    )
    logger.debug("Collapse paths shared between subtrees: %s", collapse_shared)
    logger.debug("Least active expand subtree: %s", least_active)
    logger.debug("Expand paths shared between subtrees: %s", expand_shared)


def log_final_plans(plans: Dict[Partition, Dict[str, Any]]) -> None:
    """
    Log the final generated plans.
    
    Args:
        plans: Dictionary mapping subtrees to their collapse/expand plans
    """
    logger.debug("=== FINAL PLANS ===")
    for subtree, plan in plans.items():
        logger.debug("Subtree: %s", subtree)
        logger.debug("  Collapse path: %s", plan['collapse']['path_segment'])
        logger.debug("  Expand path: %s", plan['expand']['path_segment'])
    logger.debug("===================")


def validate_and_log_plan_segments(
    plans: Dict[Partition, Dict[str, Any]],
    active_edge: Partition,
    to_collapse: Set[Partition],
    to_expand: Set[Partition]
) -> None:
    """
    Validate and log plan segments against expected splits.
    
    Args:
        plans: Dictionary mapping subtrees to their collapse/expand plans
        active_edge: The active changing edge being processed
        to_collapse: Set of splits that should be collapsed
        to_expand: Set of splits that should be expanded
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
        
    logger.debug("=== DEBUGGING PLANS ===")
    logger.debug("Active changing edge: %s", active_edge)
    logger.debug("To be collapsed splits: %s", to_collapse)
    logger.debug("To be expanded splits: %s", to_expand)

    for subtree, plan in plans.items():
        _validate_subtree_segments(
            subtree,
            plan["collapse"]["path_segment"],
            plan["expand"]["path_segment"],
            to_collapse,
            to_expand
        )
    logger.debug("=======================")


def log_no_more_subtrees() -> None:
    """Log when there are no more active collapse subtrees with shared splits."""
    logger.debug("No more active collapse subtrees with shared splits.")


def _validate_subtree_segments(
    subtree: Partition,
    collapse_segments: List[Partition],
    expand_segments: List[Partition],
    valid_collapse: Set[Partition],
    valid_expand: Set[Partition]
) -> None:
    """
    Validate segments for a single subtree.
    
    Args:
        subtree: The subtree being validated
        collapse_segments: List of collapse segments for this subtree
        expand_segments: List of expand segments for this subtree
        valid_collapse: Set of valid collapse splits
        valid_expand: Set of valid expand splits
    """
    logger.debug("Subtree %s:", subtree)
    
    logger.debug("  Collapse segments that should exist in current tree:")
    for seg in collapse_segments:
        if seg in valid_collapse:
            logger.debug("    ✓ %s", seg)
        else:
            logger.warning("    ✗ %s (NOT IN TREE!)", seg)

    logger.debug("  Expand segments that should be created:")
    for seg in expand_segments:
        if seg in valid_expand:
            logger.debug("    ✓ %s", seg)
        else:
            logger.warning("    ✗ %s (NOT A VALID TARGET!)", seg)

