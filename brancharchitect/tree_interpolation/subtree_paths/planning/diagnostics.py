"""
Logging utilities for interpolation path plan generation.

This module provides logging functions for debugging and validating
tree interpolation plans.
"""

import logging
from typing import Dict, Any
from brancharchitect.elements.partition import Partition

logger = logging.getLogger(__name__)


def log_final_plans(plans: Dict[Partition, Dict[str, Any]]) -> None:
    """
    Log the final generated plans with clear, readable output.
    
    Shows both taxon names and indices for each subtree's collapse/expand paths.
    """
    logger.info("=== INTERPOLATION PLANS ===")
    
    for i, (subtree, plan) in enumerate(plans.items(), 1):
        # Get subtree info with both names and indices
        subtree_name = str(subtree)  # Uses Partition.__str__ for taxon names
        subtree_indices = tuple(int(idx) for idx in subtree)
        
        logger.info(f"Plan {i}. Subtree: {subtree_name}")
        logger.debug(f"  Subtree indices: {subtree_indices}")
        
        # Log collapse path
        collapse_segments = plan["collapse"]["path_segment"]
        if collapse_segments:
            logger.info("  Collapse path:")
            for seg in collapse_segments:
                seg_name = str(seg)
                seg_indices = tuple(int(idx) for idx in seg)
                logger.info(f"    - {seg_name}")
                logger.debug(f"      (indices: {seg_indices})")
        else:
            logger.info("  Collapse path: (none)")
        
        # Log expand path  
        expand_segments = plan["expand"]["path_segment"]
        if expand_segments:
            logger.info("  Expand path:")
            for seg in expand_segments:
                seg_name = str(seg)
                seg_indices = tuple(int(idx) for idx in seg)
                logger.info(f"    - {seg_name}")
                logger.debug(f"      (indices: {seg_indices})")
        else:
            logger.info("  Expand path: (none)")
    
    logger.info(f"=== Total plans: {len(plans)} ===")