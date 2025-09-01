"""Phylogenetic movie API - Main interface for tree processing.

This module provides a clean interface to the modularized tree processing pipeline.
The original monolithic implementation has been broken down into separate modules:
- core.types: Type definitions
- core.tree_processor: Tree processing operations
- core.jumping_taxa_analyzer: Jumping taxa analysis
- core.tree_pipeline: Pipeline orchestration
"""

import logging

# Main API function - simplified interface
from typing import Optional

# Re-export main components for backward compatibility
from brancharchitect.movie_pipeline.types import (
    TreeList,
    TreePairSolution,
    InterpolationSequence,
    PipelineConfig,
)
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)


def process_trees(
    trees: TreeList,
    enable_rooting: bool = True,
    logger: Optional[logging.Logger] = None,
    optimization_iterations: int = 10,
    bidirectional_optimization: bool = False,
) -> InterpolationSequence:
    """
    Process trees through the complete pipeline.

    Args:
        trees: List of phylogenetic trees to process
        enable_rooting: Whether to apply midpoint rooting
        logger: Optional logger for tracking operations
        optimization_iterations: Number of iterations for tree order optimization
        bidirectional_optimization: Whether to use bidirectional optimization

    Returns:
        InterpolationSequence containing all pipeline outputs
    """
    config = PipelineConfig(
        enable_rooting=enable_rooting,
        optimization_iterations=optimization_iterations,
        bidirectional_optimization=bidirectional_optimization,
    )
    pipeline = TreeInterpolationPipeline(config=config, logger=logger)
    return pipeline.process_trees(trees)


__all__ = [
    "TreeList",
    "TreePairSolution",
    "InterpolationSequence",
    "PipelineConfig",
    "TreeInterpolationPipeline",
    "process_trees",
]
