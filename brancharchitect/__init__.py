# mypy: ignore-errors
"""Core BranchArchitect package."""

# Export movie API for external use
from .movie_api import (
    TreeList,
    TreePairSolution,
    InterpolationSequence,
    PipelineConfig,
    TreeInterpolationPipeline,
    process_trees,
)

# Export additional types for phylo-movies integration
from .movie_pipeline.types import (
    TreeMetadata,
    DistanceMetrics,
)

__all__ = [
    "TreeList",
    "TreePairSolution",
    "InterpolationSequence",
    "PipelineConfig",
    "TreeInterpolationPipeline",
    "process_trees",
    "TreeMetadata",
    "DistanceMetrics",
]
