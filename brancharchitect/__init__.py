# mypy: ignore-errors
"""Core BranchArchitect package."""

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

def __getattr__(name):
    if name in {
        "TreeList",
        "TreePairSolution",
        "InterpolationSequence",
        "PipelineConfig",
        "TreeInterpolationPipeline",
        "process_trees",
    }:
        from .movie_api import (
            TreeList,
            TreePairSolution,
            InterpolationSequence,
            PipelineConfig,
            TreeInterpolationPipeline,
            process_trees,
        )
        return locals()[name]
    if name in {"TreeMetadata", "DistanceMetrics"}:
        from .movie_pipeline.types import (  # type: ignore
            TreeMetadata,
            DistanceMetrics,
        )
        return locals()[name]
    raise AttributeError(name)
