"""Movie pipeline package for phylogenetic tree processing."""

from .tree_interpolation_pipeline import TreeInterpolationPipeline
from .types import (
    TreeList,
    TreePairSolution,
    TreeMetadata,
    InterpolationSequence,
    PipelineConfig,
    DistanceMetrics,
    create_empty_interpolation_sequence,
    create_single_tree_interpolation_sequence,
)

__all__ = [
    "TreeInterpolationPipeline",
    "TreeList",
    "TreePairSolution",
    "TreeMetadata",
    "InterpolationSequence",
    "PipelineConfig",
    "DistanceMetrics",
    "create_empty_interpolation_sequence",
    "create_single_tree_interpolation_sequence",
]