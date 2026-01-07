from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for tree interpolation pipeline."""

    enable_rooting: bool = False
    optimization_iterations: int = 10
    bidirectional_optimization: bool = False
    use_anchor_ordering: bool = True
    # Controls how stable anchors are positioned across trees when using
    # anchor-based ordering. 'destination' keeps anchors aligned, so only
    # jumping taxa contribute to large positional changes.
    anchor_weight_policy: str = "destination"
    # Circular rendering support: rotate final permutations to place the
    # linear boundary at a visually appropriate position for circular layouts.
    circular: bool = False
    circular_boundary_policy: str = (
        "between_anchor_blocks"  # or "largest_mover_at_zero"
    )
    logger_name: str = "brancharchitect.movie_pipeline"
    enable_debug_visualization: bool = False
