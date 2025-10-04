from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for tree interpolation pipeline."""

    enable_rooting: bool = False
    optimization_iterations: int = 10
    bidirectional_optimization: bool = False
    use_anchor_ordering: bool = True
    logger_name: str = "brancharchitect.movie_pipeline"
