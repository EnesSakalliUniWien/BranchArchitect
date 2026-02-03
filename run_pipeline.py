"""Tree processing pipeline."""

from typing import List
from brancharchitect.movie_pipeline.types import (
    PipelineConfig,
    InterpolationResult,
)
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.io import read_newick
from brancharchitect.tree import Node


def main():
    # Define variables for testing
    enable_rooting = False
    trees: Node | List[Node] = read_newick(
        "./test/data/current_testfiles/small_example.newick",
        treat_zero_as_epsilon=True,
    )

    # Process trees through the pipeline
    config = PipelineConfig(
        enable_rooting=enable_rooting,
        bidirectional_optimization=False,
        logger_name="webapp_pipeline",
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="between_anchor_blocks",
    )
    pipeline = TreeInterpolationPipeline(config=config)
    processed_data: InterpolationResult = pipeline.process_trees(trees=trees)

    print(f"Number of input trees: {len(trees)}")


if __name__ == "__main__":
    main()
