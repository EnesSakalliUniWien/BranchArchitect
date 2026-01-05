#!/usr/bin/env python
"""Profile the full pipeline execution."""

import cProfile
import pstats
import io


def main():
    from brancharchitect.movie_pipeline.types import PipelineConfig
    from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
        TreeInterpolationPipeline,
    )
    from brancharchitect.io import read_newick
    from brancharchitect.logger import jt_logger

    # Disable logging for cleaner profiling
    jt_logger.disabled = True

    # Load trees
    trees = read_newick(
        "52_bootstrap.newick",
        treat_zero_as_epsilon=True,
    )

    print(f"Number of input trees: {len(trees)}")
    print(f"Leaves per tree: {len(trees[0].leaves)}")

    # Configure pipeline
    config = PipelineConfig(
        enable_rooting=False,
        bidirectional_optimization=False,
        logger_name="profiling",
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="between_anchor_blocks",
    )

    # Profile the pipeline execution
    profiler = cProfile.Profile()
    profiler.enable()

    pipeline = TreeInterpolationPipeline(config=config)
    result = pipeline.process_trees(trees=trees)

    profiler.disable()

    print(f"\nInterpolated trees: {len(result['interpolated_trees'])}")

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(60)
    print(s.getvalue())


if __name__ == "__main__":
    main()
