import argparse
import logging

from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.io import read_newick


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tree interpolation with focused logging",
    )
    parser.add_argument(
        "tree_path",
        nargs="?",
        default="current_testfiles/first_two_trees_350.newick",
        help="Path to Newick tree(s) file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging to capture planner and microstep diagnostics
    logging.basicConfig(level=logging.DEBUG)
    # Focused logging: enable DEBUG for planner/path extraction to trace missing splits
    logging.getLogger(
        "brancharchitect.tree_interpolation.subtree_paths.execution.step_executor"
    ).setLevel(logging.INFO)
    logging.getLogger(
        "brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry"
    ).setLevel(logging.DEBUG)
    logging.getLogger(
        "brancharchitect.tree_interpolation.subtree_paths.pivot_sequence_orchestrator"
    ).setLevel(logging.INFO)
    logging.getLogger(
        "brancharchitect.tree_interpolation.subtree_paths.execution.step_executor"
    ).setLevel(logging.INFO)
    logging.getLogger(
        "brancharchitect.tree_interpolation.subtree_paths.paths.transition_builder"
    ).setLevel(logging.DEBUG)

    # Load trees
    print(f"Loading trees from {args.tree_path}...")
    trees = read_newick(args.tree_path, treat_zero_as_epsilon=True)
    print(f"Loaded {len(trees) if hasattr(trees, '__len__') else '1'} tree(s)")

    # Pipeline config (mirrors run_pipeline defaults with anchor ordering)
    config = PipelineConfig(
        enable_rooting=False,
        bidirectional_optimization=False,
        logger_name="interpolation_run",
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="between_anchor_blocks",
    )

    pipeline = TreeInterpolationPipeline(config=config)
    result = pipeline.process_trees(trees=trees)

    print("--- Pipeline Summary ---")
    print(f"Original trees: {result['original_tree_count']}")
    print(f"Interpolated trees: {result['interpolated_tree_count']}")


if __name__ == "__main__":
    main()
