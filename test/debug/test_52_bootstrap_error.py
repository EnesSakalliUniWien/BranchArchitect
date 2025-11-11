"""Test case for 52_bootstrap.newick error - Missing pivot edge in tree."""

from brancharchitect.movie_pipeline.types import (
    PipelineConfig,
    InterpolationResult,
)
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.io import read_newick


def test_52_bootstrap_missing_pivot_edge():
    """
    Test case that catches the error when a pivot edge is missing in one of the trees.

    This reproduces the error:
    ValueError: Active-changing edge (...) missing in source or destination tree.

    The error occurs when processing bootstrap trees where some pivot edges
    identified by the lattice algorithm don't exist in both trees of a pair.
    """
    print("=" * 80)
    print("TEST: 52_bootstrap.newick - Missing Pivot Edge Error")
    print("=" * 80)

    # Read the bootstrap trees
    trees = read_newick(
        "./52_bootstrap.newick",
        treat_zero_as_epsilon=True,
    )

    print(f"\nLoaded {len(trees)} trees from 52_bootstrap.newick")
    print(f"Number of taxa in first tree: {len(trees[0].taxa_encoding)}")

    # Configure the pipeline
    config = PipelineConfig(
        enable_rooting=False,
        bidirectional_optimization=False,
        logger_name="test_pipeline",
        use_anchor_ordering=False,
    )

    pipeline = TreeInterpolationPipeline(config=config)

    # With the consensus collapse fix, this should now succeed
    try:
        processed_data = pipeline.process_trees(trees=trees)
        print(f"\n✓ SUCCESS: Pipeline completed without errors!")
        print(f"   The consensus collapse fix is working correctly.")
        print(f"   Zero-length branches with destination splits are now preserved.")

    except ValueError as e:
        error_msg = str(e)
        if "missing in source or destination tree" in error_msg:
            print(f"\n✗ REGRESSION: {e}")
            print("\nThe fix didn't work - we're still losing destination splits!")
            raise
        else:
            print(f"\n✗ UNEXPECTED ERROR: {e}")
            raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR TYPE: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    test_52_bootstrap_missing_pivot_edge()
