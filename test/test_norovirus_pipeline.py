#!/usr/bin/env python3
"""
Test script for the full MSA -> Trees -> Interpolation pipeline.
Uses the norovirus dataset to verify tree generation and interpolation work correctly.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add msa_to_trees to path
sys.path.insert(0, str(Path(__file__).parent / "msa_to_trees"))

from msa_to_trees.pipeline import run_pipeline, FastTreeConfig
from brancharchitect.io import read_newick
from brancharchitect.movie_pipeline.tree_rooting import root_trees
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)


def test_tree_generation():
    """Test Step 1: Generate trees from MSA using sliding windows."""
    logger.info("=" * 60)
    logger.info("STEP 1: Tree Generation from MSA")
    logger.info("=" * 60)

    input_file = "datasets/norovirus_full_genome_490taxa_aligned.fasta"
    output_dir = "/tmp/norovirus_pipeline_test"

    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return None

    fasttree_config = FastTreeConfig(use_gtr=True, use_gamma=True, no_ml=True)

    result = run_pipeline(
        input_file=input_file,
        output_directory=output_dir,
        window_size=2500,
        step_size=1500,
        fasttree_config=fasttree_config,
        progress_callback=lambda msg: logger.info(f"[pipeline] {msg}"),
    )

    logger.info(f"Tree file: {result.tree_file_path}")
    logger.info(f"Total taxa: {result.total_taxa}")
    logger.info(f"Kept taxa: {result.kept_taxa}")
    logger.info(f"Dropped taxa: {len(result.dropped_taxa)}")
    logger.info(f"Num windows/trees: {result.num_windows}")

    if result.dropped_taxa:
        logger.warning(f"Dropped taxa: {result.dropped_taxa[:5]}...")

    return result


def test_tree_parsing(tree_file_path: Path, apply_midpoint_root: bool = True):
    """Test Step 2: Parse the generated trees and optionally apply midpoint rooting."""
    logger.info("=" * 60)
    logger.info("STEP 2: Parse Generated Trees")
    logger.info("=" * 60)

    try:
        trees = read_newick(str(tree_file_path), force_list=True)
        logger.info(f"Successfully parsed {len(trees)} trees")

        # Apply midpoint rooting if requested
        if apply_midpoint_root:
            logger.info("Applying midpoint rooting to all trees...")
            trees = root_trees(trees)
            logger.info(f"✓ Midpoint rooting applied to {len(trees)} trees")

        # Check taxa consistency
        taxa_sets = []
        for i, tree in enumerate(trees):
            taxa = set(tree.get_current_order())
            taxa_sets.append(taxa)
            logger.info(f"Tree {i}: {len(taxa)} taxa")

        # Verify all trees have same taxa
        first_taxa = taxa_sets[0]
        all_same = all(t == first_taxa for t in taxa_sets)

        if all_same:
            logger.info(
                f"✓ All trees have identical taxa sets ({len(first_taxa)} taxa)"
            )
        else:
            logger.error("✗ Trees have DIFFERENT taxa sets!")
            for i, taxa in enumerate(taxa_sets):
                diff = taxa.symmetric_difference(first_taxa)
                if diff:
                    logger.error(f"  Tree {i} differs by: {diff}")

        return trees

    except Exception as e:
        logger.error(f"Failed to parse trees: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_interpolation(trees):
    """Test Step 3: Run interpolation between consecutive trees."""
    logger.info("=" * 60)
    logger.info("STEP 3: Tree Interpolation")
    logger.info("=" * 60)

    if len(trees) < 2:
        logger.error("Need at least 2 trees for interpolation")
        return None

    try:
        # Test interpolation between first two trees
        logger.info(f"Testing interpolation between Tree 0 and Tree 1...")

        builder = SequentialInterpolationBuilder()

        # Try to build interpolation for first pair
        source = trees[0]
        dest = trees[1]

        logger.info(f"Source tree taxa: {len(source.get_current_order())}")
        logger.info(f"Dest tree taxa: {len(dest.get_current_order())}")

        # Check if taxa match
        source_taxa = set(source.get_current_order())
        dest_taxa = set(dest.get_current_order())

        if source_taxa != dest_taxa:
            logger.error(f"Taxa mismatch!")
            logger.error(f"  Only in source: {source_taxa - dest_taxa}")
            logger.error(f"  Only in dest: {dest_taxa - source_taxa}")
            return None

        logger.info("Taxa match ✓")

        # Try the full sequence interpolation
        logger.info(f"Building interpolation sequence for all {len(trees)} trees...")

        result = builder.build(trees)

        logger.info(f"✓ Interpolation successful!")
        logger.info(f"  Total interpolated trees: {len(result.interpolated_trees)}")

        return result

    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run the full pipeline test."""
    logger.info("=" * 60)
    logger.info("NOROVIRUS PIPELINE TEST")
    logger.info("=" * 60)

    # Step 1: Generate trees
    pipeline_result = test_tree_generation()
    if not pipeline_result:
        logger.error("Tree generation failed!")
        return 1

    # Step 2: Parse trees
    trees = test_tree_parsing(pipeline_result.tree_file_path)
    if not trees:
        logger.error("Tree parsing failed!")
        return 1

    # Step 3: Interpolation
    interp_result = test_interpolation(trees)
    if not interp_result:
        logger.error("Interpolation failed!")
        return 1

    logger.info("=" * 60)
    logger.info("ALL TESTS PASSED ✓")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
