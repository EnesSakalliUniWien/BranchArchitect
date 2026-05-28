"""
MSA to Movie Pipeline

Full pipeline: Multiple Sequence Alignment → Sliding Window Trees → Interpolation

Usage:
    python examples/msa_to_movie.py alignment.fasta
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from msa_to_trees import run_pipeline as run_msa_pipeline
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.io import read_newick
from brancharchitect.tree import Node


def main(input_fasta: str) -> None:
    """Run the full MSA → Trees → Interpolation pipeline."""

    # 1. MSA → Trees (sliding window + FastTree)
    output_dir = Path("results/full_run")
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Generating trees from MSA: {input_fasta}")
    try:
        pipeline_result = run_msa_pipeline(
            input_file=input_fasta,
            output_directory=str(output_dir),
            window_size=200,
            step_size=100,
        )
        tree_file = pipeline_result.tree_file_path
    except Exception as e:
        print(f"MSA Pipeline failed: {e}")
        return

    parsed_trees = read_newick(
        str(tree_file), force_list=True, treat_zero_as_epsilon=True
    )
    if not isinstance(parsed_trees, list):
        print("MSA Pipeline produced an unexpected tree format.")
        return
    print(f"      → {len(parsed_trees)} trees written to {tree_file}")

    # 2. Trees → Interpolation
    print("[2/2] Interpolating tree sequence...")
    trees: List[Node] = parsed_trees

    config = PipelineConfig(
        enable_rooting=False,
        use_anchor_ordering=True,
        circular=True,
    )
    pipeline = TreeInterpolationPipeline(config=config)
    interpolation_result = pipeline.process_trees(trees=trees)

    print("\nResults:")
    print(f"  Input trees: {len(trees)}")
    print(f"  Interpolated frames: {len(interpolation_result['interpolated_trees'])}")

    # Print first and last frame as Newick
    print(
        f"\nFirst frame: "
        f"{interpolation_result['interpolated_trees'][0].to_newick()[:80]}..."
    )
    print(
        f"Last frame:  "
        f"{interpolation_result['interpolated_trees'][-1].to_newick()[:80]}..."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_pipeline.py <fasta_file>")
        sys.exit(1)
    main(sys.argv[1])
