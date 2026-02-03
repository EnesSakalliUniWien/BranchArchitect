import sys
from pathlib import Path

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).parent))

from msa_to_trees.pipeline import run_pipeline as run_msa_pipeline
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.io import read_newick
from brancharchitect.tree import Node
from typing import List


def main(input_fasta):
    # 1. MSA -> Trees
    output_dir = Path("results/full_run")
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running MSA pipeline on {input_fasta}")
    try:
        tree_file = run_msa_pipeline(
            input_file=input_fasta,
            output_directory=str(output_dir),
            window_size=200,
            step_size=100,
        )
    except Exception as e:
        print(f"MSA Pipeline failed: {e}")
        return

    # 2. Trees -> Movie (Interpolation)
    print(f"Running Interpolation pipeline on {tree_file}")
    parsed_trees = read_newick(str(tree_file), treat_zero_as_epsilon=True)

    trees: List[Node] = (
        [parsed_trees] if isinstance(parsed_trees, Node) else parsed_trees
    )

    config = PipelineConfig(
        enable_rooting=False,
        bidirectional_optimization=False,  # Optimised for speed
        logger_name="full_pipeline",
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="between_anchor_blocks",
    )
    pipeline = TreeInterpolationPipeline(config=config)
    result = pipeline.process_trees(trees=trees)

    print(f"Interpolation complete.")
    print(f"Initial trees: {len(trees)}")
    print(f"Interpolated frames: {len(result['interpolated_trees'])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_pipeline.py <fasta_file>")
        sys.exit(1)
    main(sys.argv[1])
