import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brancharchitect.io import read_newick
from brancharchitect.tree import Node
from brancharchitect.movie_pipeline.types import PipelineConfig, InterpolationResult
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)


def analyze_mappings():
    tree_file = "small_example copy 3.tree"
    print(f"Reading trees from {tree_file}...")
    try:
        trees = read_newick(tree_file, treat_zero_as_epsilon=True)
    except FileNotFoundError:
        print(f"File not found: {tree_file}")
        # Try full path if running from root relative
        print(f"Current CWD: {os.getcwd()}")
        return

    if isinstance(trees, Node):
        trees = [trees]

    print(f"Loaded {len(trees)} trees.")

    # We need at least 2 trees
    if len(trees) < 2:
        print("Not enough trees for interpolation (need at least 2)")
        return

    config = PipelineConfig(
        enable_rooting=False,
        bidirectional_optimization=False,
        logger_name="debug_pipeline",
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="between_anchor_blocks",
    )

    pipeline = TreeInterpolationPipeline(config=config)
    print("Running pipeline...")
    result: InterpolationResult = pipeline.process_trees(trees=trees)

    print("\n--- Listing Mappings ---")
    solutions = result.get("tree_pair_solutions", {})

    sorted_keys = sorted(
        solutions.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0
    )

    for key in sorted_keys:
        solution = solutions[key]
        print(f"\n# Pair: {key}")

        # Access mappings
        dest_map = solution.get("solution_to_destination_map")
        source_map = solution.get("solution_to_source_map")

        print("\n## Destination Mapping (solution_to_destination_map):")
        # Format printing
        if dest_map:
            for pivot_edge, elem_map in dest_map.items():
                print(f"  Pivot Edge: {pivot_edge}")
                for sol_elem, mapping in elem_map.items():
                    print(f"    Solution Element: {sol_elem} -> Maps to: {mapping}")
        else:
            print(
                "  (Empty) - No lattice processing or identical trees (or optimization skipped)"
            )

        print("\n## Source Mapping (solution_to_source_map):")
        if source_map:
            for pivot_edge, elem_map in source_map.items():
                print(f"  Pivot Edge: {pivot_edge}")
                for sol_elem, mapping in elem_map.items():
                    print(f"    Solution Element: {sol_elem} -> Maps to: {mapping}")
        else:
            print("  (Empty)")


if __name__ == "__main__":
    analyze_mappings()
