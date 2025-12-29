#!/usr/bin/env python
import cProfile
import pstats
import io
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.getcwd())

from brancharchitect.io import read_newick
from brancharchitect.distances.distances import compute_tree_pair_component_paths, calculate_normalised_matrix
from brancharchitect.tree import Node
from brancharchitect.logger import jt_logger

def main():
    # Ensure logging is disabled for profiling
    jt_logger.disabled = True

    # Load trees
    newick_path = "notebooks/data/alltrees/alltrees.trees.newick"
    if not os.path.exists(newick_path):
        print(f"Error: {newick_path} not found.")
        return

    trees = read_newick(newick_path, force_list=True)
    trees = trees[:30]  # Limit to 30 trees for significant profiling
    print(f"Loaded {len(trees)} trees from {newick_path}")

    profiler = cProfile.Profile()
    profiler.enable()

    results = []
    # Run sequentially for profiling to capture all calls
    # i > j to avoid duplicate pairs and diagonal
    for i in range(len(trees)):
        for j in range(i):
            res = compute_tree_pair_component_paths(i, j, trees[i], trees[j])
            if res:
                results.append(res)

    norm_matrix = calculate_normalised_matrix(results, len(trees))

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(60)
    print(f"\n--- Profiling Results ({len(results)} pairs) ---")
    print(s.getvalue())

if __name__ == "__main__":
    main()
