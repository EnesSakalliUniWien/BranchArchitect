#!/usr/bin/env python
"""Profile the lattice solver by running a test case multiple times."""
import cProfile
import pstats
import io
import sys

sys.path.insert(0, "test/colouring")

def run_test():
    """Run a representative test case."""
    from utils import build_trees_from_data
    import json

    test_file = "test/colouring/trees/heiko_5_test_tree/heiko_5_test_tree.json"
    with open(test_file) as f:
        data = json.load(f)
    data["name"] = "profiling"

    t1, t2 = build_trees_from_data(data)

    from brancharchitect.jumping_taxa.lattice.compute_pivot_solutions_with_deletions import (
        compute_pivot_solutions_with_deletions,
    )
    compute_pivot_solutions_with_deletions(t1, t2)

def main():
    # Run 50 iterations for profiling
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(50):
        run_test()

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(40)
    print(s.getvalue())

if __name__ == "__main__":
    main()
