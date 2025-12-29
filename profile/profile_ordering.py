#!/usr/bin/env python
"""Profile the tree order optimization algorithms."""

import cProfile
import pstats
import io
import sys
import os

sys.path.insert(0, os.getcwd())

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.leaforder.tree_order_optimiser import (
    RotationTreeOrderOptimizer,
    AnchorTreeOrderOptimizer,
)


# Newick strings for test cases
BIRD_SOURCE = "(Emu,(((((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)),Ostrich),(((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus))))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);"
BIRD_DEST = "(Emu,((Ostrich,((((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus)))),((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);"

HEIKO_SOURCE = "((O1,O2),(((A,B),C),((((F,G),E),D),H)));"
HEIKO_DEST = "((O1,O2),(((A,B),(C,(F,(E,G)))),(D,H)));"


def fresh_bird_trees():
    """Get fresh bird trees (single mover case) with proper encoding."""
    source = parse_newick(BIRD_SOURCE)
    dest = parse_newick(BIRD_DEST)
    dest.initialize_split_indices(source.taxa_encoding)
    return [source, dest]


def fresh_heiko_trees():
    """Get fresh heiko trees (multiple movers) with proper encoding."""
    source = parse_newick(HEIKO_SOURCE)
    dest = parse_newick(HEIKO_DEST)
    dest.initialize_split_indices(source.taxa_encoding)
    return [source, dest]


def run_anchor_optimization(tree_factory, iterations=10):
    """Run anchor-based optimization multiple times."""
    for _ in range(iterations):
        trees = tree_factory()
        optimizer = AnchorTreeOrderOptimizer(trees)
        optimizer.optimize(
            anchor_weight_policy="destination",
            circular=True,
        )


def run_rotation_optimization(tree_factory, iterations=10):
    """Run rotation-based optimization multiple times."""
    for _ in range(iterations):
        trees = tree_factory()
        optimizer = RotationTreeOrderOptimizer(trees)
        optimizer.optimize(n_iterations=3, bidirectional=False)


def profile_method(name, func, tree_factory, iterations=50):
    """Profile a specific optimization method."""
    sample_trees = tree_factory()
    print(f"\n{'=' * 60}")
    print(f"Profiling: {name} ({iterations} iterations)")
    print(f"Trees: {len(sample_trees)}, Leaves: {len(sample_trees[0].leaves)}")
    print(f"{'=' * 60}")

    profiler = cProfile.Profile()
    profiler.enable()

    func(tree_factory, iterations)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


def main():
    print("Tree Order Optimization Profiling")
    print("=" * 60)

    # Profile bird trees (single mover - light case)
    sample = fresh_bird_trees()
    print(f"\n--- Bird Trees (single mover, {len(sample[0].leaves)} leaves) ---")
    profile_method("Anchor-based (bird)", run_anchor_optimization, fresh_bird_trees, 50)
    profile_method(
        "Rotation-based (bird)", run_rotation_optimization, fresh_bird_trees, 50
    )

    # Profile heiko trees (multiple movers)
    sample = fresh_heiko_trees()
    print(f"\n--- Heiko Trees (multiple movers, {len(sample[0].leaves)} leaves) ---")
    profile_method(
        "Anchor-based (heiko)", run_anchor_optimization, fresh_heiko_trees, 50
    )
    profile_method(
        "Rotation-based (heiko)", run_rotation_optimization, fresh_heiko_trees, 50
    )


if __name__ == "__main__":
    main()
