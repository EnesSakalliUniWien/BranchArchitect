#!/usr/bin/env python
"""
Comprehensive debug script to trace the full interpolation flow for trees 3-4.
This traces:
1. Lattice solver output (jumping taxa)
2. Anchor ordering (how taxa are positioned)
3. Microsteps execution (reordering step)
4. Why Ostrich appears to be the "first" moving subtree
"""

from brancharchitect.io import read_newick
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.tree_interpolation.subtree_paths.paths import (
    calculate_subtree_paths,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
)
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from collections import OrderedDict


def print_section(title):
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def main():
    # Load trees
    trees = read_newick("small_example copy 3.tree", treat_zero_as_epsilon=True)
    source = trees[3]
    dest = trees[4]

    print_section("1. LATTICE SOLVER OUTPUT")

    # Run lattice solver
    solver = LatticeSolver(source, dest)
    solutions, _ = solver.solve_iteratively()

    print(f"Found {len(solutions)} pivot edges with jumping taxa:")
    for pivot_edge, jumping_taxa_list in solutions.items():
        print(f"\nPivot edge: {sorted(pivot_edge.taxa)}")
        print(f"  Number of jumping taxa groups: {len(jumping_taxa_list)}")
        for i, jt in enumerate(jumping_taxa_list):
            print(f"  JT {i + 1}: {sorted(jt.taxa)}")

    print_section("2. SUBTREE PATHS CALCULATION")

    # Calculate subtree paths
    dest_paths, source_paths = calculate_subtree_paths(solutions, dest, source)

    print("Destination paths (expand paths):")
    for pivot_edge, subtree_paths in dest_paths.items():
        print(f"\n  Pivot: {sorted(pivot_edge.taxa)[:5]}...")
        for subtree, paths in subtree_paths.items():
            print(f"    Subtree {sorted(subtree.taxa)}: {len(paths)} paths")
            for p in list(paths)[:3]:
                print(f"      - {sorted(p.taxa)}")

    print("\nSource paths (collapse paths):")
    for pivot_edge, subtree_paths in source_paths.items():
        print(f"\n  Pivot: {sorted(pivot_edge.taxa)[:5]}...")
        for subtree, paths in subtree_paths.items():
            print(f"    Subtree {sorted(subtree.taxa)}: {len(paths)} paths")
            for p in list(paths)[:3]:
                print(f"      - {sorted(p.taxa)}")

    print_section("3. EDGE PLAN BUILDING (Selection Order)")

    # Get the pivot edge
    pivot_edge = list(solutions.keys())[0]
    jumping_taxa_list = solutions[pivot_edge]

    # Get paths for this pivot
    expand_paths = dest_paths.get(pivot_edge, {})
    collapse_paths = source_paths.get(pivot_edge, {})

    print(f"Building plan for pivot: {sorted(pivot_edge.taxa)[:5]}...")
    print(f"  Expand paths by subtree: {len(expand_paths)} subtrees")
    print(f"  Collapse paths by subtree: {len(collapse_paths)} subtrees")

    # Build the edge plan
    plans = build_edge_plan(
        expand_splits_by_subtree=expand_paths,
        collapse_splits_by_subtree=collapse_paths,
        collapse_tree=source,
        expand_tree=dest,
        current_pivot_edge=pivot_edge,
    )

    print(f"\nEdge plan has {len(plans)} selections (subtrees to process):")
    for i, (subtree, plan) in enumerate(plans.items()):
        print(f"\n  Selection {i + 1}: Subtree = {sorted(subtree.taxa)}")
        collapse_count = len(plan.get("collapse", {}).get("path_segment", []))
        expand_count = len(plan.get("expand", {}).get("path_segment", []))
        print(f"    Collapse paths: {collapse_count}")
        print(f"    Expand paths: {expand_count}")

    print_section("4. JUMPING TAXA ORDER ANALYSIS")

    print("Jumping taxa from lattice solver (in order):")
    for i, jt in enumerate(jumping_taxa_list):
        print(f"  {i + 1}. {sorted(jt.taxa)}")

    print("\nSelection order from edge plan:")
    for i, subtree in enumerate(plans.keys()):
        print(f"  {i + 1}. {sorted(subtree.taxa)}")

    # Check if Ostrich is first
    first_selection = list(plans.keys())[0] if plans else None
    if first_selection:
        print(f"\nFirst selection subtree: {sorted(first_selection.taxa)}")
        if "Ostrich" in first_selection.taxa:
            print("  -> Ostrich IS in the first selection!")
        else:
            print("  -> Ostrich is NOT in the first selection")

    print_section("5. ANCHOR ORDERING ANALYSIS")

    # Check how anchor_order.py processes this
    from brancharchitect.leaforder.anchor_order import (
        _get_solution_mappings,
        blocked_order_and_apply,
    )

    # Get solution mappings
    mappings_t1, mappings_t2 = _get_solution_mappings(
        source, dest, precomputed_solution=solutions
    )

    print("Solution mappings for source tree (t1):")
    for edge, mapping in mappings_t1.items():
        print(f"\n  Edge: {sorted(edge.taxa)[:5]}...")
        for sol, mapped in mapping.items():
            print(f"    Solution {sorted(sol.taxa)} -> Mapped {sorted(mapped.taxa)}")

    print("\nSolution mappings for destination tree (t2):")
    for edge, mapping in mappings_t2.items():
        print(f"\n  Edge: {sorted(edge.taxa)[:5]}...")
        for sol, mapped in mapping.items():
            print(f"    Solution {sorted(sol.taxa)} -> Mapped {sorted(mapped.taxa)}")

    print_section("6. TREE STRUCTURE COMPARISON")

    # Find Ostrich's position in tree structure
    def find_path_to_taxon(node, taxon_name, path=None):
        if path is None:
            path = []
        path = path + [node]
        if not node.children:
            if node.name == taxon_name:
                return path
            return None
        for child in node.children:
            result = find_path_to_taxon(child, taxon_name, path)
            if result:
                return result
        return None

    print("Path to Ostrich in SOURCE tree:")
    path = find_path_to_taxon(source, "Ostrich")
    if path:
        for i, node in enumerate(path):
            if node.children:
                print(f"  Level {i}: Internal ({len(node.children)} children)")
            else:
                print(f"  Level {i}: Leaf '{node.name}'")

    print("\nPath to Ostrich in DESTINATION tree:")
    path = find_path_to_taxon(dest, "Ostrich")
    if path:
        for i, node in enumerate(path):
            if node.children:
                print(f"  Level {i}: Internal ({len(node.children)} children)")
            else:
                print(f"  Level {i}: Leaf '{node.name}'")

    print_section("7. KEY INSIGHT: WHY OSTRICH APPEARS FIRST")

    print("""
The lattice solver identifies TWO jumping taxa groups under the same pivot edge:
1. Moa/Tinamou clade: {Dinornis, EasternMoa, lbmoa, ECtinamou, Gtinamou, Crypturellus}
2. Ostrich: {Ostrich}

The ORDER of these groups in the lattice solution determines the order of processing
in the microsteps. Looking at the edge plan selections:
""")

    for i, (subtree, plan) in enumerate(plans.items()):
        print(f"  Selection {i + 1}: {sorted(subtree.taxa)}")

    print("""
The reordering step in microsteps.py calls reorder_tree_toward_destination()
for EACH selection in order. However, the reorder_taxa() method in Node
can only reorder children within the tree's topology - it CANNOT move
a taxon across clade boundaries.

This is why Ostrich doesn't actually move to position 16 during the
reordering step - the tree topology constrains where Ostrich can be placed.

The actual movement of Ostrich happens through the TOPOLOGY CHANGE steps:
- Collapse (C): Removes splits that don't exist in destination
- Expand (IT_up): Creates splits that exist in destination

These topology changes are what actually move Ostrich from being grouped
with (GreatRhea, LesserRhea) to being grouped with the Moa/Tinamou clade.
""")


if __name__ == "__main__":
    main()
