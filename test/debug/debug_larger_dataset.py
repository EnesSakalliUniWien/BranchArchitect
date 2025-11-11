"""Debug script for test_larger_dataset_all_splits_handled."""

from brancharchitect.parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
)
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from typing import Dict
import sys


def prepare_simple_subtree_paths(tree1, tree2, active_edge, jumping_subtrees):
    """
    Simplified version of prepare_subtree_paths for testing.
    Assigns splits to subtrees based on jumping solutions.
    """
    # Get splits within active edge scope
    node1 = tree1.find_node_by_split(active_edge)
    node2 = tree2.find_node_by_split(active_edge)

    if not node1 or not node2:
        return {
            "collapse_splits_by_subtree": {},
            "expand_splits_by_subtree": {},
        }

    splits1 = node1.to_splits()
    splits2 = node2.to_splits()

    to_collapse = splits1 - splits2
    to_expand = splits2 - splits1

    # Get jumping subtrees for this edge
    solutions = jumping_subtrees.get(active_edge, [])

    collapse_splits_by_subtree: Dict[Partition, PartitionSet[Partition]] = {}
    expand_splits_by_subtree: Dict[Partition, PartitionSet[Partition]] = {}

    # Assign splits to each subtree partition directly
    for subtree in solutions:
        # Assign collapse splits that overlap with this subtree
        if subtree not in collapse_splits_by_subtree:
            collapse_splits_by_subtree[subtree] = PartitionSet(
                encoding=tree1.taxa_encoding
            )

        for split in to_collapse:
            # If split overlaps with subtree, assign it
            if set(subtree.indices) & set(split.indices):
                collapse_splits_by_subtree[subtree].add(split)

        # Assign expand splits that overlap with this subtree
        if subtree not in expand_splits_by_subtree:
            expand_splits_by_subtree[subtree] = PartitionSet(
                encoding=tree1.taxa_encoding
            )

        for split in to_expand:
            # If split overlaps with subtree, assign it
            if set(subtree.indices) & set(split.indices):
                expand_splits_by_subtree[subtree].add(split)

    return {
        "collapse_splits_by_subtree": collapse_splits_by_subtree,
        "expand_splits_by_subtree": expand_splits_by_subtree,
    }


# Load test data
newicks = [
    "(Emu,(((((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)),Ostrich),(((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus))))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
    "(Emu,(((((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)),Ostrich),(((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus))))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
    "(Emu,((Ostrich,(((((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus)))),(((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone)))),(GreatRhea,LesserRhea))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
]

taxa_order = [
    "Emu",
    "BrushTurkey",
    "Chicken",
    "magpiegoose",
    "duck",
    "LBPenguin",
    "GaviaStellata",
    "oystercatcher",
    "turnstone",
    "GreatRhea",
    "LesserRhea",
    "Ostrich",
    "lbmoa",
    "EasternMoa",
    "Dinornis",
    "Alligator",
    "Caiman",
    "ECtinamou",
    "Gtinamou",
    "Crypturellus",
    "BrownKiwi",
    "LSKiwi",
    "gskiwi",
    "Cassowary",
]

trees = [parse_newick(n, order=taxa_order) for n in newicks]
encoding = trees[0].taxa_encoding

tree1, tree2 = trees[0], trees[2]

print("=== RUNNING LATTICE ALGORITHM ===")
jumping_subtrees, _ = iterate_lattice_algorithm(tree1, tree2, list(encoding.keys()))

if not jumping_subtrees:
    print("No jumping subtrees - trees identical")
    sys.exit(0)

active_edge = next(iter(jumping_subtrees.keys()))
print(f"Active edge: {active_edge}")
print()

print("=== PREPARE SUBTREE PATHS ===")
subtree_paths = prepare_simple_subtree_paths(
    tree1, tree2, active_edge, jumping_subtrees
)

print(f"Number of subtrees: {len(subtree_paths['collapse_splits_by_subtree'])}")
print()

# Collect all collapse splits from input
all_collapse = PartitionSet(encoding=encoding)
for subtree, splits in subtree_paths["collapse_splits_by_subtree"].items():
    print(f"Subtree {subtree}:")
    print(f"  Collapse splits: {len(splits)}")
    for split in splits:
        print(f"    {split}")
    all_collapse |= splits
print()

print(f"Total collapse splits in input: {len(all_collapse)}")
print()

print("=== BUILD EDGE PLAN ===")
plan = build_edge_plan(
    subtree_paths["expand_splits_by_subtree"],
    subtree_paths["collapse_splits_by_subtree"],
    tree1,
    tree2,
    active_edge,
)

print(f"Number of subtrees in plan: {len(plan)}")
print()

# Collect all collapse splits from plan
planned_collapse = PartitionSet(encoding=encoding)
for subtree, subtree_plan in plan.items():
    collapse_path = subtree_plan["collapse"]["path_segment"]
    print(f"Subtree {subtree}:")
    print(f"  Planned collapse splits: {len(collapse_path)}")
    for split in collapse_path:
        print(f"    {split}")
    planned_collapse |= PartitionSet(collapse_path, encoding=encoding)
print()

print(f"Total collapse splits in plan: {len(planned_collapse)}")
print()

print("=== COMPARISON ===")
missing = all_collapse - planned_collapse
extra = planned_collapse - all_collapse

if missing:
    print(f"MISSING {len(missing)} collapse splits in plan:")
    for split in missing:
        print(f"  {split}")
else:
    print("✅ All collapse splits accounted for")

if extra:
    print(f"\nEXTRA {len(extra)} collapse splits in plan (not in input):")
    for split in extra:
        print(f"  {split}")
