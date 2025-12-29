"""
Comprehensive test for tree interpolation split handling.

This test verifies that:
1. ALL collapsed splits are properly deleted from the tree
2. ALL expanded splits are properly applied to the tree
3. Errors are thrown when splits cannot be applied
4. The compute_pivot_solutions_with_deletions is used correctly
"""

import unittest
from typing import Dict
from brancharchitect.parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.topology_ops.weights import (
    calculate_intermediate_tree,
)
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    collapse_zero_length_branches_for_node,
)
from brancharchitect.tree_interpolation.topology_ops.expand import (
    apply_split_simple,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
)


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

    # For each subtree partition, assign splits directly (flat list of partitions)
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


class TestCompleteSplitHandling(unittest.TestCase):
    """Test complete split handling in tree interpolation."""

    def setUp(self):
        """Set up test trees from small_example.newick."""
        # Trees from small_example.newick
        newick1 = "((O1,O2),(((((A,A1),A2),(B,B1)),C),((D,(E,(((F,G),I),M))),H)));"
        newick2 = "((O1,O2),(((((A,A1),B1),(B,A2)),(C,(E,(((F,M),I),G)))),(D,H)));"
        newick3 = "((O1,O2),(((((A,A1),B1),(B,A2)),(C,(E,(((F,M),I),G)))),(D,H)));"

        self.taxa_order = [
            "O1",
            "O2",
            "A",
            "A1",
            "A2",
            "B",
            "B1",
            "C",
            "D",
            "E",
            "F",
            "G",
            "I",
            "M",
            "H",
        ]

        self.tree1 = parse_newick(newick1, order=self.taxa_order)
        self.tree2 = parse_newick(newick2, order=self.taxa_order)
        self.tree3 = parse_newick(newick3, order=self.taxa_order)

        self.encoding = self.tree1.taxa_encoding

    def test_all_collapsed_splits_deleted(self):
        """Verify ALL collapsed splits are deleted from intermediate tree."""
        # Get splits for both trees
        splits1 = self.tree1.to_splits()
        splits2 = self.tree2.to_splits()

        # Splits to be collapsed = in tree1 but not in tree2
        to_collapse = splits1 - splits2

        # Create intermediate tree by setting collapsed splits to 0
        split_dict = {split: 1.0 for split in splits2}
        intermediate = calculate_intermediate_tree(self.tree1, split_dict)

        # Collapse zero-length branches
        collapse_zero_length_branches_for_node(intermediate)

        # Verify: ALL collapsed splits should be gone
        intermediate_splits = intermediate.to_splits()

        for collapsed_split in to_collapse:
            self.assertNotIn(
                collapsed_split,
                intermediate_splits,
                f"Collapsed split {list(collapsed_split.indices)} still present in tree",
            )

    def test_all_expanded_splits_applied(self):
        """Verify ALL expanded splits are applied to final tree after collapsing conflicts."""
        # Get splits for both trees
        splits1 = self.tree1.to_splits()
        splits2 = self.tree2.to_splits()

        # Splits to be collapsed = in tree1 but not in tree2
        to_collapse = splits1 - splits2

        # Splits to be expanded = in tree2 but not in tree1
        to_expand = splits2 - splits1

        # Start with tree1
        intermediate_tree = self.tree1.deep_copy()

        print("\n=== PHASE 1: Collapsing conflicting splits ===")
        print(f"Starting tree has {len(intermediate_tree.to_splits())} splits")
        print(f"Need to collapse {len(to_collapse)} splits first")

        # PHASE 1: Collapse all conflicting splits from tree1
        for collapse_split in to_collapse:
            names = [collapse_split.reverse_encoding[i] for i in collapse_split.indices]
            print(
                f"Collapsing split {list(collapse_split.indices)} = ({', '.join(names)})"
            )

            # Set branch length to 0 to mark for collapse
            node = intermediate_tree.find_node_by_split(collapse_split)
            if node:
                node.length = 0.0

        # Actually collapse zero-length branches
        collapse_zero_length_branches_for_node(intermediate_tree, tol=0.0)

        print(f"After collapse: tree has {len(intermediate_tree.to_splits())} splits")
        print("Tree structure after collapse:")
        print(intermediate_tree.to_newick())

        print("\n=== PHASE 2: Expanding new splits ===")
        print(f"Need to expand {len(to_expand)} splits")

        # PHASE 2: Now apply all expand splits from tree2
        for expand_split in to_expand:
            names = [expand_split.reverse_encoding[i] for i in expand_split.indices]
            print(
                f"\nApplying split {list(expand_split.indices)} = ({', '.join(names)})"
            )

            # Apply split should succeed now that conflicts are removed
            try:
                apply_split_simple(expand_split, intermediate_tree)
                print("✓ Successfully applied split")
            except Exception as e:
                print(f"✗ FAILED to apply split: {e}")
                print("\nTree structure at failure:")
                print(intermediate_tree.to_newick())
                self.fail(
                    f"Failed to apply expand split {list(expand_split.indices)} after collapsing: {e}"
                )

        print(f"\nFinal tree has {len(intermediate_tree.to_splits())} splits")
        print("Final tree structure:")
        print(intermediate_tree.to_newick())

        # Verify: ALL collapsed splits should be gone
        final_splits = intermediate_tree.to_splits()
        for collapsed_split in to_collapse:
            self.assertNotIn(
                collapsed_split,
                final_splits,
                f"Collapsed split {list(collapsed_split.indices)} still present in final tree",
            )

        # Verify: ALL expanded splits should now be present
        for expanded_split in to_expand:
            self.assertIn(
                expanded_split,
                final_splits,
                f"Expanded split {list(expanded_split.indices)} not present in final tree",
            )

    def test_all_subtrees_processed_in_plan(self):
        """Verify ALL subtrees in plan are processed."""
        # Use compute_pivot_solutions_with_deletions to get jumping subtrees
        jumping_subtrees, deleted_taxa = LatticeSolver(
            self.tree1, self.tree2
        ).solve_iteratively()

        if not jumping_subtrees:
            self.skipTest("No jumping subtrees found between these trees")

        # Get an active changing edge (first one with solutions)
        active_edge = next(iter(jumping_subtrees.keys()))

        # Prepare subtree paths
        subtree_paths = prepare_simple_subtree_paths(
            self.tree1, self.tree2, active_edge, jumping_subtrees
        )

        # Build edge plan
        plan = build_edge_plan(
            subtree_paths["expand_splits_by_subtree"],
            subtree_paths["collapse_splits_by_subtree"],
            self.tree1,
            self.tree2,
            active_edge,
        )

        # Verify: ALL subtrees have plans
        all_subtrees = set(subtree_paths["collapse_splits_by_subtree"].keys()) | set(
            subtree_paths["expand_splits_by_subtree"].keys()
        )

        processed_subtrees = set(plan.keys())

        self.assertEqual(
            all_subtrees,
            processed_subtrees,
            f"Not all subtrees processed. Missing: {all_subtrees - processed_subtrees}",
        )

    def test_plan_covers_all_collapse_splits(self):
        """Verify plan includes ALL collapse splits across all subtrees."""
        # Use compute_pivot_solutions_with_deletions
        jumping_subtrees, _ = LatticeSolver(self.tree1, self.tree2).solve_iteratively()

        if not jumping_subtrees:
            self.skipTest("No jumping subtrees found")

        active_edge = next(iter(jumping_subtrees.keys()))
        subtree_paths = prepare_simple_subtree_paths(
            self.tree1, self.tree2, active_edge, jumping_subtrees
        )

        plan = build_edge_plan(
            subtree_paths["expand_splits_by_subtree"],
            subtree_paths["collapse_splits_by_subtree"],
            self.tree1,
            self.tree2,
            active_edge,
        )

        # Collect all collapse splits from original subtree assignments
        all_expected_collapse_splits = PartitionSet(encoding=self.encoding)
        for splits in subtree_paths["collapse_splits_by_subtree"].values():
            all_expected_collapse_splits |= splits

        # Collect all collapse splits from plan
        all_planned_collapse_splits = PartitionSet(encoding=self.encoding)
        for subtree_plan in plan.values():
            collapse_path = subtree_plan["collapse"]["path_segment"]
            all_planned_collapse_splits |= PartitionSet(
                collapse_path, encoding=self.encoding
            )

        # Verify: ALL expected collapse splits appear somewhere in the plan
        for split in all_expected_collapse_splits:
            self.assertIn(
                split,
                all_planned_collapse_splits,
                f"Collapse split {list(split.indices)} not in any subtree plan",
            )

    def test_plan_covers_all_expand_splits(self):
        """Verify plan includes ALL expand splits across all subtrees."""
        jumping_subtrees, _ = LatticeSolver(self.tree1, self.tree2).solve_iteratively()

        if not jumping_subtrees:
            self.skipTest("No jumping subtrees found")

        active_edge = next(iter(jumping_subtrees.keys()))
        subtree_paths = prepare_simple_subtree_paths(
            self.tree1, self.tree2, active_edge, jumping_subtrees
        )

        plan = build_edge_plan(
            subtree_paths["expand_splits_by_subtree"],
            subtree_paths["collapse_splits_by_subtree"],
            self.tree1,
            self.tree2,
            active_edge,
        )

        # Collect all expand splits from original subtree assignments
        all_expected_expand_splits = PartitionSet(encoding=self.encoding)
        for splits in subtree_paths["expand_splits_by_subtree"].values():
            all_expected_expand_splits |= splits

        # Collect all expand splits from plan
        all_planned_expand_splits = PartitionSet(encoding=self.encoding)
        for subtree_plan in plan.values():
            expand_path = subtree_plan["expand"]["path_segment"]
            all_planned_expand_splits |= PartitionSet(
                expand_path, encoding=self.encoding
            )

        # Verify: ALL expected expand splits appear somewhere in the plan
        for split in all_expected_expand_splits:
            self.assertIn(
                split,
                all_planned_expand_splits,
                f"Expand split {list(split.indices)} not in any subtree plan",
            )

    def test_split_application_error_handling(self):
        """Verify error is thrown when split cannot be applied."""
        from brancharchitect.tree_interpolation.topology_ops.expand import (
            SplitApplicationError,
        )

        # Create a split that is incompatible with tree structure
        # Example: try to apply a split that would create a contradiction

        tree = self.tree1.deep_copy()
        original_splits = tree.to_splits()

        # Create an artificial split that should fail
        # This split groups taxa that are already separated in the tree
        # Making it incompatible
        impossible_indices = (0, 1, 5, 6)  # O1, O2, B, B1 - non-contiguous in tree
        impossible_split = Partition(impossible_indices, self.encoding)

        # Only test if this split is not already in the tree
        if impossible_split not in original_splits:
            # Attempting to apply should either succeed (if compatible) or raise SplitApplicationError
            # The key is: it should NOT silently fail
            try:
                apply_split_simple(impossible_split, tree)
                # If it succeeded, verify it's actually in the tree
                self.assertIn(
                    impossible_split,
                    tree.to_splits(),
                    "Split application claimed success but split not in tree",
                )
            except SplitApplicationError as e:
                # Expected behavior: incompatible splits raise SplitApplicationError
                self.assertIn(
                    "incompatible",
                    str(e).lower(),
                    f"Error message should mention incompatibility: {e}",
                )

    def test_iterate_lattice_algorithm_integration(self):
        """Test complete workflow with compute_pivot_solutions_with_deletions."""
        # Run compute_pivot_solutions_with_deletions
        jumping_subtrees, deleted_taxa = LatticeSolver(
            self.tree1, self.tree2
        ).solve_iteratively()

        # Should find solutions
        self.assertGreater(
            len(jumping_subtrees),
            0,
            "compute_pivot_solutions_with_deletions found no solutions",
        )

        # Each edge should have valid subtree partitions
        for active_edge, subtrees in jumping_subtrees.items():
            self.assertIsInstance(active_edge, Partition)
            self.assertGreater(len(subtrees), 0, f"No solutions for {active_edge}")
            for subtree in subtrees:
                self.assertGreater(len(subtree), 0, "Empty subtree partition found")

    def test_complete_interpolation_workflow(self):
        """Test complete interpolation: collapse → expand → verify."""
        # Use compute_pivot_solutions_with_deletions
        jumping_subtrees, _ = LatticeSolver(self.tree1, self.tree2).solve_iteratively()

        if not jumping_subtrees:
            self.skipTest("No jumping subtrees needed")

        # Step 3: Execute plan
        current_tree = self.tree1.deep_copy()

        for active_edge in jumping_subtrees.keys():
            # Step 1: Prepare subtree paths
            subtree_paths = prepare_simple_subtree_paths(
                current_tree, self.tree2, active_edge, jumping_subtrees
            )

            # Step 2: Build plan
            plan = build_edge_plan(
                subtree_paths["expand_splits_by_subtree"],
                subtree_paths["collapse_splits_by_subtree"],
                current_tree,
                self.tree2,
                active_edge,
            )

            for subtree, subtree_plan in plan.items():
                # Collapse phase
                collapse_splits = subtree_plan["collapse"]["path_segment"]
                split_dict = {s: 0.0 for s in collapse_splits}
                current_tree = calculate_intermediate_tree(current_tree, split_dict)

                # Collapse zero-length branches
                active_node = current_tree.find_node_by_split(active_edge)
                if active_node:
                    collapse_zero_length_branches_for_node(active_node)

                # Expand phase
                expand_splits = subtree_plan["expand"]["path_segment"]
                for expand_split in expand_splits:
                    if expand_split not in current_tree.to_splits():
                        try:
                            apply_split_simple(expand_split, current_tree)
                        except Exception as e:
                            self.fail(
                                f"Failed to apply expand split {list(expand_split.indices)} "
                                f"in subtree {list(subtree.indices)}: {e}"
                            )

        # Step 4: Verify final tree has expected topology
        final_splits = current_tree.to_splits()

        # All splits from target tree should be present (or close to it)
        target_splits = self.tree2.to_splits()

        # Count how many target splits are present
        present_count = sum(1 for s in target_splits if s in final_splits)
        coverage = present_count / len(target_splits) if target_splits else 0

        self.assertGreater(
            coverage,
            0.8,
            f"Only {present_count}/{len(target_splits)} target splits present after interpolation",
        )

    def test_no_duplicate_splits_in_plan(self):
        """Verify no split appears multiple times in same subtree plan."""
        jumping_subtrees, _ = LatticeSolver(self.tree1, self.tree2).solve_iteratively()

        if not jumping_subtrees:
            self.skipTest("No jumping subtrees")

        active_edge = next(iter(jumping_subtrees.keys()))
        subtree_paths = prepare_simple_subtree_paths(
            self.tree1, self.tree2, active_edge, jumping_subtrees
        )

        plan = build_edge_plan(
            subtree_paths["expand_splits_by_subtree"],
            subtree_paths["collapse_splits_by_subtree"],
            self.tree1,
            self.tree2,
            active_edge,
        )

        for subtree, subtree_plan in plan.items():
            # Check collapse path for duplicates
            collapse_path = subtree_plan["collapse"]["path_segment"]
            collapse_set = set()
            for split in collapse_path:
                self.assertNotIn(
                    split,
                    collapse_set,
                    f"Duplicate collapse split {list(split.indices)} in subtree {list(subtree.indices)}",
                )
                collapse_set.add(split)

            # Check expand path for duplicates
            expand_path = subtree_plan["expand"]["path_segment"]
            expand_set = set()
            for split in expand_path:
                self.assertNotIn(
                    split,
                    expand_set,
                    f"Duplicate expand split {list(split.indices)} in subtree {list(subtree.indices)}",
                )
                expand_set.add(split)


class TestLargerDatasetSplitHandling(unittest.TestCase):
    """Test split handling with larger dataset from small_example_cli.newick."""

    def setUp(self):
        """Set up with first few trees from small_example_cli.newick."""
        # First 3 trees (simpler than full dataset)
        newicks = [
            "(Emu,(((((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)),Ostrich),(((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus))))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
            "(Emu,(((((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)),Ostrich),(((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus))))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
            "(Emu,((Ostrich,(((((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus)))),(((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone)))),(GreatRhea,LesserRhea))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
        ]

        self.taxa_order = [
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

        self.trees = [parse_newick(newick, order=self.taxa_order) for newick in newicks]
        self.encoding = self.trees[0].taxa_encoding

    def test_larger_dataset_all_splits_handled(self):
        """Test that larger dataset properly handles all splits."""
        tree1, tree2 = self.trees[0], self.trees[2]

        # Run compute_pivot_solutions_with_deletions
        jumping_subtrees, _ = LatticeSolver(tree1, tree2).solve_iteratively()

        if not jumping_subtrees:
            # Trees might be identical or very similar
            return

        active_edge = next(iter(jumping_subtrees.keys()))
        subtree_paths = prepare_simple_subtree_paths(
            tree1, tree2, active_edge, jumping_subtrees
        )

        plan = build_edge_plan(
            subtree_paths["expand_splits_by_subtree"],
            subtree_paths["collapse_splits_by_subtree"],
            tree1,
            tree2,
            active_edge,
        )

        # Verify coverage
        all_collapse = PartitionSet(encoding=self.encoding)
        all_expand = PartitionSet(encoding=self.encoding)

        for splits in subtree_paths["collapse_splits_by_subtree"].values():
            all_collapse |= splits
        for splits in subtree_paths["expand_splits_by_subtree"].values():
            all_expand |= splits

        planned_collapse = PartitionSet(encoding=self.encoding)
        planned_expand = PartitionSet(encoding=self.encoding)

        for subtree_plan in plan.values():
            planned_collapse |= PartitionSet(
                subtree_plan["collapse"]["path_segment"], encoding=self.encoding
            )
            planned_expand |= PartitionSet(
                subtree_plan["expand"]["path_segment"], encoding=self.encoding
            )

        # Verify all originally assigned collapse splits are in the plan
        # Note: plan may include additional incompatible splits from collapse-first strategy
        missing_collapse = all_collapse - planned_collapse
        self.assertEqual(
            len(missing_collapse),
            0,
            f"Originally assigned collapse splits missing from plan: {missing_collapse}",
        )

        # Expand might have additional contingent splits, so check subset
        for split in all_expand:
            self.assertIn(
                split, planned_expand, f"Expand split {list(split.indices)} not planned"
            )


if __name__ == "__main__":
    unittest.main()
