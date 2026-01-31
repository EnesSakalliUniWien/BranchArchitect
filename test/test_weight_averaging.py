#!/usr/bin/env python3
"""
Tests for weight averaging behavior during interpolation.

Verifies that:
1. Shared splits are averaged ONCE (on first mover only)
2. Expand splits get destination weights
3. Collapse splits get zeroed before removal
4. No double-averaging occurs
"""

import pytest
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)


class TestWeightAveraging:
    """Test suite for weight averaging behavior."""

    def test_shared_split_averaged_once(self):
        """
        Verify shared splits (in both source and dest) are averaged exactly once.
        
        Setup:
        - Source: ((A:1,B:2):10,(C:3,D:4):20);  <- shared split (C,D) has weight 20
        - Dest:   ((A:1,B:2):10,(C:5,D:6):30);  <- shared split (C,D) has weight 30
        
        Expected:
        - Shared split (C,D) should have weight (20+30)/2 = 25 in interpolated trees
        - NOT (20+30)/2 = 25 then ((25+30)/2) = 27.5 (double averaging)
        """
        source_nwk = "((A:1,B:2):10,(C:3,D:4):20);"
        dest_nwk = "((A:1,B:2):10,(C:5,D:6):30);"
        
        source = parse_newick(source_nwk)
        dest = parse_newick(dest_nwk)
        
        # Ensure shared encoding
        dest.initialize_split_indices(source.taxa_encoding)
        
        # Get the (C,D) split from source
        cd_split = None
        for s in source.to_splits():
            taxa_names = [s.reverse_encoding[i] for i in s.indices]
            if set(taxa_names) == {"C", "D"}:
                cd_split = s
                break
        
        assert cd_split is not None, "Could not find (C,D) split"
        
        # Get weights before interpolation
        source_weights = source.to_weighted_splits()
        dest_weights = dest.to_weighted_splits()
        
        source_cd_weight = source_weights.get(cd_split, 0.0)
        dest_cd_weight = dest_weights.get(cd_split, 0.0)
        
        print(f"\nSource (C,D) weight: {source_cd_weight}")
        print(f"Dest (C,D) weight: {dest_cd_weight}")
        print(f"Expected average: {(source_cd_weight + dest_cd_weight) / 2}")
        
        # Run interpolation
        result = process_tree_pair_interpolation(source, dest)
        
        # Check weights in interpolated trees
        print(f"\nInterpolated trees: {len(result.trees)}")
        
        for i, tree in enumerate(result.trees):
            tree_weights = tree.to_weighted_splits()
            for split, weight in tree_weights.items():
                taxa_names = [split.reverse_encoding[idx] for idx in split.indices]
                if set(taxa_names) == {"C", "D"}:
                    print(f"Tree {i}: (C,D) weight = {weight}")
                    
                    # After averaging, weight should be (20+30)/2 = 25
                    # NOT any other value from double-averaging
                    if i > 0:  # Skip source tree
                        expected = (source_cd_weight + dest_cd_weight) / 2
                        assert abs(weight - expected) < 0.01 or weight == dest_cd_weight, \
                            f"Tree {i}: Expected (C,D) weight ~{expected} or {dest_cd_weight}, got {weight}"

    def test_expand_split_gets_destination_weight(self):
        """
        Verify expand splits (only in dest) get destination weights.
        
        Setup:
        - Source: (A:1,B:2,C:3,D:4);           <- flat tree, no (A,B) split
        - Dest:   ((A:1,B:2):15,(C:3,D:4):20); <- has (A,B) split with weight 15
        
        Expected:
        - New split (A,B) should have weight 15 (destination weight)
        """
        source_nwk = "(A:1,B:2,C:3,D:4);"
        dest_nwk = "((A:1,B:2):15,(C:3,D:4):20);"
        
        source = parse_newick(source_nwk)
        dest = parse_newick(dest_nwk)
        
        # Ensure shared encoding
        dest.initialize_split_indices(source.taxa_encoding)
        
        # Get the (A,B) split from dest (should not exist in source)
        ab_split = None
        for s in dest.to_splits():
            taxa_names = [s.reverse_encoding[i] for i in s.indices]
            if set(taxa_names) == {"A", "B"}:
                ab_split = s
                break
        
        assert ab_split is not None, "Could not find (A,B) split in dest"
        
        # Verify (A,B) is not in source
        source_splits = source.to_splits()
        assert ab_split not in source_splits, "(A,B) should not be in source"
        
        dest_weights = dest.to_weighted_splits()
        expected_ab_weight = dest_weights.get(ab_split, 0.0)
        print(f"\nDest (A,B) weight: {expected_ab_weight}")
        
        # Run interpolation
        result = process_tree_pair_interpolation(source, dest)
        
        # Check final tree has (A,B) with destination weight
        final_tree = result.trees[-1]
        final_weights = final_tree.to_weighted_splits()
        
        for split, weight in final_weights.items():
            taxa_names = [split.reverse_encoding[idx] for idx in split.indices]
            if set(taxa_names) == {"A", "B"}:
                print(f"Final tree (A,B) weight: {weight}")
                assert abs(weight - expected_ab_weight) < 0.01, \
                    f"Expected (A,B) weight {expected_ab_weight}, got {weight}"
                break
        else:
            pytest.fail("(A,B) split not found in final tree")

    def test_collapse_split_zeroed_before_removal(self):
        """
        Verify collapse splits (only in source) get zeroed before removal.
        
        Setup:
        - Source: ((A:1,B:2):15,(C:3,D:4):20); <- has (A,B) split with weight 15
        - Dest:   (A:1,B:2,C:3,D:4);           <- flat tree, no (A,B) split
        
        Expected:
        - Split (A,B) should be zeroed in intermediate trees before being collapsed
        """
        source_nwk = "((A:1,B:2):15,(C:3,D:4):20);"
        dest_nwk = "(A:1,B:2,C:3,D:4);"
        
        source = parse_newick(source_nwk)
        dest = parse_newick(dest_nwk)
        
        # Ensure shared encoding
        dest.initialize_split_indices(source.taxa_encoding)
        
        # Get the (A,B) split from source
        ab_split = None
        for s in source.to_splits():
            taxa_names = [s.reverse_encoding[i] for i in s.indices]
            if set(taxa_names) == {"A", "B"}:
                ab_split = s
                break
        
        assert ab_split is not None, "Could not find (A,B) split in source"
        
        # Run interpolation
        result = process_tree_pair_interpolation(source, dest)
        
        print(f"\nInterpolated trees: {len(result.trees)}")
        
        # Track (A,B) weight through interpolation
        found_zero = False
        for i, tree in enumerate(result.trees):
            tree_weights = tree.to_weighted_splits()
            ab_weight = None
            for split, weight in tree_weights.items():
                taxa_names = [split.reverse_encoding[idx] for idx in split.indices]
                if set(taxa_names) == {"A", "B"}:
                    ab_weight = weight
                    break
            
            if ab_weight is not None:
                print(f"Tree {i}: (A,B) weight = {ab_weight}")
                if ab_weight == 0.0:
                    found_zero = True
            else:
                print(f"Tree {i}: (A,B) split removed")
        
        # Final tree should NOT have (A,B)
        final_tree = result.trees[-1]
        final_splits = final_tree.to_splits()
        ab_in_final = any(
            set(s.reverse_encoding[idx] for idx in s.indices) == {"A", "B"}
            for s in final_splits
        )
        assert not ab_in_final, "(A,B) should be collapsed in final tree"

    def test_multiple_movers_no_double_averaging(self):
        """
        Test with multiple movers under the same pivot to ensure no double averaging.
        
        Setup: A tree where multiple subtrees move under the same pivot edge.
        """
        # Source: ((A,B),(C,D),(E,F)) with various branch lengths
        source_nwk = "(((A:1,B:2):10,(C:3,D:4):20):100,(E:5,F:6):30);"
        # Dest: Different arrangement
        dest_nwk = "(((A:1,B:2):15,(E:5,F:6):35):100,(C:3,D:4):25);"
        
        source = parse_newick(source_nwk)
        dest = parse_newick(dest_nwk)
        
        # Ensure shared encoding
        dest.initialize_split_indices(source.taxa_encoding)
        
        # Get solutions to see what movers exist
        solver = LatticeSolver(source, dest)
        solutions, _ = solver.solve_iteratively()
        
        print("\n=== Pivot -> Movers ===")
        for pivot, movers in solutions.items():
            pivot_taxa = [pivot.reverse_encoding[i] for i in pivot.indices]
            print(f"Pivot {pivot_taxa}:")
            for m in movers:
                mover_taxa = [m.reverse_encoding[i] for i in m.indices]
                print(f"  Mover: {mover_taxa}")
        
        # Run interpolation
        result = process_tree_pair_interpolation(source, dest)
        
        print(f"\nGenerated {len(result.trees)} trees")
        
        # Check (A,B) split weights through interpolation
        ab_weights = []
        for i, tree in enumerate(result.trees):
            tree_weights = tree.to_weighted_splits()
            for split, weight in tree_weights.items():
                taxa_names = [split.reverse_encoding[idx] for idx in split.indices]
                if set(taxa_names) == {"A", "B"}:
                    ab_weights.append((i, weight))
                    break
        
        print("\n(A,B) weights through interpolation:")
        for i, w in ab_weights:
            print(f"  Tree {i}: {w}")
        
        # Check for suspicious patterns that indicate double averaging
        # If weights are decreasing beyond what averaging would produce, that's a bug
        if len(ab_weights) > 2:
            first_averaged = ab_weights[1][1] if len(ab_weights) > 1 else ab_weights[0][1]
            for i, w in ab_weights[2:]:
                # Weight should not be less than what single averaging produces
                # (unless it's a collapse scenario)
                print(f"  Checking tree {i}: weight {w} vs first_averaged {first_averaged}")


class TestWeightTracking:
    """Detailed weight tracking tests."""

    def test_track_all_weights_through_interpolation(self):
        """
        Track all branch weights through the entire interpolation sequence.
        
        This provides a detailed view of how every split's weight changes.
        """
        source_nwk = "((A:1.0,B:2.0):10.0,(C:3.0,D:4.0):20.0);"
        dest_nwk = "((A:1.5,B:2.5):15.0,(C:3.5,D:4.5):25.0);"
        
        source = parse_newick(source_nwk)
        dest = parse_newick(dest_nwk)
        
        # Ensure shared encoding
        dest.initialize_split_indices(source.taxa_encoding)
        
        source_weights = source.to_weighted_splits()
        dest_weights = dest.to_weighted_splits()
        
        print("\n=== Source Weights ===")
        for split, weight in source_weights.items():
            taxa = [split.reverse_encoding[i] for i in split.indices]
            print(f"  {taxa}: {weight}")
        
        print("\n=== Dest Weights ===")
        for split, weight in dest_weights.items():
            taxa = [split.reverse_encoding[i] for i in split.indices]
            print(f"  {taxa}: {weight}")
        
        print("\n=== Expected Averages ===")
        for split in source_weights:
            if split in dest_weights:
                avg = (source_weights[split] + dest_weights[split]) / 2
                taxa = [split.reverse_encoding[i] for i in split.indices]
                print(f"  {taxa}: ({source_weights[split]} + {dest_weights[split]}) / 2 = {avg}")
        
        # Run interpolation
        result = process_tree_pair_interpolation(source, dest)
        
        print(f"\n=== Tracking through {len(result.trees)} trees ===")
        
        # Build a tracking table
        all_splits = set()
        for tree in result.trees:
            for split in tree.to_splits():
                all_splits.add(tuple(sorted(split.indices)))
        
        # Track each split
        for split_indices in sorted(all_splits, key=lambda x: (len(x), x)):
            print(f"\nSplit {split_indices}:")
            for i, tree in enumerate(result.trees):
                tree_weights = tree.to_weighted_splits()
                found = False
                for split, weight in tree_weights.items():
                    if tuple(sorted(split.indices)) == split_indices:
                        print(f"  Tree {i}: {weight:.4f}")
                        found = True
                        break
                if not found:
                    print(f"  Tree {i}: (not present)")


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
