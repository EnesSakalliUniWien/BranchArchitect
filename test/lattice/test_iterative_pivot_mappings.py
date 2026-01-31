"""
Unit tests for map_iterative_pivot_edges_to_original in
brancharchitect/jumping_taxa/lattice/mapping/iterative_pivot_mappings.py

Covers:
- Direct pivot mapping (no jumping taxa) -> choose MAXIMUM non-root containing split
- Pivot with jumping taxa -> choose MINIMUM containing split
- Fallback to root when no non-root split contains pivot ∪ J
- Integration-like check on bootstrap_52 scenario: mapped pivots are valid
  original common splits and contain pivot ∪ first-solution taxa
"""

from __future__ import annotations

from typing import List, Tuple

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.mapping.iterative_pivot_mappings import (
    map_iterative_pivot_edges_to_original,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)


def _build_identical_small_tree():
    # Tree: (((A,B),C),(D,E));
    trees = parse_newick("(((A,B),C),(D,E));(((A,B),C),(D,E));")
    return trees[0], trees[1]


def _build_different_trees():
    """Build two trees with topological difference.

    T1: (((A,B),C),(D,E)) - C is sibling to clade (A,B)
    T2: ((A,(B,C)),(D,E)) - C has moved to be sibling of B under A's parent

    Common splits: root, (D,E), (A,B,C)
    Pivot edge: The (A,B,C) clade has different internal structure.
    """
    trees = parse_newick("(((A,B),C),(D,E));((A,(B,C)),(D,E));")
    return trees[0], trees[1]


def _enc(tree):
    return tree.taxa_encoding


def _p(names: Tuple[str, ...], enc) -> Partition:
    return Partition(tuple(sorted(enc[n] for n in names)), enc)


def test_direct_pivot_maps_to_minimum_containing_split():
    """Test that a pivot edge with no jumping taxa maps to itself (minimum split)."""
    t1, t2 = _build_different_trees()
    enc = _enc(t1)

    # (A,B,C) is a common split and is a pivot edge (has different internal structure)
    pivot = _p(("A", "B", "C"), enc)

    # No jumping taxa -> maps to itself (the minimum containing common split)
    mapped = map_iterative_pivot_edges_to_original([pivot], t1, t2, [[]])

    # Expect the same split since it's already a common split
    expected = _p(("A", "B", "C"), enc)
    assert len(mapped) == 1
    assert mapped[0] == expected


def test_pivot_with_jumping_maps_to_minimum_containing_split():
    """Test that pivot + jumping taxa maps to minimum common split containing both."""
    t1, t2 = _build_different_trees()
    enc = _enc(t1)

    # Use (D,E) as a common split that exists in both trees
    pivot = _p(("D", "E"), enc)
    # Jumping taxa: adding A means we need a split containing {D,E,A}
    # The only common split containing all of these is the root
    jumping = _p(("A",), enc)

    mapped = map_iterative_pivot_edges_to_original([pivot], t1, t2, [[jumping]])

    # Minimal containing split for {D,E} ∪ {A} is the root (A,B,C,D,E)
    expected = _p(("A", "B", "C", "D", "E"), enc)
    assert len(mapped) == 1
    assert mapped[0] == expected


def test_pivot_with_jumping_falls_back_to_root_if_needed():
    """Test that when no non-root split contains pivot ∪ jumping, we get the root."""
    t1, t2 = _build_different_trees()
    enc = _enc(t1)

    # (A,B) only exists in T1, not T2 - but the function should still work
    # by finding the minimum common split containing the taxa
    pivot = _p(("A", "B"), enc)
    jumping = _p(("D",), enc)

    mapped = map_iterative_pivot_edges_to_original([pivot], t1, t2, [[jumping]])

    # No non-root common split contains {A,B} ∪ {D}, so expect root split
    root = t1.split_indices  # full set
    assert len(mapped) == 1
    assert mapped[0] == root


def test_bootstrap_52_mapping_produces_valid_common_splits():
    # Load bootstrap_52 scenario
    import json
    from pathlib import Path

    data = json.loads(
        Path("test/colouring/trees/bootstrap_52/bootstrap_52_test.json").read_text()
    )
    trees = parse_newick(data["tree1"] + data["tree2"])
    orig_t1, orig_t2 = trees[0], trees[1]

    # Get pivots and solutions from lattice
    sols_dict = LatticeSolver(orig_t1, orig_t2).solve()
    pivot_edges = list(sols_dict.keys())
    solutions_list: List[List[Partition]] = list(sols_dict.values())

    # Map to original trees
    mapped = map_iterative_pivot_edges_to_original(
        pivot_edges, orig_t1, orig_t2, solutions_list
    )

    # All mapped splits must be in the original common splits
    common = orig_t1.to_splits() & orig_t2.to_splits()
    common_masks = {p.bitmask for p in common}
    for idx, mp in enumerate(mapped):
        assert mp.bitmask in common_masks

        # Validate containment: mapped must contain pivot ∪ first solution set's taxa
        expected_mask = pivot_edges[idx].bitmask
        if solutions_list[idx]:
            for part in solutions_list[idx]:
                expected_mask |= part.bitmask
        assert (expected_mask & mp.bitmask) == expected_mask
