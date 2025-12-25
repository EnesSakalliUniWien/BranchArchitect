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
from brancharchitect.jumping_taxa.lattice.solvers.pivot_edge_solver import (
    lattice_algorithm,
)


def _build_identical_small_tree():
    # Tree: (((A,B),C),(D,E));
    trees = parse_newick("(((A,B),C),(D,E));(((A,B),C),(D,E));")
    return trees[0], trees[1]


def _enc(tree):
    return tree.taxa_encoding


def _p(names: Tuple[str, ...], enc) -> Partition:
    return Partition(tuple(sorted(enc[n] for n in names)), enc)


def test_direct_pivot_maps_to_maximum_nonroot_split():
    t1, t2 = _build_identical_small_tree()
    enc = _enc(t1)

    # Pivot edge under left big clade: (A,B)
    pivot = _p(("A", "B"), enc)

    # No jumping taxa -> direct pivot
    mapped = map_iterative_pivot_edges_to_original([pivot], t1, t2, [[]])

    # Expect the split that strictly matches the pivot in the pruned context: (A,B)
    expected = _p(("A", "B"), enc)
    assert len(mapped) == 1
    assert mapped[0] == expected


def test_pivot_with_jumping_maps_to_minimum_containing_split():
    t1, t2 = _build_identical_small_tree()
    enc = _enc(t1)

    pivot = _p(("A", "B"), enc)
    # Jumping taxa include C
    jumping = _p(("C",), enc)

    mapped = map_iterative_pivot_edges_to_original([pivot], t1, t2, [[jumping]])

    # Minimal containing split for {A,B} ∪ {C} is (A,B,C)
    expected = _p(("A", "B", "C"), enc)
    assert len(mapped) == 1
    assert mapped[0] == expected


def test_pivot_with_jumping_falls_back_to_root_if_needed():
    t1, t2 = _build_identical_small_tree()
    enc = _enc(t1)

    pivot = _p(("A", "B"), enc)
    jumping = _p(("D",), enc)

    mapped = map_iterative_pivot_edges_to_original([pivot], t1, t2, [[jumping]])

    # No non-root split contains {A,B} ∪ {D}, so expect root split
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
    sols_dict = lattice_algorithm(orig_t1, orig_t2, orig_t1, orig_t2)
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
