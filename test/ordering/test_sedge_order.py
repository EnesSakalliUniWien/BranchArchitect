"""Determinism check for s-edge ordering between optimizer and interpolation.

Verifies that for the same tree pair (T_target, T_reference), the s-edge
ordering produced by the interpolation path matches the ordering used by the
optimizer (subset-aware depth ordering with deterministic tie-breakers).
"""

from typing import List, Tuple

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition

# Interpolation-side ordering utilities
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from brancharchitect.tree_interpolation.types.lattice_edge_data import (
    LatticeEdgeData,
)

# Optimizer-side utilities
from brancharchitect.leaforder.rotation_functions import (
    get_s_edge_splits,
)
from brancharchitect.jumping_taxa.lattice.depth_computation import (
    compute_lattice_edge_depths,
)


def _as_index_tuples(parts: List[Partition]) -> List[Tuple[int, ...]]:
    return [tuple(int(i) for i in p) for p in parts]


def _interpolation_order(target: Node, reference: Node) -> List[Tuple[int, ...]]:
    # Use the same discovery and ordering as the interpolation path
    sols = iterate_lattice_algorithm(target.deep_copy(), reference.deep_copy())
    edges = list(sols.keys())
    data = LatticeEdgeData(edges, sols)
    data.compute_depths(target, reference)
    ordered = data.get_sorted_edges(use_reference=False, ascending=True)
    return _as_index_tuples(ordered)


def _optimizer_order(target: Node, reference: Node) -> List[Tuple[int, ...]]:
    # Use the same ordering as optimizer: depth map + deterministic ties
    s_edges = list(get_s_edge_splits(reference, target))  # optimizer uses (ref, target)
    if not s_edges:
        return []
    depth_map = compute_lattice_edge_depths(s_edges, target)
    s_edges.sort(
        key=lambda sp: (
            depth_map.get(sp, 0),
            len(tuple(sp)),
            tuple(int(i) for i in sp),
        )
    )
    return _as_index_tuples(s_edges)


def _build_pair(nw1: str, nw2: str) -> Tuple[Node, Node]:
    t1 = parse_newick(nw1)
    t2 = parse_newick(nw2, t1._order)
    return t1, t2


def test_sedge_order_matches_on_simple_pair():
    # A small pair with a couple of structural differences
    # Target (T) vs Reference (R) newicks
    T = "((A:1,(B:1,C:1):1):1,(D:1,(E:1,F:1):1):1);"
    R = "(((A:1,B:1):1,C:1):1,((D:1,E:1):1,F:1):1);"

    target, reference = _build_pair(T, R)
    interp_order = _interpolation_order(target, reference)
    opt_order = _optimizer_order(target, reference)

    assert (
        interp_order == opt_order
    ), f"Interpolation order {interp_order} != Optimizer order {opt_order}"


def test_sedge_order_empty_for_identical_pair():
    # Identical pair should produce zero s-edges on both sides
    NW = "((A:1,B:1):1,(C:1,D:1):1);"
    t1, t2 = _build_pair(NW, NW)
    assert _interpolation_order(t1, t2) == []
    assert _optimizer_order(t1, t2) == []

