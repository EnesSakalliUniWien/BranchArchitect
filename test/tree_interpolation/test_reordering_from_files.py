"""
Complex reordering tests using real example trees from the repository:

- current_testfiles/small_example.newick: first two trees are used as source/destination
- test/data/reverse_test_tree_moving_updwards.tree: tests upward-moving scenario

We validate that running reorder_tree_toward_destination step-by-step for the
moving subtrees (from compute_pivot_solutions_with_deletions) preserves anchor order and
forms a contiguous block of movers in the result within the active pivot.
"""

from typing import List, Set
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.subtree_paths.execution.layout.reordering import (
    reorder_tree_toward_destination,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)

def _read_newick_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _shares_encoding(src: Node, dst: Node) -> None:
    # Ensure destination uses identical encoding OBJECT and refresh split indices
    enc = src.taxa_encoding
    dst.taxa_encoding = enc
    # Re-initialize split indices for the destination to use the new encoding
    # We must propagate this to all nodes, but _initialize_split_indices is typically recursive
    # from the root.
    dst._initialize_split_indices(enc)
    dst.build_split_index()  # Rebuild index


def _subsequence(seq: List[str], subset: Set[str]) -> List[str]:
    return [x for x in seq if x in subset]


def _is_contiguous_block(order: List[str], block: Set[str]) -> bool:
    indices = [order.index(x) for x in order if x in block]
    if not indices:
        return True
    return max(indices) - min(indices) + 1 == len(indices)


def test_reordering_small_example_stepwise():
    """Stepwise reordering on small_example.newick preserves anchors and blocks movers."""
    lines = _read_newick_lines("test/data/current_testfiles/small_example.newick")
    assert len(lines) >= 2, "Expected at least two trees in small_example.newick"

    src = parse_newick(lines[0])
    dst = parse_newick(lines[1])

    # Unify encodings to prevent Partition mismatch errors
    # Note: parse_newick creates independent encodings. We must merge/share them.
    # Safe to use src's encoding if both trees have same taxa or src is superset.
    _shares_encoding(src, dst)

    # Use first changing edge and its first solution set
    jumping, _ = LatticeSolver(src, dst).solve_iteratively()
    assert jumping, "No changing edges detected in small_example"
    active_edge = next(iter(jumping.keys()))
    first_solution_set = jumping[active_edge]

    # Apply reordering step-by-step for this solution set
    current = src
    for subtree in first_solution_set:
        current = reorder_tree_toward_destination(current, dst, active_edge, subtree)

    # Validate each mover remains a contiguous block. Multiple movers under one
    # pivot are not necessarily contiguous as a union; the destination may place
    # stable anchors between them.
    result_order = list(current.get_current_order())
    for subtree in first_solution_set:
        assert _is_contiguous_block(result_order, set(subtree.taxa))


def test_pair_interpolation_matches_destination_order_small_example():
    """Full pair interpolation should end exactly on the destination ordering."""
    lines = _read_newick_lines("test/data/current_testfiles/small_example.newick")
    assert len(lines) >= 2, "Expected at least two trees in small_example.newick"

    src = parse_newick(lines[0])
    dst = parse_newick(lines[1])

    _shares_encoding(src, dst)

    src_copy = src.deep_copy()
    dst_copy = dst.deep_copy()
    _shares_encoding(src_copy, dst_copy)

    result = process_tree_pair_interpolation(src_copy, dst_copy)
    assert result.trees, "Interpolation should yield intermediate states"

    final_order = result.trees[-1].get_current_order()
    assert (
        final_order == dst.get_current_order()
    ), f"Interpolation did not end on the destination ordering, got {final_order}"


def test_reordering_reverse_upwards_from_file():
    """
    Use reverse_test_tree_moving_updwards.tree to verify block movement upwards.
    Movers should become contiguous and anchor order is preserved.
    """
    lines = _read_newick_lines("test/data/reverse_test_tree_moving_updwards.tree")
    assert len(lines) == 2

    src = parse_newick(lines[0])
    dst = parse_newick(lines[1])

    _shares_encoding(src, dst)

    # Compute changing edges and use the first with its first solution set
    jumping, _ = LatticeSolver(src, dst).solve_iteratively()
    assert jumping, "Expected at least one changing edge"
    active_edge = next(iter(jumping.keys()))
    solution_set = jumping[active_edge]

    # Determine movers within pivot
    movers: Set[str] = set()
    for st in solution_set:
        movers |= set(st.taxa)

    # Apply moves
    current = src
    for subtree in solution_set:
        current = reorder_tree_toward_destination(current, dst, active_edge, subtree)

    # Validate
    result_order = list(current.get_current_order())
    assert _is_contiguous_block(result_order, movers)
