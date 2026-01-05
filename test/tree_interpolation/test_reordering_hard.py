"""
Hard tests for interpolate_subtree_order to stress tricky ordering cases.

Focus:
- Multi-step reordering stability: applying consecutive moves preserves
  previously established anchor and mover block order.
- Large mover blocks with conflicting anchor order in destination.
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.subtree_paths.execution.reordering import (
    reorder_tree_toward_destination,
)

# Alias for backward compatibility with test names
move_subtree_to_destination = reorder_tree_toward_destination


def _get_tree(parsed_result) -> Node:
    if isinstance(parsed_result, list):
        return parsed_result[0]
    return parsed_result


def _setup_tree_pair(source_newick: str, dest_newick: str):
    source_tree = _get_tree(parse_newick(source_newick))
    dest_tree = _get_tree(parse_newick(dest_newick))
    dest_tree.taxa_encoding = source_tree.taxa_encoding
    return source_tree, dest_tree, source_tree.taxa_encoding


def test_multi_step_reordering_stability():
    """
    Apply two successive moves and ensure stability across steps:
    - First move: bring C forward between A and B
    - Second move: move E to the front
    Verify after each step:
      - mover blocks are contiguous and preserve SOURCE internal order
      - anchors preserve SOURCE order relative to each other
    """
    # Step 0: initial
    src, dst, enc = _setup_tree_pair(
        "(A:0.1,B:0.2,C:0.3,D:0.4,E:0.5,F:0.6);",
        "(E:0.5,C:0.3,A:0.1,B:0.2,D:0.4,F:0.6);",
    )

    active = src.split_indices

    # First move: C is mover
    mover1 = Partition((2,), enc)  # C
    after_first = move_subtree_to_destination(src, dst, active, mover1)
    order1 = list(after_first.get_current_order())
    # Expect C ahead of B (inserted before first anchor after A), and anchors A,B preserve SOURCE order
    assert order1.index("C") < order1.index("B")
    assert order1.index("A") < order1.index("B")

    # Second move: now move E to the very front; use the intermediate as source
    mover2 = Partition((4,), enc)  # E
    after_second = move_subtree_to_destination(after_first, dst, active, mover2)
    order2 = list(after_second.get_current_order())

    # E should be at front
    assert order2[0] == "E"
    # Previously established relative order A < B must remain
    assert order2.index("A") < order2.index("B")
    # C remains before B (contiguous move logic respected)
    assert order2.index("C") < order2.index("B")


def test_large_mover_block_with_conflicting_dest_anchors():
    """
    Move a large block (B,C,D,E) where destination reverses anchor order.
    The mover block must retain SOURCE internal order, anchors retain SOURCE order.
    """
    src, dst, enc = _setup_tree_pair(
        "(A:0.1,B:0.2,C:0.3,D:0.4,E:0.5,F:0.6);",
        "(E:0.5,D:0.4,C:0.3,B:0.2,A:0.1,F:0.6);",
    )

    active = src.split_indices
    mover = Partition((1, 2, 3, 4), enc)  # B,C,D,E
    result = move_subtree_to_destination(src, dst, active, mover)
    order = list(result.get_current_order())

    # Mover block must be contiguous and preserve SOURCE internal order (B,C,D,E)
    start = order.index("B")
    assert order[start : start + 4] == ["B", "C", "D", "E"]

    # Anchors A and F preserve SOURCE relative order (A before F)
    assert order.index("A") < order.index("F")
