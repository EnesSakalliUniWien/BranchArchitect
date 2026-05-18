"""Deterministic ordering keys for mover subtrees."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

from brancharchitect.elements.partition import Partition

_MISSING_VISUAL_ORDER = 10**12


def order_key_for_subtree(
    subtree: Partition,
    subtree_order_key: Optional[Mapping[Partition, Tuple[int, ...]]],
) -> Tuple[int, ...]:
    """Return the visual-order key for a mover subtree, with bitmask fallback."""
    if not subtree_order_key:
        return (subtree.bitmask,)

    order_key = subtree_order_key.get(subtree)
    if order_key is None:
        return (_MISSING_VISUAL_ORDER, subtree.bitmask)
    return (*order_key, subtree.bitmask)
