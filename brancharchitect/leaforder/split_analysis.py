import logging
from collections import OrderedDict
from typing import Optional

# Assuming these imports point to valid modules in your project structure
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition

logger = logging.getLogger(__name__)


TreePairCache = OrderedDict[tuple[int, int], tuple[Node, Node, PartitionSet[Partition]]]

_UNIQUE_SPLITS_CACHE: TreePairCache = OrderedDict()
_COMMON_SPLITS_CACHE: TreePairCache = OrderedDict()
_ACTIVE_CHANGING_SPLITS_CACHE: TreePairCache = OrderedDict()


def _get_cached_split_set(
    cache: TreePairCache, tree1: Node, tree2: Node
) -> Optional[PartitionSet[Partition]]:
    """Read cached split results by object identity, avoiding Node.__eq__."""
    key = (id(tree1), id(tree2))
    cached = cache.get(key)
    if cached is None:
        return None

    cached_tree1, cached_tree2, value = cached
    if cached_tree1 is not tree1 or cached_tree2 is not tree2:
        del cache[key]
        return None

    cache.move_to_end(key)
    return value


def _store_cached_split_set(
    cache: TreePairCache,
    tree1: Node,
    tree2: Node,
    value: PartitionSet[Partition],
    maxsize: int,
) -> PartitionSet[Partition]:
    """Store cached split results while retaining tree refs to prevent id reuse."""
    key = (id(tree1), id(tree2))
    cache[key] = (tree1, tree2, value)
    cache.move_to_end(key)

    while len(cache) > maxsize:
        cache.popitem(last=False)

    return value


def get_unique_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns the set of splits that are in tree2 but not in tree1.
    Uses an identity-based LRU cache for repeated calls on the same tree pair.
    """
    cached = _get_cached_split_set(_UNIQUE_SPLITS_CACHE, tree1, tree2)
    if cached is not None:
        return cached

    s1: PartitionSet[Partition] = tree1.to_splits()
    s2: PartitionSet[Partition] = tree2.to_splits()
    return _store_cached_split_set(_UNIQUE_SPLITS_CACHE, tree1, tree2, s2 - s1, 128)


def get_common_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns the set of splits that are common to both tree1 and tree2.
    Uses an identity-based LRU cache for repeated calls on the same tree pair.
    """
    cached = _get_cached_split_set(_COMMON_SPLITS_CACHE, tree1, tree2)
    if cached is not None:
        return cached

    s1: PartitionSet[Partition] = tree1.to_splits()
    s2: PartitionSet[Partition] = tree2.to_splits()
    return _store_cached_split_set(_COMMON_SPLITS_CACHE, tree1, tree2, s1 & s2, 128)


def get_active_changing_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns active changing splits: common splits in tree2 where children differ.
    Uses an identity-based LRU cache for repeated calls on the same tree pair.
    """
    cached = _get_cached_split_set(_ACTIVE_CHANGING_SPLITS_CACHE, tree1, tree2)
    if cached is not None:
        return cached

    tree1_copy = tree1.deep_copy()
    tree2_copy = tree2.deep_copy()
    # Use deep copies to prevent the lattice algorithm from modifying the original trees
    # LatticeSolver.solve_iteratively returns a tuple: (dict, list)
    from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
        LatticeSolver,
    )

    active_changing_split_solutions, _ = LatticeSolver(
        tree1_copy, tree2_copy
    ).solve_iteratively()
    active_changing_splits_list = list(active_changing_split_solutions.keys())
    # Convert to PartitionSet for consistency with function signature
    active_changing_splits_set: PartitionSet[Partition] = PartitionSet(
        set(active_changing_splits_list), encoding=tree1.taxa_encoding
    )
    return _store_cached_split_set(
        _ACTIVE_CHANGING_SPLITS_CACHE,
        tree1,
        tree2,
        active_changing_splits_set,
        64,
    )


def clear_split_pair_cache() -> None:
    """
    Clear the LRU caches. Call this after any tree mutation.
    """
    _UNIQUE_SPLITS_CACHE.clear()
    _COMMON_SPLITS_CACHE.clear()
    _ACTIVE_CHANGING_SPLITS_CACHE.clear()
