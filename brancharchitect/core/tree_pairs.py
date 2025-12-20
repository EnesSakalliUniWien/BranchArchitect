"""Utilities for working with consecutive tree pairs."""

from __future__ import annotations
from typing import List, Iterator, Callable, TypeVar
from brancharchitect.tree import Node

T = TypeVar("T")


def iter_consecutive_pairs(trees: List[Node]) -> Iterator[tuple[int, Node, Node, bool, bool]]:
    """
    Iterate over consecutive tree pairs with basic metadata.

    Yields tuples of (index, source, target, is_first, is_last).
    """
    if len(trees) < 2:
        return

    pair_count = len(trees) - 1

    for i in range(pair_count):
        yield (i, trees[i], trees[i + 1], i == 0, i == pair_count - 1)


def map_pairs(trees: List[Node], func: Callable[[tuple[int, Node, Node, bool, bool]], T]) -> List[T]:
    """
    Apply a function to each consecutive tree pair.

    Functional programming pattern for pair operations.
    """
    return [func(pair) for pair in iter_consecutive_pairs(trees)]


def reduce_pairs(
    trees: List[Node],
    func: Callable[[T, tuple[int, Node, Node, bool, bool]], T],
    initial: T,
) -> T:
    """
    Reduce consecutive tree pairs to a single value.
    """
    acc = initial
    for pair in iter_consecutive_pairs(trees):
        acc = func(acc, pair)
    return acc
