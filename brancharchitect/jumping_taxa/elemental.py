import numpy as np
import sys
from typing import List, Callable, Optional, Collection
from brancharchitect.jumping_taxa.functional_tree import FunctionalTree, ComponentSet, X, Y

# ==== Functional Programming Style ====
def remove_last_component_if_longer_than_one(
    component_set: ComponentSet,
) -> ComponentSet:
    if len(component_set) != 1:
        return component_set[:-1]
    else:
        return component_set


# ============================================== Set Based Operations ====================================================== #
def cartesian(c1: Collection[X], c2: Collection[Y]) -> list[tuple[X, Y]]:
    r: list[tuple[X, Y]] = []
    for x in c1:
        for y in c2:
            r.append((x, y))
    return r


def map1(f: Callable[[X], Y], l: Collection[tuple[X, X]]) -> list[Y]:
    r: list[Y] = []
    for x in l:
        r.append(f(x))
    return r


def map2(f: Callable[[X, X], Y], l: Collection[tuple[X, X]]) -> list[Y]:
    r: list[Y] = []
    for x in l:
        a, b = x
        r.append(f(a, b))
    return r


def reduce(f: Callable[[X, X], X], l: Collection[X]) -> Optional[X]:
    r: list[X] = []
    if len(l) == 0:
        return []
    x0: X = l[0]  # type: ignore
    for x in l[1:]:  # type: ignore
        x0 = f(x0, x)
    return x0


def union(a: List[X], b: List[X]) -> List[X]:
    return a + b


def count(a: List[X], x: X) -> int:
    return a.count(x)


def size(a: Collection[X]) -> int:
    return len(a)


def argmax(l: Collection[X], f: Callable[[X], int]) -> list[X]:
    count: int = -1
    args: list[X] = []
    for x in l:
        c: int = f(x)
        if c > count:
            args = [x]
            count = c
        elif c == count:
            args.append(x)
    return args


def argmin(l: Collection[X], f: Callable[[X], float]) -> list[X]:
    count: float = np.inf
    args: list[X] = []
    for x in l:
        c = f(x)
        if c < count:
            args = [x]
            count = c
        elif c == count:
            args.append(x)
    return args


def filter_(f: Callable[[X], bool], l: Collection[X]):
    r: list[X] = []
    for i in l:
        if f(i):
            r.append(i)
    return r


def intersect(a: Collection[X], b: Collection[X]) -> Collection[X]:
    a_uniques = set(a)
    b_uniques = set(b)
    a_and_b = a_uniques.intersection(b_uniques)
    return a_and_b


def symm(a: list[X], b: list[X]) -> list[X]:
    a_uniques = set(a)
    b_uniques = set(b)
    a_and_b = a_uniques.symmetric_difference(b_uniques)
    return a_and_b


def filter_components_from_arms(cond, arms):
    filtered_arms = []
    for component_set in arms:
        filtered_component_set = []
        for component in component_set:
            if cond(component):
                filtered_component_set.append(component)
        if filtered_component_set:
            filtered_arms.append(tuple(filtered_component_set))
    return filtered_arms


def remove_empty_lists(lst):
    return [item for item in lst if item != []]


def merge_sedges(edge_set_one, edge_set_two):
    d = {}

    for e in edge_set_one:
        d[e.split_indices] = e

    for e in edge_set_two:
        d[e.split_indices] = e

    return list(d.values())


def decode_indices_to_taxa(high_list: List, sorted_nodes: List):
    for_one_tree_decoded = []
    for leave_index in high_list:
        for_one_tree_decoded.append(sorted_nodes[leave_index])
    return for_one_tree_decoded


# ============================================== Calculate Component Set ====================================================== #

def calculate_component_set(t: FunctionalTree, sedge) -> list[ComponentSet]:
    return t._arms[sedge.split_indices]
