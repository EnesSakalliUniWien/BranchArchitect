import numpy as np
from typing import List, Callable, Optional, Collection
from brancharchitect.jumping_taxa.functional_tree import (
    FunctionalTree,
    ComponentSet,
    X,
    Y,
)
from logging import getLogger

logger = getLogger(__name__)


def max_relative_intersection(A, B):
    """
    Identifies the pair(s) (a, b) from A × B where:
        1. b is a subset of a (i.e., every element in b is also in a),
        2. The intersection |a ∩ b| is maximized.

    Returns the intersection tuple p the maximum size, e.g., ((1,),)

    Parameters:
        A (tuple): Collection of components, each being a tuple of tuples.
        B (tuple): Another collection of components, each being a tuple of tuples.

    Returns:
        tuple: The intersection tuple with the maximum size, e.g., ((1,),)
    """

    # -------------------------
    # STEP 1: CARTESIAN PRODUCT
    # -------------------------
    cartesian_product = cartesian(A, B)

    # --------------------------------
    # STEP 2: COMPUTE INTERSECTIONS
    # --------------------------------
    intersections = map2(intersect, cartesian_product)

    # --------------------------------------------------
    # STEP 3: ZIP PAIRS WITH THEIR INTERSECTIONS
    # --------------------------------------------------
    pairs_and_inters = list(zip(cartesian_product, intersections))

    # --------------------------------------------------
    # STEP 4: FILTER EXACT MATCHES WHERE INTERSECTION == B
    # --------------------------------------------------
    exact_match_inters = [
        inter for (pair, inter) in pairs_and_inters if inter == set(pair[1])
    ]

    # --------------------------------------------------
    # STEP 5: IDENTIFY THE MAXIMUM INTERSECTION SIZE
    # --------------------------------------------------
    if not exact_match_inters:
        print("No exact matches found.")
        return None  # Or handle as appropriate

    max_size = max(size(inter) for inter in exact_match_inters)

    # --------------------------------------------------
    # STEP 6: SELECT THE INTERSECTIONS WITH MAXIMUM SIZE
    # --------------------------------------------------
    best_inters = [inter for inter in exact_match_inters if size(inter) == max_size]

    if not best_inters:
        print("No intersections found.")
        return None

    # -----------------------------------------------------------
    # STEP 7: PRINT AND RETURN THE RESULT
    # -----------------------------------------------------------
    best_inters = best_inters[0]
    print("Max intersection is:", best_inters)
    return tuple(best_inters)


# ==== Functional Programming Style ====
def remove_last_component_if_longer_than_one(
    component_set: ComponentSet,
) -> ComponentSet:
    sorted_component_set = sorted(component_set, key=lambda x: len(x))
    if len(sorted_component_set) != 1:
        return list(component_set)[:-1]
    else:
        return component_set


def find_exact_max_intersection(A, B):
    """
    Identifies the pair(s) (a, b) from A × B where:
        1. a == b,
        2. |a ∩ b| is maximized.

    Mathematically:
        BestPairs = { (a, b) ∈ A × B | a == b and |a ∩ b| = max_{(a', b') ∈ A × B} |a' ∩ b'| }

    Parameters:
        A (tuple): Collection of components, each being a tuple of tuples.
        B (tuple): Another collection of components, each being a tuple of tuples.

    Returns:
        tuple: The intersection tuple with the maximum size, e.g., ((1,),)
    """

    # -------------------------
    # STEP 1: CARTESIAN PRODUCT
    # -------------------------
    # Generate all possible ordered pairs (a, b) where a ∈ A and b ∈ B.
    cartesian_product = cartesian(A, B)

    # --------------------------------
    # STEP 2: COMPUTE INTERSECTIONS
    # --------------------------------
    # For each pair (a, b), compute the intersection a ∩ b using the 'intersect' function.
    intersections = map2(intersect, cartesian_product)

    # --------------------------------------------------
    # STEP 3: ZIP PAIRS WITH THEIR INTERSECTIONS
    # --------------------------------------------------
    # Combine each pair with its corresponding intersection.
    pairs_and_inters = list(zip(cartesian_product, intersections))

    # --------------------------------------------------
    # STEP 4: FILTER EXACT MATCHES WHERE a == b AND INTERSECTION == b
    # --------------------------------------------------
    # Select only those pairs where a exactly equals b and the intersection equals b.
    exact_match_pairs = [
        (pair, inter)
        for (pair, inter) in pairs_and_inters
        if pair[0] == pair[1] and inter == set(pair[1])
    ]

    # --------------------------------------------------
    # STEP 5: IDENTIFY THE MAXIMUM INTERSECTION SIZE
    # --------------------------------------------------
    if not exact_match_pairs:
        print("No exact matches found.")
        return None  # Or handle as appropriate

    # Determine the maximum size among exact matches.
    max_size = max(size(inter) for (_, inter) in exact_match_pairs)

    # --------------------------------------------------
    # STEP 6: SELECT THE INTERSECTIONS WITH MAXIMUM SIZE
    # --------------------------------------------------
    # Extract all intersections that have the maximum size.
    best_inters = [inter for (_, inter) in exact_match_pairs if size(inter) == max_size]

    if not best_inters:
        print("No intersections found.")
        return None

    # -----------------------------------------------------------
    # STEP 7: PRINT AND RETURN THE RESULT
    # -----------------------------------------------------------
    # Assuming you want to print the intersection and return it as a tuple.
    best_inter = best_inters[0]  # Select the first best intersection.
    print("Max intersection is:", best_inter)
    return list(tuple(best_inter))
    # Expected Return Value: ((1,),)


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


def union(a: set, b: set) -> List[X]:
    _a = set(a)
    _b = set(b)
    return _a.union(_b)


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


def cut(a: Collection[X], b: Collection[X]) -> Collection[X]:
    a_uniques = set(a)
    b_uniques = set(b)
    a_and_b = a_uniques.intersection(b_uniques)
    return sorted(list(a_and_b))
