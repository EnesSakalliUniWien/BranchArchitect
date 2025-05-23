from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.lattice_solver import iterate_lattice_algorithm
import numpy as np
import pandas as pd
from brancharchitect.partition_set import Partition, PartitionSet
from collections import Counter
from typing import List, Tuple, Callable


class ComponentTranslator:
    def __init__(self, tree):
        self.tree = tree
        self.r_cache = {}

    def to_idx(self, c):
        idx = self.tree._index(c)
        self.r_cache[idx] = c
        return idx

    def to_str(self, c):
        return self.r_cache[c]


def component_distance(
    tree1: Node, tree2: Node, components: List[tuple[str, ...]], weighted: bool = False
) -> List[float]:
    translator = ComponentTranslator(tree1)
    # Convert tuple[str, ...] to Partition (which is likely tuple[int, ...])
    components_idx: List[Partition] = [translator.to_idx(c) for c in components]
    distances: List[float] = []

    for component in components_idx:
        d1 = jump_distance(tree1, tree2.to_splits(), component, weighted=weighted)
        d2 = jump_distance(tree2, tree1.to_splits(), component, weighted=weighted)
        distances.append(d1 + d2)
    return distances


def jump_distance(
    node: Node, reference: PartitionSet, component: Partition, weighted: bool = False
) -> float:
    path: List[Node] = jump_path(node, reference, component)
    if weighted:
        return sum(n.length for n in path if n.length is not None)
    else:
        return float(len(path))


def jump_path(node: Node, reference: PartitionSet, component: Partition) -> List[Node]:
    path: List[Node] = []
    while set(node.split_indices) != set(component):
        if node.split_indices in reference:
            path = []
        else:
            path.append(node)
        for child in node.children:
            if set(component) <= set(child.split_indices):
                node = child
                break
        else:
            break
            # Component is actually not a component
    return path


def jump_path_distance(
    tree1: Node, tree2: Node, components: List[tuple[str, ...]], weighted: bool = False
) -> List[float]:
    translator = ComponentTranslator(tree1)
    components_idx: List[Partition] = [translator.to_idx(c) for c in components]

    paths1 = [
        jump_path(tree1, tree2.to_splits(), component) for component in components_idx
    ]
    paths2 = [
        jump_path(tree2, tree1.to_splits(), component) for component in components_idx
    ]

    counter = Counter(node.split_indices for path in paths1 + paths2 for node in path)

    distances: List[float] = []
    for p1, p2 in zip(paths1, paths2):
        if weighted:
            d = sum(
                (n.length if n.length is not None else 0.0) / counter[n.split_indices]
                for n in p1 + p2
            )
        else:
            d = sum(1.0 / counter[n.split_indices] for n in p1 + p2)
        distances.append(d)
    return distances

def extract_leaf_order(trees: List[Node]) -> List[str]:
    """
    Extract a canonical leaf order from the first tree (or consensus of all trees).
    Assumes all trees have the same set of leaves.
    """
    if not trees:
        raise ValueError('No trees provided.')
    return [leaf.name for leaf in trees[0].get_leaves()]

def extract_components(trees: List[Node]) -> List[Tuple[str, ...]]:
    """
    Extract all unique components (splits) from all trees as tuples of leaf names.
    Returns a list of tuples, each representing a component.
    """
    components_set = set()
    for tree in trees:
        leaf_names = [leaf.name for leaf in tree.get_leaves()]
        for split in tree.to_splits():
            # Convert split indices to leaf names
            names = tuple(sorted([leaf_names[i] for i in split]))
            if 1 < len(names) < len(leaf_names):  # Exclude trivial splits
                components_set.add(names)
    return sorted(components_set)

def get_lattice_solution_sizes(tree1: Node, tree2: Node, leaf_order: List[str]) -> List[int]:
    """
    Run the iterative lattice algorithm and return the sizes of all minimal reconciliation solutions.
    Returns an empty list if no solutions are found.
    """
    solutions = iterate_lattice_algorithm(tree1, tree2, leaf_order)
    if not solutions:
        return []
    return [len(sol) for sol in solutions]

def validate_component(component: Tuple[str, ...], tree1: Node, tree2: Node) -> bool:
    """
    Check if all leaves in the component are present in both trees.
    """
    leaves1 = {leaf.name for leaf in tree1.get_leaves()}
    leaves2 = {leaf.name for leaf in tree2.get_leaves()}
    return all(leaf in leaves1 and leaf in leaves2 for leaf in component)

def compute_combined_distances(
    trees: List[Node],
    components: List[Tuple[str, ...]],
    leaf_order: List[str],
    jump_distance_fn: Callable[[Node, Node, Tuple[str, ...]], float],
) -> pd.DataFrame:
    """
    Compute combined distances for all tree pairs and all components.

    For each tree pair and component, computes:
      - The jump (component) distance
      - The sizes of all minimal lattice solutions
      - The combined distances (jump + each solution size)
      - Summary statistics (min, mean, etc.)

    Args:
        trees: List of tree objects.
        components: List of components (tuples of leaf names).
        leaf_order: List of leaf names for lattice algorithm.
        jump_distance_fn: Function to compute jump distance for a component.

    Returns:
        DataFrame with columns:
            Tree1, Tree2, Component, JumpDistance, LatticeSolutionSizes, CombinedDistances,
            MinCombinedDistance, MeanCombinedDistance, etc.
    """
    records = []
    num_trees = len(trees)
    for i in range(num_trees):
        for j in range(i + 1, num_trees):
            tree1, tree2 = trees[i], trees[j]
            lattice_sizes = get_lattice_solution_sizes(tree1, tree2, leaf_order)
            for component in components:
                if not validate_component(component, tree1, tree2):
                    continue  # Skip invalid components
                jd = jump_distance_fn(tree1, tree2, component)
                combined = [jd + s for s in lattice_sizes] if lattice_sizes else [np.nan]
                record = {
                    'Tree1': i + 1,
                    'Tree2': j + 1,
                    'Component': component,
                    'JumpDistance': jd,
                    'LatticeSolutionSizes': lattice_sizes,
                    'CombinedDistances': combined,
                    'MinCombinedDistance': np.nanmin(combined) if combined and not np.isnan(combined).all() else np.nan,
                    'MeanCombinedDistance': np.nanmean(combined) if combined and not np.isnan(combined).all() else np.nan,
                }
                records.append(record)
    return pd.DataFrame.from_records(records)