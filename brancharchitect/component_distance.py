from brancharchitect.tree import Node, SplitIndices
from typing import Tuple, List


def component_distance(tree1: Node, tree2: Node, components: List[tuple[str, ...]], weighted: bool=False) -> dict[tuple[str, ...], float]:
    names_to_splits = {c: tree1._index(c) for c in components}
    # string component can be "in the wrong order", so we need to remember the original order
    splits_to_names = {s: name for name, s in names_to_splits.items()}
    components = [names_to_splits[component] for component in components]

    distances: dict[SplitIndices, float] = {}

    for component in components:
        d1 = jump_distance(tree1, tree2.to_splits(), component, weighted=weighted)
        d2 = jump_distance(tree2, tree1.to_splits(), component, weighted=weighted)
        distances[component] = (d1 + d2)

    translated_distances = {splits_to_names[split]: value for split, value in distances.items()}
    return translated_distances


def jump_distance(node: Node, reference: Node, component: SplitIndices, distance=0, weighted=False) -> int:

    if node.split_indices in reference:
        distance = 0
    else:
        if weighted:
            distance += node.length
        else:
            distance += 1

    for child in node.children:
        if set(component) < set(child.split_indices):
            #only one child can contain the component
            return jump_distance(child, reference, component, distance, weighted=weighted)

    return distance
