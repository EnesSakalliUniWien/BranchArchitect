from brancharchitect.tree import Node, SplitIndices
from collections import Counter
from typing import Tuple, List

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


def component_distance(tree1: Node, tree2: Node, components: List[tuple[str, ...]], weighted: bool=False) -> list[float]:
    translator = ComponentTranslator(tree1)
    components = [translator.to_idx(c) for c in components]
    distances = []

    for component in components:
        d1 = jump_distance(tree1, tree2.to_splits(), component, weighted=weighted)
        d2 = jump_distance(tree2, tree1.to_splits(), component, weighted=weighted)
        distances.append(d1+d2)
    return distances


def jump_distance(node: Node, reference: Node, component: SplitIndices, weighted=False) -> float:
    path = jump_path(node, reference, component)
    if weighted:
        return sum(node.length for node in path)
    else:
        return len(path)


def jump_path(node, reference, component):
    path = []
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


def jump_path_distance(tree1: Node, tree2: Node, components: List[tuple[str, ...]], weighted:bool = False) -> dict[tuple[str, ...], float]:
    translator = ComponentTranslator(tree1)
    components = [translator.to_idx(c) for c in components]

    paths1 = [jump_path(tree1, tree2.to_splits(), component) for component in components]
    paths2 = [jump_path(tree2, tree1.to_splits(), component) for component in components]

    counter = Counter(node.split_indices for path in paths1+paths2 for node in path)

    distances = []
    for p1, p2 in zip(paths1, paths2):
        print([n.split_indices for n in p1], [n.split_indices for n in p2])
        if weighted:
            d = sum(n.length / counter[n.split_indices] for n in p1+p2)
        else:
            d = sum(1 / counter[n.split_indices] for n in p1+p2)
        distances.append(d)
    return distances
