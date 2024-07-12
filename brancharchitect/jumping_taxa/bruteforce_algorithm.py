from brancharchitect.jumping_taxa.deletion_algorithm import delete_taxa
from itertools import combinations
import time

def traverse(root):
    yield root
    for child in root.children:
        yield from traverse(child)


def trees_equal(t1, t2):
    split_set_1 = set(n.split_indices for n in traverse(t1))
    split_set_2 = set(n.split_indices for n in traverse(t2))
    return split_set_1 == split_set_2


def contains(a, b):
    return all(bb in a for bb in b)


def reduce(components, others):
    r = []
    for component in components:
        for other in others:
            if other == component:
                continue
            if contains(component, other):
                break
        else:
            r.append(component)
    return r


def _get_components(node, s):
    if len(node.children) == 0:
        return [node.split_indices], True

    is_component = []
    components = [] 
    for child in node.children:
        cs, is_c = _get_components(child, s)
        is_component.append(is_c)
        components.extend(cs)

    if all(is_component) and node.split_indices in s:
            return [node.split_indices], True
    return components, False


def get_components(t1, t2):
    s1 = set(t1.to_splits())
    s2 = set(t2.to_splits())
    s = s1 & s2

    c1, _ = _get_components(t1, s)
    c2, _ = _get_components(t2, s)

    c1 = reduce(c1, c2)
    c2 = reduce(c2, c1)
    components = list(set(c1) | set(c2))
    return components


def algorithm(t1, t2, _, timeout=60*60):
    components = get_components(t1, t2)
    jts = core(t1, t2, components, timeout)
    return jts


def core(t1, t2, components, timeout=60*60):
    leaves = [c for c in traverse(t1) if len(c.children) == 0]

    jumping_taxa = []
    depth = 1

    start = time.time()

    while len(jumping_taxa) == 0 and depth < int(len(leaves)/2):
        i = 0

        for candidates in combinations(components, depth):
            to_remove = [i for c in candidates for i in c]
            it1 = delete_taxa(t1.deep_copy(), to_remove)
            it2 = delete_taxa(t2.deep_copy(), to_remove)

            if trees_equal(it1, it2):
                jumping_taxa.append(candidates)

            i += 1
            if i % 1000 == 0:
                if time.time() - start >= timeout:
                    raise ValueError('too much time, giving up')
        depth += 1
    return jumping_taxa

