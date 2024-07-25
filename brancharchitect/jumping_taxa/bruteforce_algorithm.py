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

    if node.split_indices in s and all(is_component):
            return [node.split_indices], True
    return components, False


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


def get_components(t1, t2):
    s1 = set(t1.to_splits())
    s2 = set(t2.to_splits())
    c1, _ = _get_components(t1, s2)
    c2, _ = _get_components(t2, s1)

    c1 = reduce(c1, c2)
    c2 = reduce(c2, c1)
    components = list(set(c1) | set(c2))

    return components


def algorithm(t1, t2, _=None, timeout=60*60):
    jts = core(t1, t2, timeout)
    sorted(jts, key=lambda solution: sum(len(component) for component in solution))
    return jts[0]


def core(t1, t2, timeout=60*60, max_depth=None):
    leaves = [c for c in traverse(t1) if len(c.children) == 0]
    if max_depth is None:
        max_depth = int(len(leaves) / 2)

    jumping_taxa = []
    depth = 1

    start = time.time()

    stack = [(t1, t2, [0], [get_components(t1, t2)])]
    i = 0
    while len(jumping_taxa) == 0 and depth < max_depth:

        while(len(stack)) > 0:

            t1, t2, idxs, cs = stack.pop()

            li = idxs[-1]
            lc = cs[-1]

            if li >= len(lc):
                if len(stack) >= 1:
                    t1, t2, idxs, cs = stack.pop()
                    stack.append((t1, t2, idxs[:-1] + [idxs[-1]+1], cs))
            elif len(idxs) < depth:
                stack.append((t1, t2, idxs, cs))

                to_remove = lc[li]
                it1 = delete_taxa(t1.deep_copy(), to_remove)
                it2 = delete_taxa(t2.deep_copy(), to_remove)

                stack.append((it1, it2, idxs + [0], cs + [get_components(it1, it2)]))
            else:
                to_remove = lc[li]
                it1 = delete_taxa(t1.deep_copy(), to_remove)
                it2 = delete_taxa(t2.deep_copy(), to_remove)

                if trees_equal(it1, it2):
                    jumping_taxa.append(list(tuple([t for t in c[i]]) for c, i in zip(cs, idxs)))

                stack.append((t1, t2, idxs[:-1] + [idxs[-1]+1], cs))

            i += 1
            if i % 1000 == 0:
                if time.time() - start >= timeout:
                    raise ValueError('too much time, giving up')

        depth += 1
        stack = [(t1, t2, [0], [get_components(t1, t2)])]
    return jumping_taxa

