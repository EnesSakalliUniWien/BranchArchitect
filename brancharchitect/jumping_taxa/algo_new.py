from brancharchitect.jumping_taxa.deletion_algorithm import delete_taxa

def traverse(root):
    yield root
    for child in root.children:
        yield from traverse(child)

def multisplit(node):
    assert len(node.children) > 0
    m = set(child.split_indices for child in node.children)
    return m

def multi_intersection(m1, m2):
    result = set()
    for a in m1:
        for b in m2:
            result.add(tuple(sorted(set(a) & set(b))))
    return result

def multi_symmetric_difference(m1, m2):
    result = set()
    for a in m1:
        for b in m2:
            result.add(tuple(sorted(set(a) ^ set(b))))
    return result

def algorithm(t1, t2, _):
    jt = []
    i = 0
    leaves = sum(1 for c in traverse(t1) if len(c.children) == 0)
    while s := core(t1, t2):
        i += 1
        jt.extend(s)

        to_delete = [idx for split in s for idx in split]

        t1 = delete_taxa(t1, to_delete)
        t2 = delete_taxa(t2, to_delete)

        if i >= 2*leaves:
            raise Exception(f'There are only {leaves} many taxa, however we are at iteration {i}. Something is wrong.')
    return jt

def core(t1, t2):

    s1 = {node.split_indices: node for node in traverse(t1)}
    s2 = {node.split_indices: node for node in traverse(t2)}

    # note that `a | b` is `a.union(b)` and `a & b` is `a.intersection(b)` and `a ^ b` is `a.symmetric_difference(b)`

    all_splits = set(s1) | set(s2)

    jumping_taxa = []

    for split in all_splits:
        if len(split) == 1:
            continue
        if split in s1 and split in s2:
            m1 = multisplit(s1[split])
            m2 = multisplit(s2[split])

            i = m1 & m2

            if m1 != i != m2:
                mi = multi_intersection(m1, m2)
                ms = multi_symmetric_difference(m1, m2)

                print('mi', mi)
                print('ms', ms)
                r = mi & ms
                r = [split for split in r if len(split) > 0]
                print('r', r)
                if len(r) == 0:
                    jt = sorted(i, key=len)[0]
                else:
                    jt = sorted(r, key=len)[0]
                jumping_taxa.append(jt)
    return jumping_taxa





