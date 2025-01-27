from brancharchitect.jumping_taxa.tree_interpolation import interpolate_tree
from brancharchitect.tree import Node


def call_jumping_taxa(tree1: Node, tree2: Node, algorithm="rule"):

    import brancharchitect.jumping_taxa.algorithm_five
    import brancharchitect.jumping_taxa.algo_new
    import brancharchitect.jumping_taxa.bruteforce_algorithm

    ALGORITHMS = {
        "rule": brancharchitect.jumping_taxa.algorithm_five.algorithm_five,
        "set": brancharchitect.jumping_taxa.algo_new.algorithm,
        "bruteforce": brancharchitect.jumping_taxa.bruteforce_algorithm.algorithm,
    }

    if tree1._order != tree2._order:
        raise ValueError("Trees have incompatible leaf order")
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"algorithm {algorithm} not supported, only supported values are {list(ALGORITHMS.keys())}"
        )

    it1, c1, c2, it2 = interpolate_tree(tree1, tree2)

    f = ALGORITHMS[algorithm]

    jumping_taxa = f(it1, it2, tree1._order)
    print(jumping_taxa)
    jumping_taxa = [tuple(tree1._order[i] for i in idx) for idx in jumping_taxa]
    return jumping_taxa
