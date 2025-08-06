from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.algorithm_5.algorithm_5 import algorithm_five


def call_jumping_taxa(
    tree1: Node, tree2: Node, algorithm: str = "rule"
) -> list[tuple[int, ...]]:
    """
    Example: calls an algorithm from brancharchitect.jumping_taxa and
    logs debug-level info to see how solutions come about.
    """
    # 2) Import the modules that contain your algorithms.
    #    (You only need to do this once at the top of the file, but here’s an example inline)
    import brancharchitect.jumping_taxa.lattice.lattice_solver
    import brancharchitect.jumping_taxa.just_set_based.algo_new
    import brancharchitect.jumping_taxa.bruteforce.bruteforce_algorithm

    # 3) Prepare a lookup of available algorithms
    ALGORITHMS = {
        "rule": algorithm_five,
        "set": brancharchitect.jumping_taxa.just_set_based.algo_new.algorithm,
        "bruteforce": brancharchitect.jumping_taxa.bruteforce.bruteforce_algorithm,
        "lattice": brancharchitect.jumping_taxa.lattice.lattice_solver.adapter_iterate_lattice_algorithm,
    }

    # 4) Sanity checks
    if tree1._order != tree2._order:
        raise ValueError("Trees have incompatible leaf order")
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"algorithm {algorithm} not supported, only supported values are {list(ALGORITHMS.keys())}"
        )

    # 5) Interpolate the two trees
    # it1, c1, c2, it2 = interpolate_tree(tree1, tree2)

    # 6) Choose the desired algorithm
    f = ALGORITHMS[algorithm]

    copy_tree_one = tree1.deep_copy()
    copy_tree_two = tree2.deep_copy()

    # 7) Call the algorithm (Algorithm 5 or otherwise).
    #    Because we set logging to DEBUG above, all .debug() calls inside
    #    'algorithm_five' will now be printed here.
    jumping_taxa = f(copy_tree_one, copy_tree_two, tree1._order)

    return jumping_taxa
