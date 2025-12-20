from brancharchitect.tree import Node


def call_jumping_taxa(
    tree1: Node, tree2: Node, algorithm: str = "lattice"
) -> list[tuple[int, ...]]:
    """
    Example: calls an algorithm from brancharchitect.jumping_taxa and
    logs debug-level info to see how solutions come about.
    """

    # 2) Import the modules that contain your algorithms.
    #    (You only need to do this once at the top of the file, but here’s an example inline)
    import brancharchitect.jumping_taxa.bruteforce.bruteforce_algorithm
    from brancharchitect.jumping_taxa.lattice.orchestration.compute_pivot_solutions_with_deletions import (
        adapter_compute_pivot_solutions_with_deletions,
    )

    # 3) Prepare a lookup of available algorithms
    ALGORITHMS = {
        "bruteforce": brancharchitect.jumping_taxa.bruteforce.bruteforce_algorithm,
        "lattice": adapter_compute_pivot_solutions_with_deletions,
    }
    # 6) Choose the desired algorithm
    f = ALGORITHMS[algorithm]
    copy_tree_one = tree1.deep_copy()
    copy_tree_two = tree2.deep_copy()
    jumping_taxa = f(copy_tree_one, copy_tree_two, list(tree1.get_current_order()))
    return jumping_taxa
