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
    from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
        LatticeSolver,
    )
    from typing import List, Tuple, Set

    def _lattice_adapter(
        t1: Node, t2: Node, _order: List[str]
    ) -> List[Tuple[int, ...]]:
        solutions, deleted_taxa_list = LatticeSolver(t1, t2).solve_iteratively()
        # Collect all deleted indices
        all_deleted = set()
        for s in deleted_taxa_list:
            all_deleted.update(s)

        result = []
        seen = set()
        for partitions in solutions.values():
            for p in partitions:
                indices = tuple(sorted(p.resolve_to_indices()))
                if indices and any(idx in all_deleted for idx in indices):
                    if indices not in seen:
                        seen.add(indices)
                        result.append(indices)
        return result

    # 3) Prepare a lookup of available algorithms
    ALGORITHMS = {
        "bruteforce": brancharchitect.jumping_taxa.bruteforce.bruteforce_algorithm,
        "lattice": _lattice_adapter,
    }
    # 6) Choose the desired algorithm
    f = ALGORITHMS[algorithm]
    copy_tree_one = tree1.deep_copy()
    copy_tree_two = tree2.deep_copy()
    jumping_taxa = f(copy_tree_one, copy_tree_two, list(tree1.get_current_order()))
    return jumping_taxa
