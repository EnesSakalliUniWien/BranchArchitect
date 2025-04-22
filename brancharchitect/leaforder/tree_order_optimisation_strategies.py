# In a Jupyter Notebook cell
from typing import List
from brancharchitect.tree import Node

# from brancharchitect.io import read_newick

from brancharchitect.consensus_tree import create_majority_consensus_tree_extended
from brancharchitect.leaforder.tree_order_optimisation_local import (
    smooth_order_unique_sedge,
)

from brancharchitect.leaforder.tree_order_optimisation_global import (
    generate_permutations,
    find_minimal_distance_permutation,
    optimize_tree_rotation,
)


def order_by_global_consensus(
    trees: List[Node],
    iterations_tree_optimization=20,
    n_iterations=20,
    pair_wise_optimization=True,
    back_and_forth=True,
):
    mcte = create_majority_consensus_tree_extended(trees)
    # # Optimize tree rotation
    optimized_tree, distances = optimize_tree_rotation(mcte, trees, n_iterations=iterations_tree_optimization)
    optimized_order = optimized_tree.get_current_order()
    for tree in trees:
        tree.reorder_taxa(optimized_order)
    if pair_wise_optimization:
        smooth_order_unique_sedge(
            trees, n_iterations=n_iterations, backward=back_and_forth
        )

def order_by_random_permutation(
    trees,
    num_permutations=100,
    pair_wise_optimisation=True,
    n_iterations=20,
    back_and_forth=True,
):
    # --- Method 3: Global Optimal Permutation ---
    taxa = sorted({leaf.name for tree in trees for leaf in tree.get_leaves()})
    # Generate random permutations and find the one that minimizes total distances
    random_perms = generate_permutations(taxa, sample_size=num_permutations, seed=42)
    minimal_perm = find_minimal_distance_permutation(trees, random_perms)
    for tree in trees:
        tree.reorder_taxa(minimal_perm)
    if pair_wise_optimisation:
        smooth_order_unique_sedge(
            trees, n_iterations=n_iterations, backward=back_and_forth
        )
