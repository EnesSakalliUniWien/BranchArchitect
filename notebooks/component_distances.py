from brancharchitect.jumping_taxa.lattice.lattice_solver import iterate_lattice_algorithm
from brancharchitect.component_distance import jump_distance
from brancharchitect.tree import Node
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable

def extract_leaf_order(trees: List[Node]) -> List[str]:
    """
    Extract a canonical leaf order from the first tree (or consensus of all trees).
    Assumes all trees have the same set of leaves.
    """
    if not trees:
        raise ValueError('No trees provided.')
    return [leaf.name for leaf in trees[0].get_leaves()]

def extract_components(trees: List[Node]) -> List[Tuple[str, ...]]:
    """
    Extract all unique components (splits) from all trees as tuples of leaf names.
    Returns a list of tuples, each representing a component.
    """
    components_set = set()
    for tree in trees:
        for split in tree.to_splits():
            # Convert split indices to leaf names
            names = tuple(sorted([tree.leaf_names[i] for i in split]))
            if 1 < len(names) < len(tree.leaf_names):  # Exclude trivial splits
                components_set.add(names)
    return sorted(components_set)

def get_lattice_solution_sizes(tree1: Node, tree2: Node, leaf_order: List[str]) -> List[int]:
    """
    Run the iterative lattice algorithm and return the sizes of all minimal reconciliation solutions.
    Returns an empty list if no solutions are found.
    """
    solutions = iterate_lattice_algorithm(tree1, tree2, leaf_order)
    if not solutions:
        return []
    return [len(sol) for sol in solutions]

def validate_component(component: Tuple[str, ...], tree1: Node, tree2: Node) -> bool:
    """
    Check if all leaves in the component are present in both trees.
    """
    leaves1 = {leaf.name for leaf in tree1.get_leaves()}
    leaves2 = {leaf.name for leaf in tree2.get_leaves()}
    return all(leaf in leaves1 and leaf in leaves2 for leaf in component)

def compute_combined_distances(
    trees: List[Node],
    components: List[Tuple[str, ...]],
    leaf_order: List[str],
    jump_distance_fn: Callable[[Node, Node, Tuple[str, ...]], float],
) -> pd.DataFrame:
    """
    Compute combined distances for all tree pairs and all components.

    For each tree pair and component, computes:
      - The jump (component) distance
      - The sizes of all minimal lattice solutions
      - The combined distances (jump + each solution size)
      - Summary statistics (min, mean, etc.)

    Args:
        trees: List of tree objects.
        components: List of components (tuples of leaf names).
        leaf_order: List of leaf names for lattice algorithm.
        jump_distance_fn: Function to compute jump distance for a component.

    Returns:
        DataFrame with columns:
            Tree1, Tree2, Component, JumpDistance, LatticeSolutionSizes, CombinedDistances,
            MinCombinedDistance, MeanCombinedDistance, etc.
    """
    records = []
    num_trees = len(trees)
    for i in range(num_trees):
        for j in range(i + 1, num_trees):
            tree1, tree2 = trees[i], trees[j]
            lattice_sizes = get_lattice_solution_sizes(tree1, tree2, leaf_order)
            for component in components:
                if not validate_component(component, tree1, tree2):
                    continue  # Skip invalid components
                jd = jump_distance_fn(tree1, tree2, component)
                combined = [jd + s for s in lattice_sizes] if lattice_sizes else [np.nan]
                record = {
                    'Tree1': i + 1,
                    'Tree2': j + 1,
                    'Component': component,
                    'JumpDistance': jd,
                    'LatticeSolutionSizes': lattice_sizes,
                    'CombinedDistances': combined,
                    'MinCombinedDistance': np.nanmin(combined) if combined and not np.isnan(combined).all() else np.nan,
                    'MeanCombinedDistance': np.nanmean(combined) if combined and not np.isnan(combined).all() else np.nan,
                }
                records.append(record)
    return pd.DataFrame.from_records(records)

# --- Main Combined Distance Analysis Pipeline ---

# 1. Extract canonical leaf order from the first tree
leaf_order = extract_leaf_order(trees)

# 2. Extract all unique components (splits) from all trees
components = extract_components(trees)
print(f"Extracted {len(components)} unique components from all trees.")

# 3. Compute the combined distances DataFrame
combined_df = compute_combined_distances(trees, components, leaf_order, jump_distance)

# 4. Build a symmetric distance matrix using the minimum combined distance for each tree pair
num_trees = len(trees)
distance_matrix = np.full((num_trees, num_trees), np.nan)

for _, row in combined_df.iterrows():
    i = int(row['Tree1']) - 1
    j = int(row['Tree2']) - 1
    # You can choose MinCombinedDistance, MeanCombinedDistance, or another summary
    distance_matrix[i, j] = row['MinCombinedDistance']
    distance_matrix[j, i] = row['MinCombinedDistance']  # Symmetric

# 5. Store for later use
distance_matrices['combined'] = distance_matrix

# 6. Create a DataFrame for plotting
tree_indices = np.arange(1, num_trees + 1)
df_distances = pd.DataFrame({
    'Tree1': np.repeat(tree_indices, num_trees),
    'Tree2': np.tile(tree_indices, num_trees),
    'combined': distance_matrix.flatten(),
})
distance_dataframes['combined'] = df_distances

# 7. Plot the distance matrix
plt.figure(figsize=(8, 6))
plot_distance_matrix(
    distance_matrix,
    ax=plt.gca(),
    title='Combined Distance Matrix'
)
plt.show()

# 8. Clustering and UMAP as before
cluster_labels = perform_clustering(distance_matrix, n_clusters=3)
cluster_labels_dict['combined'] = cluster_labels

embedding_umap = perform_umap(distance_matrix, n_components=3)
embeddings_umap['combined'] = embedding_umap

create_3d_scatter_plot(
    embedding_umap,
    tree_indices=tree_indices,
    labels=cluster_labels,
    method_name='UMAP',
    title='UMAP Embedding (Combined Distance)',
)