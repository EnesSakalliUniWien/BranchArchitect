from typing import List, Dict, Set, Tuple, Any, Optional
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

from brancharchitect.partition_set import PartitionSet
from brancharchitect.tree import Node


def compute_weighted_co_clustering(
    list_of_trees: List[Node],
) -> Dict[str, Dict[str, float]]:
    """
    Computes weighted co-clustering frequencies between taxa across multiple trees.
    The weight is based on branch lengths; taxa co-occurring in clades with shorter branch lengths
    are given higher weights.

    Args:
    - list_of_trees (List[Node]): List of trees to analyze.

    Returns:
    - Dict[str, Dict[str, float]]: Weighted co-clustering frequencies between taxa.
    """
    # Get list of taxa
    taxa = sorted({leaf.name for tree in list_of_trees for leaf in tree.get_leaves()})
    co_occurrence_weights = {
        taxon: {other_taxon: 0.0 for other_taxon in taxa} for taxon in taxa
    }
    total_weights = {taxon: 0.0 for taxon in taxa}

    for tree in list_of_trees:
        for node in tree.traverse():
            if node.children:
                clade_taxa = [leaf.name for leaf in node.get_leaves()]
                if len(clade_taxa) >= 2:
                    # Use inverse branch length as weight (shorter branches = higher weight)
                    length = node.length if node.length is not None else 0.0
                    # Avoid division by zero; add a small epsilon
                    epsilon = 1e-8
                    weight = 1.0 / (length + epsilon)
                    for taxon1, taxon2 in combinations(clade_taxa, 2):
                        co_occurrence_weights[taxon1][taxon2] += weight
                        co_occurrence_weights[taxon2][taxon1] += weight
                        total_weights[taxon1] += weight
                        total_weights[taxon2] += weight

    # Normalize weights to obtain frequencies
    co_clustering_freq: Dict[str, Dict[str, float]] = {}
    for taxon1 in taxa:
        co_clustering_freq[taxon1] = {}
        for taxon2 in taxa:
            if taxon1 == taxon2:
                co_clustering_freq[taxon1][taxon2] = 1.0
            else:
                total_weight = total_weights[taxon1] + total_weights[taxon2]
                freq = (
                    2 * co_occurrence_weights[taxon1][taxon2] / total_weight
                    if total_weight > 0
                    else 0.0
                )
                co_clustering_freq[taxon1][taxon2] = freq

    return co_clustering_freq


def compute_co_clustering_nmi(list_of_trees: List[Node]) -> Dict[str, Dict[str, float]]:
    """
    Computes co-clustering frequencies between taxa using Normalized Mutual Information (NMI).

    Args:
    - list_of_trees (List[Node]): List of trees to analyze.

    Returns:
    - Dict[str, Dict[str, float]]: NMI-based co-clustering frequencies between taxa.
    """
    # Get list of taxa
    taxa = sorted({leaf.name for tree in list_of_trees for leaf in tree.get_leaves()})
    taxon_indices = {taxon: idx for idx, taxon in enumerate(taxa)}
    num_taxa = len(taxa)

    # Initialize list to store clustering labels for each tree
    clustering_labels_per_tree = []

    for tree in list_of_trees:
        # Initialize labels for this tree
        labels = [0] * num_taxa
        cluster_id = 0
        clades: List[Set[str]] = []
        get_clades(tree, clades)
        # Assign cluster labels to taxa based on clades
        for clade in clades:
            cluster_id += 1
            for taxon in clade:
                idx = taxon_indices[taxon]
                labels[idx] = cluster_id
        clustering_labels_per_tree.append(labels)

    # Compute NMI between taxa
    co_clustering_nmi = {
        taxon: {other_taxon: 0.0 for other_taxon in taxa} for taxon in taxa
    }

    for i in range(num_taxa):
        taxon1 = taxa[i]
        co_clustering_nmi[taxon1][taxon1] = 1.0  # NMI with itself
        for j in range(i + 1, num_taxa):
            taxon2 = taxa[j]
            # Extract labels for the two taxa across all trees
            labels1 = [labels[i] for labels in clustering_labels_per_tree]
            labels2 = [labels[j] for labels in clustering_labels_per_tree]
            nmi = normalized_mutual_info_score(labels1, labels2)
            co_clustering_nmi[taxon1][taxon2] = nmi
            co_clustering_nmi[taxon2][taxon1] = nmi  # Symmetric

    return co_clustering_nmi


def get_clades(node: Node, clades: List[Set[str]]) -> Set[str]:
    """
    Recursively collects clades (sets of taxa) from the tree.

    Args:
    - node (Node): The current node.
    - clades (List[Set[str]]): List to collect clades.

    Returns:
    - Set[str]: The set of taxa under the current node.
    """
    if node.children:
        taxa = set()
        for child in node.children:
            child_taxa = get_clades(child, clades)
            taxa.update(child_taxa)
        clades.append(taxa)
        return taxa
    else:
        return {node.name}


# Function Definitions
def extract_splits(
    tree: Node,
    arms_t_one: Optional[List[Any]] = None,
    arms_t_two: Optional[List[Any]] = None,
    t1_unique_atoms: Optional[List[Any]] = None,
    t2_unique_atoms: Optional[List[Any]] = None,
    t1_unique_covers: Optional[List[Any]] = None,
    t2_unique_covers: Optional[List[Any]] = None,
) -> PartitionSet:
    """
    Extract splits from a tree.

    Args:
    - tree (Node): The tree from which to extract splits.
    - arms_t_one (Optional[List[Any]]): Optional list of arms for tree one.
    - arms_t_two (Optional[List[Any]]): Optional list of arms for tree two.
    - t1_unique_atoms (Optional[List[Any]]): Optional list of unique atoms for tree one.
    - t2_unique_atoms (Optional[List[Any]]): Optional list of unique atoms for tree two.
    - t1_unique_covers (Optional[List[Any]]): Optional list of unique covers for tree one.
    - t2_unique_covers (Optional[List[Any]]): Optional list of unique covers for tree two.

    Returns:
    - PartitionSet: A set of splits, where each split is represented as a tuple of indices.
    """
    # Assume tree.to_splits() returns a dictionary with split indices as keys
    return tree.to_splits()


def filter_minimal_splits(splits: List[Set[int]]) -> List[Set[int]]:
    """
    Filters the list of splits to only include minimal unique splits,
    i.e., splits that are not subsets of any other split in the list.

    Args:
    - splits (List[Set[int]]): List of splits represented as sets of taxon indices.

    Returns:
    - List[Set[int]]: Filtered list of minimal unique splits.
    """
    filtered_splits = []
    for split in splits:
        if not any(
            split < other_split for other_split in splits if split != other_split
        ):
            filtered_splits.append(split)
    return filtered_splits


def compute_taxon_co_occurrence_in_filtered_nonexistent_splits(
    list_of_trees: List[Node],
    to_filter: bool = True,
) -> Dict[str, Dict[str, float]]:
    # Ensure there are at least two trees to compare
    if len(list_of_trees) < 2:
        return {}

    # Get the list of taxa and create indices
    taxa = list_of_trees[0]._order
    taxa_indices = {taxon: idx for idx, taxon in enumerate(taxa)}
    num_taxa = len(taxa_indices)

    # Initialize co-occurrence counts matrix
    co_occurrence_counts = np.zeros((num_taxa, num_taxa), dtype=int)
    total_filtered_splits = 0

    # Create a mapping from indices to taxon names
    index_to_taxon = {idx: taxon for taxon, idx in taxa_indices.items()}

    # Iterate over pairs of consecutive trees
    for i in range(len(list_of_trees) - 1):
        tree_one = list_of_trees[i]
        tree_two = list_of_trees[i + 1]

        splits_one = set(tree_one.to_weighted_splits().keys())
        splits_two = set(tree_two.to_weighted_splits().keys())

        # Identify unique splits in both trees
        unique_splits_one = splits_one - splits_two
        unique_splits_two = splits_two - splits_one

        # Combine unique splits
        all_unique_splits = unique_splits_one.union(unique_splits_two)

        # Map split indices to taxon names
        splits_taxa_raw = [
            set(index_to_taxon.get(idx) for idx in split if idx in index_to_taxon)
            for split in all_unique_splits
        ]
        # Ensure splits_taxa is List[Set[int]] for filter_minimal_splits
        splits_taxa: List[Set[int]] = []
        for s in splits_taxa_raw:
            # Remove None and convert to indices
            indices = set(taxa_indices[t] for t in s if t is not None)
            splits_taxa.append(indices)

        if to_filter:
            # Filter minimal unique splits
            filtered_splits = filter_minimal_splits(splits_taxa)
        else:
            filtered_splits = splits_taxa

        total_filtered_splits += len(filtered_splits)

        for split in filtered_splits:
            taxa_in_split = [index_to_taxon[idx] for idx in split]
            # For each pair of taxa in the split, increment the co-occurrence count
            for taxon1, taxon2 in combinations(taxa_in_split, 2):
                idx1 = taxa_indices[taxon1]
                idx2 = taxa_indices[taxon2]
                co_occurrence_counts[idx1][idx2] += 1
                co_occurrence_counts[idx2][idx1] += 1  # Symmetric

    # Compute co-occurrence frequencies
    co_occurrence_freq: Dict[str, Dict[str, float]] = {}
    for i, taxon1 in enumerate(taxa):
        co_occurrence_freq[taxon1] = {}
        for j, taxon2 in enumerate(taxa):
            if i == j:
                co_occurrence_freq[taxon1][taxon2] = 1.0
            else:
                freq = (
                    co_occurrence_counts[i][j] / total_filtered_splits
                    if total_filtered_splits > 0
                    else 0.0
                )
                co_occurrence_freq[taxon1][taxon2] = freq

    return co_occurrence_freq


def compute_taxon_co_occurrence_in_filtered_nonexistent_splits_all_pairs(
    list_of_trees: List[Node],
    to_filter: bool = True,
) -> Dict[str, Dict[str, float]]:
    # Ensure there are at least two trees to compare
    if len(list_of_trees) < 2:
        return {}

    # Get the list of taxa and create indices
    taxa = list_of_trees[0]._order
    taxa_indices = {taxon: idx for idx, taxon in enumerate(taxa)}
    num_taxa = len(taxa_indices)

    # Initialize co-occurrence counts matrix
    co_occurrence_counts = np.zeros((num_taxa, num_taxa), dtype=int)
    total_filtered_splits = 0

    # Create a mapping from indices to taxon names
    index_to_taxon = {idx: taxon for taxon, idx in taxa_indices.items()}

    # Iterate over all pairs of trees
    tree_pairs = list(combinations(list_of_trees, 2))

    for tree_one, tree_two in tree_pairs:
        splits_one = set(tree_one.to_weighted_splits().keys())
        splits_two = set(tree_two.to_weighted_splits().keys())

        # Identify unique splits in both trees
        unique_splits_one = splits_one - splits_two
        unique_splits_two = splits_two - splits_one

        # Combine unique splits
        all_unique_splits = unique_splits_one.union(unique_splits_two)

        # Map split indices to taxon names
        splits_taxa_raw = [
            set(index_to_taxon.get(idx) for idx in split if idx in index_to_taxon)
            for split in all_unique_splits
        ]
        splits_taxa: List[Set[int]] = []
        for s in splits_taxa_raw:
            indices = set(taxa_indices[t] for t in s if t is not None)
            splits_taxa.append(indices)

        if to_filter:
            # Filter minimal unique splits
            filtered_splits = filter_minimal_splits(splits_taxa)
        else:
            filtered_splits = splits_taxa

        total_filtered_splits += len(filtered_splits)

        for split in filtered_splits:
            taxa_in_split = [index_to_taxon[idx] for idx in split]
            # For each pair of taxa in the split, increment the co-occurrence count
            for taxon1, taxon2 in combinations(taxa_in_split, 2):
                idx1 = taxa_indices[taxon1]
                idx2 = taxa_indices[taxon2]
                co_occurrence_counts[idx1][idx2] += 1
                co_occurrence_counts[idx2][idx1] += 1  # Symmetric

    # Compute co-occurrence frequencies
    co_occurrence_freq: Dict[str, Dict[str, float]] = {}
    for i, taxon1 in enumerate(taxa):
        co_occurrence_freq[taxon1] = {}
        for j, taxon2 in enumerate(taxa):
            if i == j:
                co_occurrence_freq[taxon1][taxon2] = 1.0
            else:
                freq = (
                    co_occurrence_counts[i][j] / total_filtered_splits
                    if total_filtered_splits > 0
                    else 0.0
                )
                co_occurrence_freq[taxon1][taxon2] = freq

    return co_occurrence_freq
