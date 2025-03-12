from typing import List, Dict, Set
from brancharchitect.tree import Node
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score


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
    co_clustering_freq = {}
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
        clades = []
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
