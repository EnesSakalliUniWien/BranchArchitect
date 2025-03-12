# Imports
from brancharchitect.tree import Node  # Ensure this is the correct import
from typing import List, Dict, Tuple, Set
from itertools import combinations
import numpy as np


# Function Definitions
def extract_splits(tree: Node) -> Set[Tuple[int]]:
    """
    Extract splits from a tree.

    Args:
    - tree (Node): The tree from which to extract splits.

    Returns:
    - Set[Tuple[int]]: A set of splits, where each split is represented as a tuple of indices.
    """
    # Assume tree.to_splits() returns a dictionary with split indices as keys
    return set(tree.to_splits().keys())


def get_taxa_indices(tree: Node) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Create mappings between taxa indices and names.

    Args:
    - tree (Node): The tree from which to extract taxa.

    Returns:
    - Tuple[Dict[int, str], Dict[str, int]]: Two dictionaries for index-to-taxon and taxon-to-index mappings.
    """
    index_to_taxon = {}
    taxon_to_index = {}
    for leaf in tree.get_leaves():
        index = leaf.split_indices[
            0
        ]  # Assuming split_indices is a tuple with one element
        name = leaf.name
        index_to_taxon[index] = name
        taxon_to_index[name] = index
    return index_to_taxon, taxon_to_index


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

        splits_one = set(tree_one.to_splits().keys())
        splits_two = set(tree_two.to_splits().keys())

        # Identify unique splits in both trees
        unique_splits_one = splits_one - splits_two
        unique_splits_two = splits_two - splits_one

        # Combine unique splits
        all_unique_splits = unique_splits_one.union(unique_splits_two)

        # Map split indices to taxon names
        splits_taxa = [
            set(index_to_taxon.get(idx) for idx in split if idx in index_to_taxon)
            for split in all_unique_splits
        ]

        if to_filter:
            # Filter minimal unique splits
            filtered_splits = filter_minimal_splits(splits_taxa)
        else:
            filtered_splits = splits_taxa

        total_filtered_splits += len(filtered_splits)

        for split in filtered_splits:
            taxa_in_split = list(split)
            # For each pair of taxa in the split, increment the co-occurrence count
            for taxon1, taxon2 in combinations(taxa_in_split, 2):
                idx1 = taxa_indices[taxon1]
                idx2 = taxa_indices[taxon2]
                co_occurrence_counts[idx1][idx2] += 1
                co_occurrence_counts[idx2][idx1] += 1  # Symmetric

    # Compute co-occurrence frequencies
    co_occurrence_freq = {}
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
        splits_one = set(tree_one.to_splits().keys())
        splits_two = set(tree_two.to_splits().keys())

        # Identify unique splits in both trees
        unique_splits_one = splits_one - splits_two
        unique_splits_two = splits_two - splits_one

        # Combine unique splits
        all_unique_splits = unique_splits_one.union(unique_splits_two)

        # Map split indices to taxon names
        splits_taxa = [
            set(index_to_taxon.get(idx) for idx in split if idx in index_to_taxon)
            for split in all_unique_splits
        ]

        if to_filter:
            # Filter minimal unique splits
            filtered_splits = filter_minimal_splits(splits_taxa)
        else:
            filtered_splits = splits_taxa

        total_filtered_splits += len(filtered_splits)

        for split in filtered_splits:
            taxa_in_split = list(split)
            # For each pair of taxa in the split, increment the co-occurrence count
            for taxon1, taxon2 in combinations(taxa_in_split, 2):
                idx1 = taxa_indices[taxon1]
                idx2 = taxa_indices[taxon2]
                co_occurrence_counts[idx1][idx2] += 1
                co_occurrence_counts[idx2][idx1] += 1  # Symmetric

    # Compute co-occurrence frequencies
    co_occurrence_freq = {}
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


