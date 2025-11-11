"""
Helper and processing functions for tree interpolation.

This module contains utility functions for processing split data,
extracting data, and managing the interpolation workflow.
"""

from __future__ import annotations
from typing import Dict, List
from brancharchitect.elements.partition import Partition


def get_subset_splits(
    edge: Partition, current_splits: Dict[Partition, float]
) -> List[Partition]:
    """Find all splits that are subsets of a given edge."""
    return [split for split in current_splits if split.taxa.issubset(edge.taxa)]


def filter_splits_by_subset(
    splits_dict: Dict[Partition, float],
    subset_splits: List[Partition],
) -> Dict[Partition, float]:
    """Filter a splits dictionary to only include specified subset splits.

    Args:
        splits_dict: The dictionary of splits with their weights
        subset_splits: List of splits to filter by

    Returns:
        A new dictionary containing only the splits that are in subset_splits
    """
    filtered_splits = {
        split: splits_dict[split] for split in subset_splits if split in splits_dict
    }
    return filtered_splits
