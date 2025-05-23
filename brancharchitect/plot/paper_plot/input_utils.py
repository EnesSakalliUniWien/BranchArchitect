"""
TreeViz: A library for visualizing phylogenetic trees as SVG.

This module provides the main API for generating SVG visualizations
of phylogenetic trees with various layout options.
The code has been modularized for better maintainability and clarity.
"""

from typing import List, Dict, Tuple, Optional, Set, Union
from brancharchitect.plot.tree_utils import (
    prepare_highlight_edges,
)


# -----------------------------------------------------------------------------
# C. Input Processing Functions
# -----------------------------------------------------------------------------
def normalize_input_options(
    num_trees: int,
    highlight_options: Optional[Union[List[Optional[Dict]], Dict]] = None,
    enclose_subtrees: Optional[Union[List[Optional[Dict]], Dict]] = None,
    footer_texts: Optional[Union[str, List[Optional[str]]]] = None,
    node_labels: Optional[
        Union[List[Optional[Dict[str, Dict]]], Dict[str, Dict]]
    ] = None,
    cut_edges: Optional[
        Union[List[Optional[Set[Tuple[str, str]]]], Set[Tuple[str, str]]]
    ] = None,
    branch_labels: Optional[Union[List[Optional[Dict]], Dict]] = None,
) -> Tuple[
    List[Optional[Dict]],
    List[Optional[Dict]],
    List[Optional[str]],
    List[Optional[Dict[str, Dict]]],
    List[Optional[Set[Tuple[str, str]]]],
    List[Optional[Dict]],
]:
    """
    Normalize input options for multiple trees.

    Args:
        num_trees: Number of trees to process
        highlight_options: Options for highlighting branches and leaves
        enclose_subtrees: Options for enclosing subtrees
        footer_texts: Footer texts for trees
        node_labels: Node labels for trees
        cut_edges: Set of edges to cut in each tree
        branch_labels: branch label dicts for each tree

    Returns:
        Tuple of normalized lists for each option type
    """
    tree_highlights = normalize_highlight_options(num_trees, highlight_options)
    tree_enclosures = normalize_enclosure_options(num_trees, enclose_subtrees)
    tree_footers = normalize_footer_texts(num_trees, footer_texts)
    tree_node_labels = normalize_node_labels(num_trees, node_labels)
    tree_cut_edges = normalize_cut_edges(num_trees, cut_edges)
    tree_branch_labels = normalize_branch_labels(num_trees, branch_labels)

    return (
        tree_highlights,
        tree_enclosures,
        tree_footers,
        tree_node_labels,
        tree_cut_edges,
        tree_branch_labels,
    )


def normalize_highlight_options(
    num_trees: int, highlight_options: Optional[Union[List[Optional[Dict]], Dict]]
) -> List[Optional[Dict]]:
    """
    Normalize highlight options for multiple trees.

    Args:
        num_trees: Number of trees to process
        highlight_options: Options for highlighting branches and leaves

    Returns:
        Normalized list of highlight options
    """
    tree_highlights = [None] * num_trees

    if isinstance(highlight_options, list):
        for i, opt in enumerate(highlight_options[:num_trees]):
            if opt:
                # Process edge IDs to ensure proper format
                processed_opt = opt.copy()
                if "edges" in processed_opt:
                    processed_opt["edges"] = prepare_highlight_edges(
                        processed_opt["edges"]
                    )
                tree_highlights[i] = processed_opt
            else:
                tree_highlights[i] = opt
    elif isinstance(highlight_options, dict) and num_trees > 0:
        # Process edge IDs to ensure proper format
        processed_opt = highlight_options.copy()
        if "edges" in processed_opt:
            processed_opt["edges"] = prepare_highlight_edges(processed_opt["edges"])
        tree_highlights[0] = processed_opt

    return tree_highlights


def normalize_enclosure_options(
    num_trees: int, enclose_subtrees: Optional[Union[List[Optional[Dict]], Dict]]
) -> List[Optional[Dict]]:
    """
    Normalize enclosure options for multiple trees.

    Args:
        num_trees: Number of trees to process
        enclose_subtrees: Options for enclosing subtrees

    Returns:
        Normalized list of enclosure options
    """
    tree_enclosures = [None] * num_trees

    if isinstance(enclose_subtrees, list):
        for i, opt in enumerate(enclose_subtrees[:num_trees]):
            tree_enclosures[i] = opt
    elif isinstance(enclose_subtrees, dict) and num_trees > 0:
        tree_enclosures[0] = enclose_subtrees

    return tree_enclosures


def normalize_footer_texts(
    num_trees: int, footer_texts: Optional[Union[str, List[Optional[str]]]]
) -> List[Optional[str]]:
    """
    Normalize footer texts for multiple trees.

    Args:
        num_trees: Number of trees to process
        footer_texts: Footer texts for trees

    Returns:
        Normalized list of footer texts
    """
    tree_footers = [None] * num_trees

    if isinstance(footer_texts, list):
        for i, text in enumerate(footer_texts[:num_trees]):
            tree_footers[i] = text
    elif isinstance(footer_texts, str) and num_trees > 0:
        tree_footers[0] = footer_texts

    return tree_footers


def normalize_node_labels(
    num_trees: int,
    node_labels: Optional[Union[List[Optional[Dict[str, Dict]]], Dict[str, Dict]]],
) -> List[Optional[Dict[str, Dict]]]:
    """
    Normalize node labels for multiple trees.

    Args:
        num_trees: Number of trees to process
        node_labels: Node labels for trees

    Returns:
        Normalized list of node labels
    """
    tree_node_labels = [None] * num_trees

    if isinstance(node_labels, list):
        for i, labels in enumerate(node_labels[:num_trees]):
            tree_node_labels[i] = labels
    elif isinstance(node_labels, dict) and num_trees > 0:
        tree_node_labels[0] = node_labels

    return tree_node_labels


def normalize_cut_edges(
    num_trees: int,
    cut_edges: Optional[
        Union[List[Optional[Set[Tuple[str, str]]]], Set[Tuple[str, str]]]
    ],
) -> List[Optional[Set[Tuple[str, str]]]]:
    """
    Normalize cut edges for multiple trees.

    Args:
        num_trees: Number of trees to process
        cut_edges: Set of edges to cut in each tree

    Returns:
        Normalized list of cut edges
    """
    tree_cut_edges = [None] * num_trees

    if isinstance(cut_edges, list):
        for i, edges in enumerate(cut_edges[:num_trees]):
            if edges:
                tree_cut_edges[i] = prepare_highlight_edges(edges)
            else:
                tree_cut_edges[i] = edges
    elif isinstance(cut_edges, set) and num_trees > 0:
        tree_cut_edges[0] = prepare_highlight_edges(cut_edges)

    return tree_cut_edges


def normalize_branch_labels(
    num_trees: int,
    branch_labels: Optional[Union[List[Optional[Dict]], Dict]]
) -> List[Optional[Dict]]:
    """
    Normalize branch label dicts for multiple trees.

    Args:
        num_trees: Number of trees to process
        branch_labels: branch label dicts for trees

    Returns:
        Normalized list of branch label dicts
    """
    tree_branch_labels = [None] * num_trees
    if branch_labels is None:
        return tree_branch_labels
    if isinstance(branch_labels, list):
        for i, labels in enumerate(branch_labels[:num_trees]):
            tree_branch_labels[i] = labels
    elif isinstance(branch_labels, dict) and num_trees > 0:
        tree_branch_labels[0] = branch_labels
    return tree_branch_labels