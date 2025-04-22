# Utility functions for input normalization, moved from paper_plots.py

def normalize_input_options(num_trees, highlight_options=None, enclose_subtrees=None, footer_texts=None, node_labels=None, cut_edges=None):
    """
    Normalize input options for tree visualization.

    Args:
        num_trees (int): Number of trees.
        highlight_options (list, optional): Highlight options for trees.
        enclose_subtrees (list, optional): Subtrees to enclose.
        footer_texts (list, optional): Footer texts for trees.
        node_labels (list, optional): Node labels for trees.
        cut_edges (list, optional): Edges to cut in trees.

    Returns:
        dict: Normalized input options.
    """
    return {
        'highlight_options': normalize_highlight_options(num_trees, highlight_options),
        'enclose_subtrees': normalize_enclosure_options(num_trees, enclose_subtrees),
        'footer_texts': normalize_footer_texts(num_trees, footer_texts),
        'node_labels': normalize_node_labels(num_trees, node_labels),
        'cut_edges': normalize_cut_edges(num_trees, cut_edges),
    }

def normalize_highlight_options(num_trees, highlight_options):
    """
    Normalize highlight options for trees.

    Args:
        num_trees (int): Number of trees.
        highlight_options (list): Highlight options for trees.

    Returns:
        list: Normalized highlight options.
    """
    if highlight_options is None:
        return [None] * num_trees
    return highlight_options

def normalize_enclosure_options(num_trees, enclose_subtrees):
    """
    Normalize enclosure options for trees.

    Args:
        num_trees (int): Number of trees.
        enclose_subtrees (list): Subtrees to enclose.

    Returns:
        list: Normalized enclosure options.
    """
    if enclose_subtrees is None:
        return [None] * num_trees
    return enclose_subtrees

def normalize_footer_texts(num_trees, footer_texts):
    """
    Normalize footer texts for trees.

    Args:
        num_trees (int): Number of trees.
        footer_texts (list): Footer texts for trees.

    Returns:
        list: Normalized footer texts.
    """
    if footer_texts is None:
        return [None] * num_trees
    return footer_texts

def normalize_node_labels(num_trees, node_labels):
    """
    Normalize node labels for trees.

    Args:
        num_trees (int): Number of trees.
        node_labels (list): Node labels for trees.

    Returns:
        list: Normalized node labels.
    """
    if node_labels is None:
        return [None] * num_trees
    return node_labels

def normalize_cut_edges(num_trees, cut_edges):
    """
    Normalize cut edges for trees.

    Args:
        num_trees (int): Number of trees.
        cut_edges (list): Edges to cut in trees.

    Returns:
        list: Normalized cut edges.
    """
    if cut_edges is None:
        return [None] * num_trees
    return cut_edges