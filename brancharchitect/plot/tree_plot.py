from brancharchitect.plot.svg import generate_svg_two_trees
from brancharchitect.tree import Node
from IPython.display import SVG, display
from brancharchitect.plot.svg import generate_svg_multiple_trees
from typing import List

#####################################################################
# 13) Simple convenience functions for Jupyter
#####################################################################


def plot_circular_tree_pair(
    tree1: Node,
    tree2: Node,
    width: int = 400,
    height: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = True,
) -> str:
    """
    Display exactly two circular trees side-by-side in a Jupyter notebook.

    Returns:
        svg_content (str): The raw SVG string, so you can copy or save it directly.
    """
    # We'll treat 'width // 2' as the bounding size for each tree, so total = width
    size_per_tree = width // 2
    svg_content = generate_svg_two_trees(
        tree1,
        tree2,
        size=size_per_tree,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
    )
    # Display in Jupyter
    display(SVG(svg_content))
    # Also return the raw SVG string for direct copying
    return svg_content


def plot_circular_trees_in_a_row(
    roots: List[Node],
    size: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """
    Display N circular trees side by side in a Jupyter notebook.
    Each tree has a bounding box of size x size.

    Returns:
        svg_content (str): The raw SVG string, so you can copy or save it directly.
    """
    svg_content = generate_svg_multiple_trees(
        roots=roots,
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
    )
    # Display in Jupyter
    display(SVG(svg_content))
    # Also return the raw SVG string for direct copying
    return svg_content
