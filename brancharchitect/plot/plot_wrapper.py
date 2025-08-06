"""
Wrapper functions for paper plots with automatic display and error handling.
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from brancharchitect.tree import Node
from brancharchitect.plot.paper_plots import render_trees_to_svg
from brancharchitect.plot.circular_bezier_trees import (
    plot_tree_row_with_beziers_and_distances,
)
from brancharchitect.plot.display_utils import (
    display_plot_output,
    generate_output_path,
    check_output_file,
)


def plot_trees_with_display(
    roots: List[Node],
    output_path: Optional[str] = None,
    output_format: str = "png",
    display: bool = True,
    display_options: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """
    Plot trees using render_trees_to_svg with automatic display.

    Args:
        roots: List of tree roots
        output_path: Output file path (auto-generated if None)
        output_format: Output format (png, pdf, svg)
        display: Whether to display the result
        display_options: Display options (width, height, method)
        **kwargs: All arguments for render_trees_to_svg

    Returns:
        SVG string
    """
    # Generate output path if not provided
    if output_path is None:
        output_path = generate_output_path("tree_plot", format=output_format)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Set up output options
    output_opts = kwargs.get("output_opts", {})
    if output_format == "pdf":
        output_opts["pdf_path"] = output_path
    kwargs["output_opts"] = output_opts

    # Generate SVG
    svg_string = render_trees_to_svg(roots, **kwargs)

    # Save to file based on format
    if output_format == "svg":
        with open(output_path, "w") as f:
            f.write(svg_string)
    elif output_format == "png":
        try:
            import cairosvg

            cairosvg.svg2png(
                bytestring=svg_string.encode("utf-8"), write_to=output_path
            )
        except ImportError:
            print("CairoSVG not installed. Saving as SVG instead.")
            output_path = output_path.replace(".png", ".svg")
            with open(output_path, "w") as f:
                f.write(svg_string)
    elif output_format == "pdf":
        # Already handled by render_trees_to_svg
        pass

    # Display if requested
    if display:
        if check_output_file(output_path):
            display_plot_output(output_path, display_options)
        else:
            print(f"Warning: Output file not found: {output_path}")

    return svg_string


def plot_tree_row_with_display(
    trees: List[Any],
    output_path: Optional[str] = None,
    output_format: str = "png",
    display: bool = True,
    display_options: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """
    Plot tree row with beziers using automatic display.

    Args:
        trees: List of trees
        output_path: Output file path (auto-generated if None)
        output_format: Output format (png, pdf, svg)
        display: Whether to display the result
        display_options: Display options (width, height, method)
        **kwargs: All arguments for plot_tree_row_with_beziers_and_distances
    """
    # Generate output path if not provided
    if output_path is None:
        output_path = generate_output_path("tree_row", format=output_format)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Set appropriate format parameters
    if output_format == "pdf":
        kwargs["output_path"] = output_path
        kwargs["save_format"] = "pdf"
    elif output_format == "png":
        kwargs["output_path"] = output_path
        kwargs["save_format"] = "png"
    elif output_format == "svg":
        kwargs["svg_output_path"] = output_path

    # Generate plot
    plot_tree_row_with_beziers_and_distances(trees, **kwargs)

    # Display if requested
    if display:
        if check_output_file(output_path):
            display_plot_output(output_path, display_options)
        else:
            print(f"Warning: Output file not found: {output_path}")


# Convenience functions for common use cases
def quick_plot_trees(
    roots: Union[Node, List[Node]],
    title: Optional[str] = None,
    width: int = 800,
    **kwargs,
) -> None:
    """
    Quick plotting function with sensible defaults.

    Args:
        roots: Single tree or list of trees
        title: Plot title
        width: Display width
        **kwargs: Additional options
    """
    if isinstance(roots, Node):
        roots = [roots]

    plot_trees_with_display(
        roots,
        output_format="png",
        display=True,
        display_options={"width": width},
        caption=title,
        **kwargs,
    )


def save_trees_pdf(roots: Union[Node, List[Node]], output_path: str, **kwargs) -> None:
    """
    Save trees directly to PDF without display.

    Args:
        roots: Single tree or list of trees
        output_path: PDF output path
        **kwargs: Additional options
    """
    if isinstance(roots, Node):
        roots = [roots]

    plot_trees_with_display(
        roots, output_path=output_path, output_format="pdf", display=False, **kwargs
    )
