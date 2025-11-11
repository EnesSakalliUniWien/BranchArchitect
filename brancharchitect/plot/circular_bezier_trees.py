from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import tempfile
import os
import importlib
import xml.etree.ElementTree as ET

# Local imports (always available)
from brancharchitect.distances.distances import robinson_foulds_distance
from brancharchitect.plot.tree_plot import (
    plot_circular_trees_in_a_row,
    add_svg_gridlines,
)


def _require_optional_deps(context: str, modules: Optional[List[str]] = None) -> None:
    """Ensure optional heavy dependencies are available with a friendly error.

    Args:
        context: Description of what feature needs the deps (for error message).
        modules: Specific modules to check; defaults to common plotting deps.
    """
    mods = modules or []
    missing: list[str] = []
    for name in mods:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        raise ImportError(
            f"The following optional dependencies are required for {context}: {', '.join(missing)}. "
            "Install them (e.g., pip install matplotlib numpy cairosvg pillow) and retry."
        )


# --- Parameter Normalization Utilities ---
@dataclass
class TreeRenderingParams:
    """Normalized parameters for tree rendering."""

    highlight_branches: List[Any]
    highlight_width: List[Union[int, float]]
    highlight_colors: List[Dict[Any, Any]]

    @classmethod
    def normalize(cls, n_trees: int, **kwargs) -> "TreeRenderingParams":
        """Normalize rendering parameters for n_trees."""
        return cls(
            highlight_branches=normalize_to_list(
                kwargs.get("highlight_branches"), n_trees
            ),
            highlight_width=normalize_to_list(
                kwargs.get("highlight_width", 4.0), n_trees
            ),
            highlight_colors=normalize_highlight_colors_list(
                kwargs.get("highlight_colors"), n_trees
            ),
        )


def normalize_to_list(value: Any, n: int) -> List[Any]:
    """Normalize a value to a list of length n."""
    if value is None:
        return [None] * n
    if isinstance(value, (int, float)):
        return [value] * n
    if isinstance(value, list):
        if len(value) == n:
            return value
        if len(value) == 0 or not isinstance(value[0], (list, tuple)):
            return [value] * n
    return [value] * n


def normalize_highlight_colors_list(
    highlight_colors: Optional[Any], n: int
) -> List[Dict[Any, Any]]:
    """Normalize highlight colors to a list of dicts."""
    if highlight_colors is None:
        return [{} for _ in range(n)]
    if isinstance(highlight_colors, list) and len(highlight_colors) == n:
        return [_normalize_single_highlight_color(hc) for hc in highlight_colors]
    return [_normalize_single_highlight_color(highlight_colors)] * n


def _normalize_single_highlight_color(color: Any) -> Dict[Any, Any]:
    """Normalize a single highlight color specification."""
    if color is None:
        return {}
    if isinstance(color, dict):
        normalized = {}
        for k, v in color.items():
            if isinstance(v, str):
                normalized[k] = {"highlight_color": v}
            else:
                normalized[k] = v
        return normalized
    return color


# --- Distance Computation Functions ---
def compute_rf_distances(trees: List[Any]) -> List[float]:
    """Compute Robinson-Foulds distances between consecutive trees."""
    return [
        robinson_foulds_distance(trees[i], trees[i + 1]) for i in range(len(trees) - 1)
    ]


def per_taxa_circular_distances(tree1: Any, tree2: Any) -> List[float]:
    """Compute per-taxa circular distances between two trees."""
    from brancharchitect.leaforder.circular_distances import circular_distance_tree_pair

    # Get the single distance between trees
    single_distance = circular_distance_tree_pair(tree1, tree2)

    # Get common taxa
    order1 = tree1.get_current_order()
    order2 = tree2.get_current_order()
    common_taxa = set(order1) & set(order2)

    # For now, assign equal distance to all taxa
    # This is a simplified implementation - could be enhanced later
    per_taxa_distances = [
        single_distance / len(common_taxa) if common_taxa else 0.0
    ] * len(common_taxa)

    return per_taxa_distances


def compute_per_pair_taxa_dists(trees: List[Any]) -> List[List[float]]:
    """Compute per-pair taxa distances for all consecutive tree pairs."""
    return [
        per_taxa_circular_distances(trees[i], trees[i + 1])
        for i in range(len(trees) - 1)
    ]


def compute_bezier_colors_and_widths(
    per_pair_taxa_dists: List[List[float]],
    norm: Any,
    subtle_color: Any,
    min_width: float,
    max_width: float,
) -> Tuple[List[List[str]], List[List[float]]]:
    """Compute Bezier curve colors and stroke widths from distance data."""
    bezier_colors_per_pair = [
        [subtle_color(norm(d)) for d in pair] for pair in per_pair_taxa_dists
    ]
    bezier_stroke_widths_per_pair = [
        [min_width + (max_width - min_width) * norm(d) for d in pair]
        for pair in per_pair_taxa_dists
    ]
    return bezier_colors_per_pair, bezier_stroke_widths_per_pair


# --- Main API Functions ---
def plot_tree_row_with_beziers_and_distances(
    trees: List[Any],
    size: int = 240,
    margin: int = 50,
    label_offset: int = 18,
    min_width: float = 4.0,
    max_width: float = 14.0,
    cmap_name: str = "viridis",
    ignore_branch_lengths: bool = False,
    font_family: str = "Monospace",
    font_size: str = "18",
    leaf_font_size: Optional[str] = None,
    stroke_color: str = "#000",
    output_path: Optional[str] = None,
    save_format: str = "png",
    show_plot: bool = True,
    gridlines: bool = True,
    highlight_branches: Optional[Any] = None,
    highlight_width: Optional[Any] = None,
    highlight_colors: Optional[Any] = None,
    show_zero_length_indicators: bool = False,
    zero_length_indicator_color: str = "#ff4444",
    zero_length_indicator_size: float = 6.0,
    bezier_colors: Optional[Any] = None,
    bezier_stroke_widths: Optional[Any] = None,
    show_distances: bool = True,  # New parameter to control distance plot generation
    **kwargs,
) -> Optional[Any]:
    """
    Create a comprehensive visualization showing trees with Bezier connections and distance plots.

    This is the main entry point for creating tree visualizations with distance analysis.

    Args:
        trees: List of tree objects to visualize
        show_distances: If True, generate and include distance plots (default: True)
        show_plot: If True, display plots in notebook output (default: True)
        Other parameters: See function signature for full parameter list
    """
    # Lazy-load heavy deps only when this function is called
    _require_optional_deps("tree plotting with distances", modules=["matplotlib", "numpy"])
    import matplotlib.pyplot as plt  # noqa: WPS433
    import matplotlib.colors as mcolors  # noqa: WPS433

    # Handle single tree case - no distances, just tree visualization
    if len(trees) == 1:
        svg_element = plot_circular_trees_in_a_row(
            trees,
            size=size,
            margin=margin,
            label_offset=label_offset,
            ignore_branch_lengths=ignore_branch_lengths,
            font_family=font_family,
            font_size=font_size,
            leaf_font_size=leaf_font_size,
            stroke_color=stroke_color,
            show_zero_length_indicators=show_zero_length_indicators,
            zero_length_indicator_color=zero_length_indicator_color,
            zero_length_indicator_size=zero_length_indicator_size,
        )

        # Handle output for single tree
        if output_path:
            svg_string = ET.tostring(svg_element, encoding="unicode")
            if save_format.lower() == "png":
                _require_optional_deps("SVG→PNG export", modules=["cairosvg"])
                import cairosvg  # type: ignore  # noqa: WPS433
                with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
                    tmp_svg.write(svg_string.encode())
                    tmp_svg.flush()
                    cairosvg.svg2png(url=tmp_svg.name, write_to=output_path, background_color="white")
                    os.unlink(tmp_svg.name)
            else:
                with open(output_path, "w") as f:
                    f.write(svg_string)

        if show_plot and output_path and os.path.exists(output_path):
            try:
                from IPython.display import Image, display  # noqa: WPS433
                display(Image(filename=output_path))
            except Exception:
                pass

        return svg_element

    if len(trees) < 2:
        raise ValueError(
            "At least 2 trees are required for Bezier connections and distance plots"
        )

    # Normalize parameters
    params = TreeRenderingParams.normalize(
        len(trees),
        highlight_branches=highlight_branches,
        highlight_width=highlight_width,
        highlight_colors=highlight_colors,
    )

    # Compute distances and generate colors/widths
    per_pair_taxa_dists = compute_per_pair_taxa_dists(trees)
    all_dists = [d for pair in per_pair_taxa_dists for d in pair]

    if bezier_colors is None or bezier_stroke_widths is None:
        if not all_dists:
            # Handle single tree case
            bezier_colors_per_pair, bezier_stroke_widths_per_pair = [], []
        else:
            # Create color mapping
            cmap = plt.cm.get_cmap(cmap_name)
            norm = plt.Normalize(vmin=min(all_dists), vmax=max(all_dists))

            def subtle_color(x):
                return mcolors.to_hex(cmap(x))

            bezier_colors_per_pair, bezier_stroke_widths_per_pair = (
                compute_bezier_colors_and_widths(
                    per_pair_taxa_dists, norm, subtle_color, min_width, max_width
                )
            )
    else:
        bezier_colors_per_pair = bezier_colors
        bezier_stroke_widths_per_pair = bezier_stroke_widths

    # Generate tree visualization
    svg_element = plot_circular_trees_in_a_row(
        trees,
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
        font_family=font_family,
        font_size=font_size,
        leaf_font_size=leaf_font_size,
        stroke_color=stroke_color,
        bezier_colors=bezier_colors_per_pair,
        bezier_stroke_widths=bezier_stroke_widths_per_pair,
        highlight_branches=params.highlight_branches,
        highlight_width=params.highlight_width[0] if params.highlight_width else 4.0,
        highlight_colors=params.highlight_colors,
        show_zero_length_indicators=show_zero_length_indicators,
        zero_length_indicator_color=zero_length_indicator_color,
        zero_length_indicator_size=zero_length_indicator_size,
    )

    if gridlines:
        # Determine SVG canvas dimensions for proper gridlines
        def _parse_dim(val: Optional[str], fallback: int) -> int:
            if not val:
                return fallback
            try:
                # Strip potential 'px' suffix and cast
                return int(str(val).replace("px", "").strip())
            except Exception:
                return fallback

        width_attr = svg_element.get("width")
        height_attr = svg_element.get("height")
        total_width = _parse_dim(width_attr, fallback=len(trees) * size)
        total_height = _parse_dim(height_attr, fallback=size)
        add_svg_gridlines(svg_element, total_width, total_height)

    # Only generate distance plots if show_distances=True
    if show_distances:
        # Generate distance plot
        rf_dists = compute_rf_distances(trees)
        circ_dists = [
            sum(pair) / len(pair) if pair else 0 for pair in per_pair_taxa_dists
        ]

        fig, ax = plt.subplots(figsize=(len(trees) * 1.5, 3))
        _plot_distances(ax, rf_dists, circ_dists)

        # Handle output with distance plots - always handle the figure when created
        svg_string = ET.tostring(svg_element, encoding="unicode")
        _handle_output(svg_string, fig, output_path, save_format, show_plot)
    else:
        # Handle output without distance plots - just save/display SVG
        if output_path:
            svg_string = ET.tostring(svg_element, encoding="unicode")
            if save_format.lower() == "png":
                import tempfile

                with tempfile.NamedTemporaryFile(
                    suffix=".svg", delete=False
                ) as tmp_svg:
                    tmp_svg.write(svg_string.encode())
                    tmp_svg.flush()

                    cairosvg.svg2png(
                        url=tmp_svg.name, write_to=output_path, background_color="white"
                    )
                    os.unlink(tmp_svg.name)
            else:
                with open(output_path, "w") as f:
                    f.write(svg_string)

        # If show_plot=True but show_distances=False, just display the tree SVG
        if show_plot:
            if output_path and os.path.exists(output_path):
                display(Image(filename=output_path))

    return svg_element


def _plot_distances(ax: Any, rf_dists: List[float], circ_dists: List[float]) -> None:
    """Create distance comparison plot."""
    import numpy as np  # noqa: WPS433
    x_positions = np.arange(len(rf_dists))
    x_labels = [f"Tree {i}→Tree {i + 1}" for i in range(len(rf_dists))]

    # Normalize distances for comparison
    rf_norm = (
        np.array(rf_dists) / max(rf_dists) if max(rf_dists) > 0 else np.array(rf_dists)
    )
    circ_norm = (
        np.array(circ_dists) / max(circ_dists)
        if max(circ_dists) > 0
        else np.array(circ_dists)
    )

    ax.plot(x_positions, rf_norm, "o-", label="Robinson-Foulds", markersize=8)
    ax.plot(x_positions, circ_norm, "s-", label="Circular Distance", markersize=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Normalized Distance")
    ax.set_title("Tree Distance Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _handle_output(
    svg_string: str,
    fig: Any,
    output_path: Optional[str],
    save_format: str,
    show_plot: bool,
) -> None:
    """Handle saving and/or displaying the combined visualization."""
    # Only import what we need based on output format
    
    if output_path:
        if save_format.lower() == "png":
            # Create a combined image with both trees (SVG) and distance plot (matplotlib)

            # First, convert SVG to image
            _require_optional_deps("PNG output", modules=["cairosvg", "PIL"])
            import cairosvg  # type: ignore  # noqa: WPS433
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
                tmp_svg.write(svg_string.encode())
                tmp_svg.flush()

                # Convert SVG to PNG in memory
                svg_png_data = cairosvg.svg2png(url=tmp_svg.name, background_color="white")
                os.unlink(tmp_svg.name)

            # Save matplotlib figure to memory
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mpl:
                fig.savefig(tmp_mpl.name, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
                mpl_png_path = tmp_mpl.name

            # Combine the images vertically using PIL
            from PIL import Image as PILImage  # noqa: WPS433
            import io  # noqa: WPS433

            # Load images
            svg_image = PILImage.open(io.BytesIO(svg_png_data))
            mpl_image = PILImage.open(mpl_png_path)

            # Calculate combined dimensions
            total_width = max(svg_image.width, mpl_image.width)
            total_height = svg_image.height + mpl_image.height + 20  # 20px padding

            # Create combined image
            combined = PILImage.new("RGB", (total_width, total_height), "white")

            # Paste SVG at top (centered if needed)
            svg_x = (total_width - svg_image.width) // 2
            combined.paste(svg_image, (svg_x, 0))

            # Paste matplotlib plot at bottom (centered if needed)
            mpl_x = (total_width - mpl_image.width) // 2
            combined.paste(mpl_image, (mpl_x, svg_image.height + 20))

            # Save combined image
            combined.save(output_path, "PNG", quality=95)

            # Clean up
            os.unlink(mpl_png_path)

        else:
            # For non-PNG formats, just save the SVG
            with open(output_path, "w") as f:
                f.write(svg_string)

    # Only display the plot in the notebook if show_plot=True
    if show_plot:
        try:
            from IPython.display import display  # noqa: WPS433
            display(fig)
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt  # noqa: WPS433
            plt.show()
        except Exception:
            pass

    # Always close the figure to prevent memory leaks, regardless of show_plot
    plt.close(fig)


# (Removed legacy alias plot_tree_sequence; use plot_tree_row_with_beziers_and_distances directly.)
