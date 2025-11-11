import os
from typing import List, Dict, Tuple, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from brancharchitect.elements.partition_set import PartitionSet

# --- BranchArchitect Imports ---
# (Ensure these are correctly resolved in your environment)
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.plot.paper_plot.rectanlge_plot_configuration import (
    DEFAULT_STYLE_CONFIG,
)
from brancharchitect.plot.paper_plot.lego_rectangle_plot import (
    _apply_plot_style,
    _draw_group_background,
    _draw_atom_box,
    _draw_connecting_line,
)

# =============================================================================
# Deprecated/Alternative Functions (Kept for reference, might remove later)
# =============================================================================


def _prepare_flat_drawing_data(
    cover_sets: List[PartitionSet], style_config: Dict = DEFAULT_STYLE_CONFIG
) -> List[Dict[str, Any]]:
    """Prepares data for drawing flat partitions. Assigns group/atom indices."""
    # This is similar to _prepare_atom_drawing_data but might have subtle differences
    # if the original flat plot handled things differently. Review if needed.
    drawing_elements = []
    group_atom_counters: Dict[int, int] = {}
    for group_idx, partition_set in enumerate(cover_sets):
        if group_idx not in group_atom_counters:
            group_atom_counters[group_idx] = 0
        try:
            sorted_partitions = sorted(list(partition_set), key=lambda p: str(p))
        except TypeError:
            continue  # Skip non-iterable/sortable

        for partition in sorted_partitions:
            if not isinstance(partition, Partition):
                continue
            group_atom_counters[group_idx] += 1
            label_str = str(partition).strip("()")
            color_idx = group_idx % len(style_config["colors"])
            drawing_elements.append(
                {
                    "label": label_str,
                    "partition": partition,
                    "color": style_config["colors"][color_idx],
                    "group": group_idx,
                    "atom_index": group_atom_counters[group_idx],
                }
            )
    return drawing_elements


def _draw_flat_group_summary(
    ax: plt.Axes,
    bounds: Dict,
    y: float,
    group_idx: int,
    color_info: Dict,
    config: Dict,
    unique_cover_label: Optional[str] = None,
):
    """Draws the summary box for the flat partition visualization."""
    # This function might have specific layout needs for the flat plot.
    summary_y = y + config["box_height"] + config["summary_y_offset"]
    # Use summary_box_padding_x_factor if defined, else fallback
    padding_factor = config.get("summary_box_padding_x_factor", 0.8)
    summary_padding_x = config["group_bg_padding_x"] * padding_factor
    summary_width = (bounds["end"] - bounds["start"]) + (2 * summary_padding_x)
    summary_x = bounds["start"] - summary_padding_x
    summary_height = config.get("summary_box_height", 0.2)  # Use default if missing

    summary_rect = patches.FancyBboxPatch(
        (summary_x, summary_y),
        summary_width,
        summary_height,
        boxstyle=f"round,pad=0.1,rounding_size={config.get('summary_box_corner_radius', 0.1)}",
        facecolor=color_info["fill"],
        edgecolor=color_info["stroke"],
        linewidth=config.get("summary_box_linewidth", 1.5),
        alpha=config.get("summary_box_alpha", 0.9),
        zorder=12,
    )
    ax.add_patch(summary_rect)

    summary_label_text = (
        unique_cover_label.strip("()")
        if unique_cover_label
        else f"Group {group_idx + 1}"
    )
    label_fontsize = config["group_summary_fontsize"]
    if unique_cover_label and len(summary_label_text) > 20:
        label_fontsize -= 1

    center_x = summary_x + summary_width / 2
    ax.text(
        center_x,
        summary_y + summary_height / 2,
        summary_label_text,
        ha="center",
        va="center",
        fontsize=label_fontsize,
        fontweight="medium",
        color=color_info["stroke"],
        zorder=13,
    )
    return summary_y + summary_height


def draw_flat_partition_panel(
    ax: plt.Axes,
    cover_sets: List[PartitionSet],
    unique_cover_sets: Optional[List[PartitionSet]] = None,
    style_config: Dict = DEFAULT_STYLE_CONFIG,
    draw_group_summary: bool = True,
) -> float:
    """Draws a single panel for the flat partition visualization."""
    # Uses _prepare_flat_drawing_data and _draw_flat_group_summary
    drawing_elements = _prepare_flat_drawing_data(cover_sets, style_config)
    if not drawing_elements:
        return style_config["y_position"]

    group_bounds = {}
    current_x = 0.5
    element_positions = []
    for i, element in enumerate(drawing_elements):
        group_idx = element["group"]
        if group_idx not in group_bounds:
            group_bounds[group_idx] = {"start": current_x, "end": None, "elements": []}
        group_bounds[group_idx]["end"] = current_x + style_config["box_width"]
        group_bounds[group_idx]["elements"].append(i)
        element_positions.append(current_x)
        current_x += style_config["box_width"] + style_config["element_spacing"]

    max_x = current_x - style_config["element_spacing"]
    panel_top_y = style_config["y_position"] + style_config["box_height"]

    # Draw Backgrounds
    for group_idx, bounds in sorted(group_bounds.items()):
        if bounds["start"] is not None and bounds["end"] is not None:
            color_idx = group_idx % len(style_config["colors"])
            _draw_group_background(
                ax,
                bounds,
                style_config["y_position"],
                style_config["colors"][color_idx],
                style_config,
            )

    # Draw Atoms and Lines
    for i, element in enumerate(drawing_elements):
        x_pos = element_positions[i]
        _draw_atom_box(ax, element, x_pos, style_config["y_position"], style_config)
        group_idx = element["group"]
        if i > 0 and drawing_elements[i - 1]["group"] == group_idx:
            # Basic check for adjacency based on calculated positions
            if (
                abs(
                    element_positions[i]
                    - element_positions[i - 1]
                    - (style_config["box_width"] + style_config["element_spacing"])
                )
                < 1e-6
            ):
                prev_x_pos = element_positions[i - 1]
                _draw_connecting_line(
                    ax,
                    prev_x_pos,
                    x_pos,
                    style_config["y_position"],
                    element["color"]["stroke"],
                    style_config,
                )

    # Draw Summaries
    if draw_group_summary:
        max_summary_y = panel_top_y
        for group_idx, bounds in sorted(group_bounds.items()):
            if bounds["start"] is not None and bounds["end"] is not None:
                unique_label_str: Optional[str] = None
                if unique_cover_sets and group_idx < len(unique_cover_sets):
                    current_unique_set = unique_cover_sets[group_idx]
                    if current_unique_set:
                        unique_label_str = str(current_unique_set)
                color_idx = group_idx % len(style_config["colors"])
                summary_top = _draw_flat_group_summary(
                    ax,
                    bounds,
                    style_config["y_position"],
                    group_idx,
                    style_config["colors"][color_idx],
                    style_config,
                    unique_cover_label=unique_label_str,
                )
                max_summary_y = max(max_summary_y, summary_top)
        panel_top_y = max_summary_y

    ax.set_xlim(0, max_x + 0.5)
    ax.set_ylim(0, panel_top_y + 0.3)
    return panel_top_y


def create_flat_partition_visualization(
    lattice_edge: PivotEdgeSubproblem,
    output_dir: Optional[str] = None,
    output_filename_base: str = "flat_partition_visualization",
    style_config: Dict = DEFAULT_STYLE_CONFIG,
    title: str = "Comparison of Common Atoms",
    draw_group_summary: bool = True,
    use_unique_cover_labels: bool = True,
) -> plt.Figure:
    """Creates the original two-panel flat partition visualization."""
    # --- Validation --- (Copied from previous version)
    if not isinstance(lattice_edge, PivotEdgeSubproblem):
        raise ValueError(
            "Invalid input: lattice_edge must be a PivotEdgeSubproblem object."
        )
    if not hasattr(lattice_edge, "t1_common_covers") or not hasattr(
        lattice_edge, "t2_common_covers"
    ):
        raise ValueError(
            "PivotEdgeSubproblem object missing 't1_common_covers' or 't2_common_covers'."
        )
    if use_unique_cover_labels and (
        not hasattr(lattice_edge, "t1_unique_covers")
        or not hasattr(lattice_edge, "t2_unique_covers")
    ):
        print(
            "Warning: PivotEdgeSubproblem object missing 't1_unique_covers' or 't2_unique_covers'. Defaulting to 'Group X' labels."
        )
        use_unique_cover_labels = False

    _apply_plot_style(style_config)
    left_cover_sets = lattice_edge.t1_common_covers
    right_cover_sets = lattice_edge.t2_common_covers
    left_unique_covers = (
        lattice_edge.t1_unique_covers if use_unique_cover_labels else None
    )
    right_unique_covers = (
        lattice_edge.t2_unique_covers if use_unique_cover_labels else None
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=style_config.get("fig_size", (16, 5)),
        dpi=style_config.get("dpi", 300),
    )
    fig.suptitle(
        title, fontsize=style_config["title_fontsize"], fontweight="bold", y=0.98
    )

    for ax in [ax1, ax2]:  # Basic styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.set_title(
        r"$T_1$ Common Atoms",
        fontsize=style_config["subtitle_fontsize"],
        fontweight="medium",
    )
    ax2.set_title(
        r"$T_2$ Common Atoms",
        fontsize=style_config["subtitle_fontsize"],
        fontweight="medium",
    )

    max_y1 = draw_flat_partition_panel(
        ax1,
        left_cover_sets,
        unique_cover_sets=left_unique_covers,
        style_config=style_config,
        draw_group_summary=draw_group_summary,
    )
    max_y2 = draw_flat_partition_panel(
        ax2,
        right_cover_sets,
        unique_cover_sets=right_unique_covers,
        style_config=style_config,
        draw_group_summary=draw_group_summary,
    )

    global_max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(bottom=0, top=global_max_y)
    ax2.set_ylim(bottom=0, top=global_max_y)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    if output_dir:  # Save figure
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f"{output_filename_base}.pdf")
        png_path = os.path.join(output_dir, f"{output_filename_base}.png")
        try:
            plt.savefig(pdf_path, bbox_inches="tight", dpi=style_config.get("dpi", 300))
            print(f"Saved PDF visualization to: {pdf_path}")
            plt.savefig(png_path, bbox_inches="tight", dpi=style_config.get("dpi", 300))
            print(f"Saved PNG preview to: {png_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    return fig
