# --- Section 1: Distance and Utility Functions ---
def compute_rf_distances(trees):
    from brancharchitect.distances import robinson_foulds_distance
    return [robinson_foulds_distance(trees[i], trees[i+1]) for i in range(len(trees)-1)]


def per_taxa_circular_distances(tree1, tree2):
    order1 = tree1.get_current_order()
    order2 = tree2.get_current_order()
    n = len(order1)
    idx1 = {name: i for i, name in enumerate(order1)}
    idx2 = {name: i for i, name in enumerate(order2)}
    dists = []
    for name in order1:
        diff = abs(idx1[name] - idx2[name])
        d = min(diff, n - diff) / (n // 2)
        dists.append(d)
    return dists


def compute_per_pair_taxa_dists(trees):
    num_pairs = len(trees) - 1
    return [per_taxa_circular_distances(trees[i], trees[i+1]) for i in range(num_pairs)]


def compute_bezier_colors_and_widths(per_pair_taxa_dists, norm, subtle_color, min_width, max_width):
    bezier_colors_per_pair = [
        [subtle_color(norm(d)) for d in pair] for pair in per_pair_taxa_dists
    ]
    bezier_stroke_widths_per_pair = [
        [min_width + (max_width - min_width) * norm(d) for d in pair] for pair in per_pair_taxa_dists
    ]
    return bezier_colors_per_pair, bezier_stroke_widths_per_pair


def _normalize_highlight_branches(highlight_branches, n):
    if highlight_branches is None or (isinstance(highlight_branches, list) and (len(highlight_branches) == 0 or not isinstance(highlight_branches[0], (list, tuple)))):
        return [highlight_branches] * n
    return highlight_branches


def _normalize_highlight_width(highlight_width, n):
    if isinstance(highlight_width, (int, float)):
        return [highlight_width] * n
    return highlight_width


# --- Section 2: SVG Construction and Manipulation ---
def _make_svg_element(
    trees, size, margin, bezier_colors, bezier_stroke_widths, label_offset, highlight_branches, highlight_width, highlight_colors=None,
    font_family='Monospace', font_size='12', stroke_color='#000', leaf_font_size=None
):
    from brancharchitect.plot.tree_plot import plot_circular_trees_in_a_row
    return plot_circular_trees_in_a_row(
        trees,
        size=size,
        margin=margin,
        bezier_colors=bezier_colors,
        bezier_stroke_widths=bezier_stroke_widths,
        label_offset=label_offset,
        highlight_branches=highlight_branches,
        highlight_width=highlight_width,
        highlight_colors=highlight_colors,
        font_family=font_family,
        font_size=font_size,
        stroke_color=stroke_color,
        leaf_font_size=leaf_font_size,
    )


def _get_svg_size(svg_element, size, n):
    width = int(svg_element.attrib.get("width", size * n))
    height = int(svg_element.attrib.get("height", size))
    return width, height


def _add_svg_background(svg_element, width, height, top_margin=38, bottom_margin=0):
    import xml.etree.ElementTree as ET
    background = ET.Element(
        "rect",
        {"x": "0", "y": "0", "width": str(width+100), "height": str(height+top_margin+bottom_margin+100), "fill": "#fff"},
    )
    svg_element.insert(0, background)


def _add_svg_legend(svg_element, width, height, legend_items, position="topmargin", top_margin=38):
    import xml.etree.ElementTree as ET
    legend_group = ET.Element("g", {"id": "svg_legend"})
    box_width = 320
    box_height = 28 * len(legend_items) + 28
    x0 = 18
    y0 = top_margin - box_height + 2
    box = ET.Element(
        "rect",
        {
            "x": str(x0),
            "y": str(y0),
            "width": str(box_width),
            "height": str(box_height),
            "fill": "#fff",
            "stroke": "#bbb",
            "stroke-width": "2.2",
            "rx": "10",
            "ry": "10",
            "opacity": "0.96",
        },
    )
    legend_group.append(box)
    for i, (color, label) in enumerate(legend_items):
        y = y0 + 28 + i * 28
        ET.SubElement(
            legend_group,
            "line",
            {
                "x1": str(x0 + 20),
                "y1": str(y),
                "x2": str(x0 + 60),
                "y2": str(y),
                "stroke": color,
                "stroke-width": "13",
                "stroke-linecap": "round",
            },
        )
        legend_text = ET.SubElement(
            legend_group,
            "text",
            {
                "x": str(x0 + 70),
                "y": str(y + 2),
                "font-size": "24",
                "font-family": "sans-serif",
                "fill": "#222",
                "font-weight": "bold",
                "dominant-baseline": "middle",
            },
        )
        legend_text.text = label
    svg_element.append(legend_group)


def _make_and_style_svg_element(
    trees, size, margin, bezier_colors_per_pair, bezier_stroke_widths_per_pair,
    label_offset, highlight_branches_list, highlight_width_list, highlight_colors_list, y_steps, top_margin=190,
    font_family='Monospace', font_size='12', stroke_color='#000', leaf_font_size=None
):
    svg_element = _make_svg_element(
        trees, size, margin, bezier_colors_per_pair, bezier_stroke_widths_per_pair,
        label_offset, highlight_branches_list, highlight_width_list, highlight_colors_list,
        font_family=font_family, font_size=font_size, stroke_color=stroke_color, leaf_font_size=leaf_font_size
    )
    width, height = _get_svg_size(svg_element, size, len(trees))
    _add_svg_background(svg_element, width, height, top_margin=top_margin, bottom_margin=0)
    from brancharchitect.plot.tree_plot import add_svg_gridlines, add_tree_labels, add_direction_arrow
    add_svg_gridlines(svg_element, width, height, y_steps=y_steps)
    # add_tree_labels(svg_element, len(trees), size=size, height=0, y_offset= height, font_size=32)
    # add_direction_arrow(svg_element, width, y=top_margin/8)
    return svg_element, width, height


# --- Section 3: Distance Plotting (Matplotlib) ---
def _make_distance_plot(trees, width, height, title=None, xlabel=None, ylabel=None, bottom_margin=0):
    import matplotlib.pyplot as plt
    import numpy as np
    rf_dists = compute_rf_distances(trees)
    circ_dists = [np.mean(pair) for pair in compute_per_pair_taxa_dists(trees)]
    circ_sums = [np.sum(pair) for pair in compute_per_pair_taxa_dists(trees)]
    total_circular_distance = sum(circ_sums)
    x = np.arange(0.5, len(trees)-0.5, 1)
    fig, axs = plt.subplots(2, 1, figsize=(width/80, 3.2 + bottom_margin/80), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.18})
    _plot_rf(axs[0], x, rf_dists, len(trees))
    _plot_circ(axs[1], x, circ_dists, len(trees))
    fig.subplots_adjust(bottom=0.22 + (bottom_margin/height if bottom_margin else 0))
    axs[1].annotate(
        f"Total circular distance: {total_circular_distance:.2f}",
        xy=(1, -0.32 - (bottom_margin/height if bottom_margin else 0)),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=22,
        color="tab:blue",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafd", ec="#b0c4de", lw=2, alpha=0.92),
    )
    if title:
        fig.suptitle(title, fontsize=28, fontweight="bold", y=1.03)
    if xlabel:
        axs[1].set_xlabel(xlabel, fontsize=22, fontweight="bold")
    if ylabel:
        axs[1].set_ylabel(ylabel, fontsize=20, color='tab:blue', labelpad=2, fontweight="bold")
    for ax in axs:
        ax.tick_params(axis='both', labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    for ax in axs:
        leg = ax.get_legend()
        if leg:
            for text in leg.get_texts():
                text.set_fontsize(18)
                text.set_fontweight('bold')
    plt.tight_layout(pad=0.22)
    return fig


def _plot_rf(ax, x, rf_dists, n):
    ax.plot(x, rf_dists, color='tab:red', lw=2.2, marker='o', markersize=9, label='Robinson-Foulds')
    ax.set_ylabel('RF', fontsize=18, color='tab:red', labelpad=2, fontweight="bold")
    ax.tick_params(axis='y', labelcolor='tab:red', length=0, labelsize=14)
    ax.set_xlim(0, n-1)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('tab:red')
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.22)
    ax.legend(loc='upper right', fontsize=18, frameon=False)


def _plot_circ(ax, x, circ_dists, n):
    import numpy as np
    ax.plot(x, circ_dists, color='tab:blue', lw=2.2, marker='o', markersize=9, label='Circular (mean)')
    ax.set_ylabel('Circular', fontsize=18, color='tab:blue', labelpad=2, fontweight="bold")
    ax.tick_params(axis='y', labelcolor='tab:blue', length=0, labelsize=14)
    ax.set_xlim(0, n-1)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f"T{i+1}" for i in range(n)], fontsize=16, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('tab:blue')
    ax.spines['bottom'].set_color('#888')
    ax.grid(axis='y', linestyle=':', alpha=0.22)
    ax.legend(loc='upper right', fontsize=18, frameon=False)


# --- Section 4: Image Combination and Output ---
def _combine_and_display(svg_string, fig, save_format, output_path=None):
    import tempfile
    import os
    from IPython.display import Image
    from IPython.display import display
    import cairosvg
    import PIL.Image
    if output_path is not None:
        tmpdir = os.path.dirname(output_path)
        os.makedirs(tmpdir, exist_ok=True)
        svg_path = os.path.join(tmpdir, "trees.svg")
        img_path = output_path
        dist_path = os.path.join(tmpdir, "dist.png")
        with open(svg_path, "w") as f:
            f.write(svg_string)
        if img_path.lower().endswith('.pdf'):
            cairosvg.svg2pdf(bytestring=svg_string.encode("utf-8"), write_to=img_path)
        else:
            cairosvg.svg2png(url=svg_path, write_to=os.path.join(tmpdir, "trees.png"))
            tree_img = PIL.Image.open(os.path.join(tmpdir, "trees.png"))
            fig.savefig(dist_path, bbox_inches='tight', dpi=100)
            dist_img = PIL.Image.open(dist_path)
            w = max(tree_img.width, dist_img.width)
            tree_img = tree_img.resize((w, tree_img.height), PIL.Image.LANCZOS)
            dist_img = dist_img.resize((w, dist_img.height), PIL.Image.LANCZOS)
            total_height = tree_img.height + dist_img.height
            combined = PIL.Image.new("RGBA", (w, total_height), (255,255,255,255))
            combined.paste(tree_img, (0,0))
            combined.paste(dist_img, (0, tree_img.height))
            combined.save(img_path)
            display(Image(filename=img_path))
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = os.path.join(tmpdir, "trees.svg")
            img_path = os.path.join(tmpdir, f"combined.{save_format}")
            with open(svg_path, "w") as f:
                f.write(svg_string)
            cairosvg.svg2png(url=svg_path, write_to=os.path.join(tmpdir, "trees.png"))
            tree_img = PIL.Image.open(os.path.join(tmpdir, "trees.png"))
            fig.savefig(os.path.join(tmpdir, "dist.png"), bbox_inches='tight', dpi=100)
            dist_img = PIL.Image.open(os.path.join(tmpdir, "dist.png"))
            w = max(tree_img.width, dist_img.width)
            tree_img = tree_img.resize((w, tree_img.height), PIL.Image.LANCZOS)
            dist_img = dist_img.resize((w, dist_img.height), PIL.Image.LANCZOS)
            total_height = tree_img.height + dist_img.height
            combined = PIL.Image.new("RGBA", (w, total_height), (255,255,255,255))
            combined.paste(tree_img, (0,0))
            combined.paste(dist_img, (0, tree_img.height))
            combined.save(img_path)
            display(Image(filename=img_path))


# --- Section 5: Main Plotting API ---
def plot_tree_row_with_beziers_and_distances(
    trees,
    size=240,
    margin=50,
    label_offset=18,
    y_steps=7,
    min_width=4.0,  # Thicker minimum Bezier width
    max_width=14.0, # Thicker maximum Bezier width
    cmap_name='viridis',
    save_format='png',
    highlight_branches=None,
    highlight_width=10.0,  # Thicker highlight by default
    highlight_colors=None,
    distance_plot_title=None,
    distance_plot_xlabel=None,
    distance_plot_ylabel=None,
    output_path=None,
    svg_output_path=None,
    ignore_branch_lengths=False,
    font_family='Monospace',
    font_size='18',
    leaf_font_size=None,  # Expose leaf_font_size for user control
    stroke_color='#000',
    bezier_colors=None,
    bezier_stroke_widths=None,
    glow=True,  # Enable glow for better separation
):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import xml.etree.ElementTree as ET
    import numpy as np
    n = len(trees)
    highlight_branches_list = _normalize_highlight_branches(highlight_branches, n)
    highlight_width_list = _normalize_highlight_width(highlight_width, n)

    def normalize_effects(effects):
        if effects is None:
            return [None] * n
        out = []
        for d in effects:
            if d is None:
                out.append({})
            elif isinstance(d, dict):
                newd = {}
                for k, v in d.items():
                    if isinstance(v, str):
                        newd[k] = {'highlight_color': v}
                    else:
                        newd[k] = v
                out.append(newd)
            else:
                out.append(d)
        return out

    highlight_colors_list = normalize_effects(highlight_colors) if highlight_colors is not None else [{} for _ in range(n)]
    per_pair_taxa_dists = compute_per_pair_taxa_dists(trees)
    all_dists = [d for pair in per_pair_taxa_dists for d in pair]
    min_d, max_d = min(all_dists), max(all_dists)

    def norm(d):
        return (d - min_d) / (max_d - min_d + 1e-8)

    cmap = plt.get_cmap(cmap_name)

    def subtle_color(val):
        return "#e0e0e0" if val < 1e-6 else mcolors.to_hex(cmap(val))

    bezier_colors_per_pair, bezier_stroke_widths_per_pair = compute_bezier_colors_and_widths(
        per_pair_taxa_dists, norm, subtle_color, min_width, max_width
    )
    bottom_margin = 36
    # Use leaf_font_size if provided, else fallback to font_size
    effective_leaf_font_size = str(leaf_font_size) if leaf_font_size is not None else font_size
    svg_element, width, height = _make_and_style_svg_element(
        trees, size, margin, bezier_colors_per_pair, bezier_stroke_widths_per_pair,
        label_offset, highlight_branches_list, highlight_width_list, highlight_colors_list, y_steps, top_margin=210,
        font_family=font_family, font_size=effective_leaf_font_size, stroke_color=stroke_color, leaf_font_size=effective_leaf_font_size
    )
    svg_string = ET.tostring(svg_element, encoding="unicode")
    if svg_output_path is not None:
        with open(svg_output_path, "w", encoding="utf-8") as f:
            f.write(svg_string)
    fig = _make_distance_plot(
        trees, width, height,
        title=distance_plot_title,
        xlabel=distance_plot_xlabel,
        ylabel=distance_plot_ylabel,
        bottom_margin=bottom_margin
    )
    _combine_and_display(svg_string, fig, save_format, output_path=output_path)
    plt.close(fig)