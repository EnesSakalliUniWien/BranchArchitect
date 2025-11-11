"""
Interactive tree sequence viewers for Jupyter notebooks.

This module provides interactive widgets for navigating and visualizing tree sequences
using the plot_tree_row_with_beziers_and_distances function from the circular_bezier_trees module.
"""

import os
import traceback
from typing import List, Optional

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, Image
    import matplotlib.pyplot as plt

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None

from brancharchitect.tree import Node
from brancharchitect.plot.circular_bezier_trees import (
    plot_tree_row_with_beziers_and_distances,
)


class InteractiveTreeViewer:
    """
    A basic interactive tree sequence viewer with navigation controls.

    Features:
    - Previous/Next navigation
    - Jump to specific tree with slider
    - Batch viewing (show multiple trees at once)
    - Uses plot_tree_row_with_beziers_and_distances for rendering
    """

    def __init__(
        self, trees: List[Node], tree_names: Optional[List[str]] = None, **kwargs
    ):
        """Initialize the interactive tree viewer."""
        if not WIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is required for interactive viewers. Install with: pip install ipywidgets"
            )

        self.trees = trees
        self.tree_names = tree_names or [f"Tree_{i}" for i in range(len(trees))]
        self.plot_kwargs = kwargs

        # Pop viewer-specific arguments from plot_kwargs to avoid conflicts
        self.output_dir = self.plot_kwargs.pop("output_dir", "./")
        self.size = self.plot_kwargs.pop("size", 600)
        self.cmap_name = self.plot_kwargs.pop("cmap_name", "tab10")
        self.leaf_font_size = self.plot_kwargs.pop("leaf_font_size", "42")
        self.show_zero_length_indicators = self.plot_kwargs.pop(
            "show_zero_length_indicators", False
        )
        self.zero_length_indicator_color = self.plot_kwargs.pop(
            "zero_length_indicator_color", "#ff4444"
        )
        self.zero_length_indicator_size = self.plot_kwargs.pop(
            "zero_length_indicator_size", 6.0
        )
        self.show_distances = self.plot_kwargs.pop("show_distances", False)

        self.current_index = 0
        self.batch_size = min(5, len(trees))
        self.create_widgets()
        self.update_display()

    def create_widgets(self):
        """Create and configure the UI widgets."""
        self.prev_button = widgets.Button(
            description="◀ Previous",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        self.next_button = widgets.Button(
            description="Next ▶",
            button_style="info",
            layout=widgets.Layout(width="100px"),
        )
        self.tree_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.trees) - 1,
            description="Tree:",
            continuous_update=False,
            layout=widgets.Layout(width="400px"),
        )
        self.batch_slider = widgets.IntSlider(
            value=min(5, len(self.trees)),
            min=1,
            max=min(10, len(self.trees)),
            description="Batch:",
            continuous_update=False,
            layout=widgets.Layout(width="300px"),
        )
        self.zero_indicators_toggle = widgets.Checkbox(
            value=self.show_zero_length_indicators,
            description="Show zero-length indicators",
            layout=widgets.Layout(width="200px"),
        )
        self.distances_toggle = widgets.Checkbox(
            value=self.show_distances,
            description="Show distance plots",
            layout=widgets.Layout(width="150px"),
        )
        self.info_label = widgets.HTML(value="")
        self.output_area = widgets.Output()

        self.prev_button.on_click(self.prev_clicked)
        self.next_button.on_click(self.next_clicked)
        self.tree_slider.observe(self.slider_changed, names="value")
        self.batch_slider.observe(self.batch_changed, names="value")
        self.zero_indicators_toggle.observe(self.zero_indicators_changed, names="value")
        self.distances_toggle.observe(self.distances_changed, names="value")

        nav_box = widgets.HBox(
            [self.prev_button, self.next_button, self.tree_slider, self.batch_slider]
        )
        options_box = widgets.HBox([self.zero_indicators_toggle, self.distances_toggle])
        self.control_box = widgets.VBox([nav_box, options_box, self.info_label])
        self.main_box = widgets.VBox([self.control_box, self.output_area])

    def prev_clicked(self, _):
        if self.current_index > 0:
            self.current_index = max(0, self.current_index - self.batch_size)
            self.update_display()

    def next_clicked(self, _):
        if self.current_index < len(self.trees) - 1:
            self.current_index = min(
                len(self.trees) - 1, self.current_index + self.batch_size
            )
            self.update_display()

    def slider_changed(self, change):
        self.current_index = change["new"]
        self.update_display()

    def batch_changed(self, change):
        self.batch_size = change["new"]
        self.update_display()

    def zero_indicators_changed(self, change):
        self.show_zero_length_indicators = change["new"]
        self.update_display()

    def distances_changed(self, change):
        self.show_distances = change["new"]
        self.update_display()

    def update_display(self):
        """Update the display with current trees."""
        with self.tree_slider.hold_sync():
            self.tree_slider.value = self.current_index

        end_index = min(self.current_index + self.batch_size, len(self.trees))
        trees_to_plot = self.trees[self.current_index : end_index]

        if self.batch_size == 1:
            info_text = f"<b>Tree {self.current_index + 1} of {len(self.trees)}</b><br>"
            info_text += f"Name: {self.tree_names[self.current_index]}"
        else:
            info_text = f"<b>Trees {self.current_index + 1}-{end_index} of {len(self.trees)}</b><br>"
            info_text += f"Batch size: {len(trees_to_plot)}"
        self.info_label.value = info_text

        self.prev_button.disabled = self.current_index == 0
        self.next_button.disabled = self.current_index >= len(self.trees) - 1

        with self.output_area:
            clear_output(wait=True)
            plt.close("all")
            try:
                filename = "interactive_tree_batch.png"
                filepath = os.path.join(self.output_dir, filename)

                show_distances_flag = self.show_distances and len(trees_to_plot) >= 2

                plot_tree_row_with_beziers_and_distances(
                    trees_to_plot,
                    size=self.size,
                    cmap_name=self.cmap_name,
                    output_path=filepath,
                    leaf_font_size=self.leaf_font_size,
                    show_zero_length_indicators=self.show_zero_length_indicators,
                    zero_length_indicator_color=self.zero_length_indicator_color,
                    zero_length_indicator_size=self.zero_length_indicator_size,
                    save_format="png",
                    show_plot=False,
                    show_distances=show_distances_flag,
                    bezier_stroke_widths=0.1,
                    **self.plot_kwargs,
                )

                if os.path.exists(filepath):
                    display(Image(filename=filepath))
                    if self.batch_size == 1:
                        print(f"Newick: {trees_to_plot[0].to_newick()}")
                    else:
                        print(f"Showing {len(trees_to_plot)} trees in batch.")
                        for i, tree in enumerate(trees_to_plot):
                            tree_idx = self.current_index + i
                            tree_name = self.tree_names[tree_idx]
                            print(f"- {tree_name}: {tree.to_newick()}")
                else:
                    print(f"Error: Image file not found at {filepath}")

            except Exception as e:
                print(f"Error plotting trees: {e}")
                traceback.print_exc()

    def display(self):
        """Display the viewer widget."""
        display(self.main_box)


class TreeSequenceComparisonViewer:
    """
    Side-by-side comparison viewer for two tree sequences.
    """

    def __init__(
        self,
        sequence1: List[Node],
        sequence2: List[Node],
        names1: Optional[List[str]] = None,
        names2: Optional[List[str]] = None,
        **plot_kwargs,
    ):
        if not WIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is required for interactive viewers. Install with: pip install ipywidgets"
            )

        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.names1 = names1 or [f"Seq1_Tree_{i}" for i in range(len(sequence1))]
        self.names2 = names2 or [f"Seq2_Tree_{i}" for i in range(len(sequence2))]
        self.plot_kwargs = plot_kwargs
        self.show_zero_length_indicators = self.plot_kwargs.pop(
            "show_zero_length_indicators", False
        )
        self.current_index = 0
        self.max_length = max(len(sequence1), len(sequence2))

        self.create_widgets()
        self.update_display()

    def create_widgets(self):
        """Create and configure the UI widgets."""
        self.prev_button = widgets.Button(description="◀ Previous", button_style="info")
        self.next_button = widgets.Button(description="Next ▶", button_style="info")
        self.position_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.max_length - 1,
            description="Position:",
            continuous_update=False,
            layout=widgets.Layout(width="500px"),
        )
        self.info_label = widgets.HTML(value="")

        self.prev_button.on_click(self.prev_clicked)
        self.next_button.on_click(self.next_clicked)
        self.position_slider.observe(self.position_changed, names="value")

        self.zero_indicators_toggle = widgets.Checkbox(
            value=self.show_zero_length_indicators,
            description="Show zero-length indicators",
            layout=widgets.Layout(width="200px"),
        )
        self.zero_indicators_toggle.observe(self.zero_indicators_changed, names="value")

        nav_box = widgets.HBox(
            [
                self.prev_button,
                self.next_button,
                self.position_slider,
                self.zero_indicators_toggle,
            ]
        )
        self.control_box = widgets.VBox([nav_box, self.info_label])

        self.output_area1 = widgets.Output()
        self.output_area2 = widgets.Output()
        comparison_box = widgets.HBox([self.output_area1, self.output_area2])
        self.main_box = widgets.VBox([self.control_box, comparison_box])

    def prev_clicked(self, _):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def next_clicked(self, _):
        if self.current_index < self.max_length - 1:
            self.current_index += 1
            self.update_display()

    def position_changed(self, change):
        self.current_index = change["new"]
        self.update_display()

    def zero_indicators_changed(self, change):
        self.show_zero_length_indicators = change["new"]
        self.update_display()

    def update_display(self):
        """Update the display with current trees."""
        self.position_slider.value = self.current_index

        info_text = f"<b>Position {self.current_index + 1} of {self.max_length}</b><br>"
        tree1 = (
            self.sequence1[self.current_index]
            if self.current_index < len(self.sequence1)
            else None
        )
        tree2 = (
            self.sequence2[self.current_index]
            if self.current_index < len(self.sequence2)
            else None
        )

        info_text += f"Sequence 1: {self.names1[self.current_index] if tree1 else '(no tree)'}<br>"
        info_text += (
            f"Sequence 2: {self.names2[self.current_index] if tree2 else '(no tree)'}"
        )
        self.info_label.value = info_text

        self.prev_button.disabled = self.current_index == 0
        self.next_button.disabled = self.current_index >= self.max_length - 1

        self.plot_tree(tree1, self.output_area1, "seq1_tree.png", "Sequence 1")
        self.plot_tree(tree2, self.output_area2, "seq2_tree.png", "Sequence 2")

    def plot_tree(self, tree: Optional[Node], output_area, filename: str, title: str):
        """Plot a single tree in the specified output area."""
        with output_area:
            clear_output(wait=True)
            if tree is not None:
                try:
                    # Ensure show_plot is False to avoid duplicate argument errors
                    self.plot_kwargs["show_plot"] = False
                    plot_tree_row_with_beziers_and_distances(
                        [tree],
                        output_path=filename,
                        save_format="png",
                        show_zero_length_indicators=self.show_zero_length_indicators,
                        **self.plot_kwargs,
                    )
                    if os.path.exists(filename):
                        display(Image(filename=filename))
                except Exception as e:
                    print(f"Error plotting {title}: {e}")
            else:
                print(f"{title}: No tree at this position")

    def display(self):
        """Display the comparison viewer widget."""
        display(self.main_box)


# Simplified factory functions
def create_interactive_viewer(
    trees: List[Node], tree_names: Optional[List[str]] = None, **kwargs
) -> "InteractiveTreeViewer":
    """Create and display a basic interactive tree viewer."""
    viewer = InteractiveTreeViewer(trees, tree_names, **kwargs)
    viewer.display()
    return viewer


def create_comparison_viewer(
    seq1: List[Node],
    seq2: List[Node],
    names1: Optional[List[str]] = None,
    names2: Optional[List[str]] = None,
    **kwargs,
) -> "TreeSequenceComparisonViewer":
    """Create and display a side-by-side comparison viewer for two tree sequences."""
    viewer = TreeSequenceComparisonViewer(seq1, seq2, names1, names2, **kwargs)
    viewer.display()
    return viewer
