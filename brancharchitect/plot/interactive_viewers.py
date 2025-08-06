"""
Interactive tree sequence viewers for Jupyter notebooks.

This module provides interactive widgets for navigating and visualizing tree sequences
using the plot_tree_row_with_beziers_and_distances function from the circular_bezier_trees module.
"""

import os
import time
import threading
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

        # Store parameters with defaults
        self.trees = trees
        self.tree_names = tree_names or [f"Tree_{i}" for i in range(len(trees))]
        self.output_dir = kwargs.get("output_dir", "./")
        self.size = kwargs.get("size", 600)
        self.cmap_name = kwargs.get("cmap_name", "tab10")
        self.leaf_font_size = kwargs.get("leaf_font_size", "42")
        self.show_zero_length_indicators = kwargs.get(
            "show_zero_length_indicators", False
        )
        self.zero_length_indicator_color = kwargs.get(
            "zero_length_indicator_color", "#ff4444"
        )
        self.zero_length_indicator_size = kwargs.get("zero_length_indicator_size", 6.0)
        self.show_distances = kwargs.get(
            "show_distances", False
        )  # New option to show/hide distance plots
        self.plot_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "output_dir",
                "size",
                "cmap_name",
                "leaf_font_size",
                "show_zero_length_indicators",
                "zero_length_indicator_color",
                "zero_length_indicator_size",
                "show_distances",
            }
        }

        self.current_index = 0
        self.batch_size = min(5, len(trees))
        self.create_widgets()
        self.update_display()

    def create_widgets(self):
        """Create and configure the UI widgets."""
        # Create all widgets
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

        # Bind events
        self.prev_button.on_click(self.prev_clicked)
        self.next_button.on_click(self.next_clicked)
        self.tree_slider.observe(self.slider_changed, names="value")
        self.batch_slider.observe(self.batch_changed, names="value")
        self.zero_indicators_toggle.observe(self.zero_indicators_changed, names="value")
        self.distances_toggle.observe(self.distances_changed, names="value")

        # Layout
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
        # Update slider without triggering events
        with self.tree_slider.hold_sync():
            self.tree_slider.value = self.current_index

        # Get trees to display
        end_index = min(self.current_index + self.batch_size, len(self.trees))
        trees_to_plot = self.trees[self.current_index : end_index]

        # Update info
        if self.batch_size == 1:
            info_text = f"<b>Tree {self.current_index + 1} of {len(self.trees)}</b><br>"
            info_text += f"Name: {self.tree_names[self.current_index]}"
        else:
            info_text = f"<b>Trees {self.current_index + 1}-{end_index} of {len(self.trees)}</b><br>"
            info_text += f"Batch size: {len(trees_to_plot)}"

        self.info_label.value = info_text

        # Update button states
        self.prev_button.disabled = self.current_index == 0
        self.next_button.disabled = self.current_index >= len(self.trees) - 1

        # Plot trees
        with self.output_area:
            clear_output(wait=True)
            plt.close("all")
            try:
                # Create a fixed filename for the plot to avoid creating multiple files
                filename = "interactive_tree_batch.png"
                pdf_path = os.path.join(self.output_dir, filename)

                print(
                    f"Debug: Plotting {len(trees_to_plot)} trees, show_distances={self.show_distances}"
                )

                # Always save to file and display from file - this ensures consistency
                # Handle distance display option
                if self.show_distances and len(trees_to_plot) >= 2:
                    # Include distance plots with trees
                    print("Debug: Plotting with distances...")
                    plot_tree_row_with_beziers_and_distances(
                        trees_to_plot,
                        size=self.size,
                        cmap_name=self.cmap_name,
                        output_path=pdf_path,
                        leaf_font_size=self.leaf_font_size,
                        show_zero_length_indicators=self.show_zero_length_indicators,
                        zero_length_indicator_color=self.zero_length_indicator_color,
                        zero_length_indicator_size=self.zero_length_indicator_size,
                        save_format="png",
                        show_plot=False,  # Always False - we handle display ourselves
                        show_distances=True,  # Enable distance plots for the interactive viewer
                        **self.plot_kwargs,
                    )
                else:
                    # Just trees without distance plots
                    print(
                        f"Debug: Plotting {len(trees_to_plot)} trees without distances..."
                    )
                    # For single trees or when distances are disabled, call with show_distances=False
                    plot_tree_row_with_beziers_and_distances(
                        trees_to_plot,
                        size=self.size,
                        cmap_name=self.cmap_name,
                        output_path=pdf_path,
                        leaf_font_size=self.leaf_font_size,
                        show_zero_length_indicators=self.show_zero_length_indicators,
                        zero_length_indicator_color=self.zero_length_indicator_color,
                        zero_length_indicator_size=self.zero_length_indicator_size,
                        save_format="png",
                        show_plot=False,  # Always False - we handle display ourselves
                        show_distances=False,  # Disable distance plots
                        **self.plot_kwargs,
                    )

                # Display the generated image file
                if os.path.exists(pdf_path):
                    print(f"Debug: Displaying image from {pdf_path}")
                    display(Image(filename=pdf_path))

                    # Show additional info
                    if self.batch_size == 1:
                        current_tree = trees_to_plot[0]
                        print(f"Newick: {current_tree.to_newick()}")
                    else:
                        print(f"Showing {len(trees_to_plot)} trees in batch")
                        for i, tree in enumerate(trees_to_plot):
                            tree_idx = self.current_index + i
                            tree_name = (
                                self.tree_names[tree_idx]
                                if tree_idx < len(self.tree_names)
                                else f"Tree_{tree_idx}"
                            )
                            print(f"{tree_name}: {tree.to_newick()}")

                else:
                    print(f"Error: Image file not found at {pdf_path}")

            except Exception as e:
                print(f"Error plotting trees: {e}")
                import traceback

                traceback.print_exc()

    def display(self):
        """Display the viewer widget."""
        display(self.main_box)


class EnhancedTreeViewer(InteractiveTreeViewer):
    """Enhanced interactive tree viewer with animation capabilities."""

    def __init__(self, trees, tree_names=None, **kwargs):
        super().__init__(trees, tree_names, **kwargs)
        self.is_playing = False
        self.play_speed = 1.0
        self.play_thread = None

    def create_widgets(self):
        """Create and configure the UI widgets with animation controls."""
        super().create_widgets()

        # Add animation controls
        self.play_button = widgets.Button(
            description="▶ Play",
            button_style="success",
            layout=widgets.Layout(width="80px"),
        )
        self.pause_button = widgets.Button(
            description="⏸ Pause",
            button_style="warning",
            layout=widgets.Layout(width="80px"),
        )
        self.stop_button = widgets.Button(
            description="⏹ Stop",
            button_style="danger",
            layout=widgets.Layout(width="80px"),
        )
        self.speed_slider = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=5.0,
            step=0.1,
            description="Speed:",
            continuous_update=False,
            layout=widgets.Layout(width="300px"),
        )
        self.first_button = widgets.Button(
            description="⏮ First", layout=widgets.Layout(width="80px")
        )
        self.last_button = widgets.Button(
            description="⏭ Last", layout=widgets.Layout(width="80px")
        )

        # Bind events
        self.play_button.on_click(self.play_clicked)
        self.pause_button.on_click(self.pause_clicked)
        self.stop_button.on_click(self.stop_clicked)
        self.first_button.on_click(self.first_clicked)
        self.last_button.on_click(self.last_clicked)
        self.speed_slider.observe(self.speed_changed, names="value")

        # Enhanced layout
        nav_box = widgets.HBox(
            [
                self.first_button,
                self.prev_button,
                self.play_button,
                self.pause_button,
                self.stop_button,
                self.next_button,
                self.last_button,
            ]
        )
        control_box = widgets.HBox(
            [self.tree_slider, self.batch_slider, self.speed_slider]
        )
        self.control_box = widgets.VBox([nav_box, control_box, self.info_label])
        self.main_box = widgets.VBox([self.control_box, self.output_area])

    def play_clicked(self, _):
        if not self.is_playing:
            self.is_playing = True
            self.play_thread = threading.Thread(target=self.animate)
            self.play_thread.start()

    def pause_clicked(self, _):
        self.is_playing = False

    def stop_clicked(self, _):
        self.is_playing = False
        self.current_index = 0
        self.update_display()

    def first_clicked(self, _):
        self.current_index = 0
        self.update_display()

    def last_clicked(self, _):
        self.current_index = len(self.trees) - 1
        self.update_display()

    def speed_changed(self, change):
        self.play_speed = 1.0 / change["new"]

    def animate(self):
        """Animation loop for automatic playback."""
        while self.is_playing and self.current_index < len(self.trees) - 1:
            time.sleep(self.play_speed)
            if self.is_playing:  # Check again in case it was paused
                self.current_index += 1
                self.update_display()
        self.is_playing = False


class TreeSequenceComparisonViewer:
    """
    Side-by-side comparison viewer for two tree sequences.

    Features:
    - Synchronized navigation through two sequences
    - Handles sequences of different lengths
    - Side-by-side display
    """

    def __init__(
        self,
        sequence1: List[Node],
        sequence2: List[Node],
        names1: Optional[List[str]] = None,
        names2: Optional[List[str]] = None,
        **plot_kwargs,
    ):
        """
        Initialize the comparison viewer.

        Args:
            sequence1: First tree sequence
            sequence2: Second tree sequence
            names1: Optional names for first sequence
            names2: Optional names for second sequence
            **plot_kwargs: Arguments passed to plot_tree_row_with_beziers_and_distances
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is required for interactive viewers. Install with: pip install ipywidgets"
            )

        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.names1 = names1 or [f"Seq1_Tree_{i}" for i in range(len(sequence1))]
        self.names2 = names2 or [f"Seq2_Tree_{i}" for i in range(len(sequence2))]
        self.show_zero_length_indicators = plot_kwargs.get(
            "show_zero_length_indicators", False
        )
        self.plot_kwargs = plot_kwargs
        self.current_index = 0
        self.max_length = max(len(sequence1), len(sequence2))

        self.create_widgets()
        self.update_display()

    def create_widgets(self):
        """Create and configure the UI widgets."""
        # Navigation controls
        self.prev_button = widgets.Button(description="◀ Previous", button_style="info")
        self.next_button = widgets.Button(description="Next ▶", button_style="info")

        # Position slider
        self.position_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.max_length - 1,
            description="Position:",
            continuous_update=False,
            layout=widgets.Layout(width="500px"),
        )

        # Info display
        self.info_label = widgets.HTML(value="")

        # Bind events
        self.prev_button.on_click(self.prev_clicked)
        self.next_button.on_click(self.next_clicked)
        self.position_slider.observe(self.position_changed, names="value")

        # Add the zero-length indicators toggle
        self.zero_indicators_toggle = widgets.Checkbox(
            value=self.show_zero_length_indicators,
            description="Show zero-length indicators",
            layout=widgets.Layout(width="200px"),
        )
        self.zero_indicators_toggle.observe(self.zero_indicators_changed, names="value")

        # Layout
        nav_box = widgets.HBox(
            [
                self.prev_button,
                self.next_button,
                self.position_slider,
                self.zero_indicators_toggle,
            ]
        )
        self.control_box = widgets.VBox([nav_box, self.info_label])

        # Output areas
        self.output_area1 = widgets.Output()
        self.output_area2 = widgets.Output()

        # Side-by-side layout
        comparison_box = widgets.HBox([self.output_area1, self.output_area2])
        self.main_box = widgets.VBox([self.control_box, comparison_box])

    def prev_clicked(self, button):
        """Handle previous button click."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def next_clicked(self, button):
        """Handle next button click."""
        if self.current_index < self.max_length - 1:
            self.current_index += 1
            self.update_display()

    def position_changed(self, change):
        """Handle position slider change."""
        self.current_index = change["new"]
        self.update_display()

    def zero_indicators_changed(self, change):
        """Handle zero-length indicators toggle change."""
        self.show_zero_length_indicators = change["new"]
        self.update_display()

    def update_display(self):
        """Update the display with current trees."""
        self.position_slider.value = self.current_index

        # Update info
        info_text = f"<b>Position {self.current_index + 1} of {self.max_length}</b><br>"

        # Get current trees
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

        if tree1:
            info_text += f"Sequence 1: {self.names1[self.current_index]}<br>"
        else:
            info_text += "Sequence 1: (no tree)<br>"

        if tree2:
            info_text += f"Sequence 2: {self.names2[self.current_index]}"
        else:
            info_text += "Sequence 2: (no tree)"

        self.info_label.value = info_text

        # Update button states
        self.prev_button.disabled = self.current_index == 0
        self.next_button.disabled = self.current_index >= self.max_length - 1

        # Plot trees
        self.plot_tree(
            tree1,
            self.output_area1,
            "seq1_tree.png",
            "Sequence 1",
        )
        self.plot_tree(
            tree2,
            self.output_area2,
            "seq2_tree.png",
            "Sequence 2",
        )

    def plot_tree(self, tree: Optional[Node], output_area, filename: str, title: str):
        """Plot a single tree in the specified output area."""
        with output_area:
            clear_output(wait=True)
            if tree is not None:
                try:
                    plot_tree_row_with_beziers_and_distances(
                        [tree],
                        output_path=filename,
                        save_format="png",
                        show_plot=False,
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
) -> InteractiveTreeViewer:
    """Create and display a basic interactive tree viewer."""
    viewer = InteractiveTreeViewer(trees, tree_names, **kwargs)
    viewer.display()
    return viewer


def create_enhanced_viewer(
    trees: List[Node], tree_names: Optional[List[str]] = None, **kwargs
) -> EnhancedTreeViewer:
    """Create and display an enhanced interactive tree viewer with animation."""
    viewer = EnhancedTreeViewer(trees, tree_names, **kwargs)
    viewer.display()
    return viewer


def create_comparison_viewer(
    seq1: List[Node],
    seq2: List[Node],
    names1: Optional[List[str]] = None,
    names2: Optional[List[str]] = None,
    **kwargs,
) -> TreeSequenceComparisonViewer:
    """Create and display a side-by-side comparison viewer for two tree sequences."""
    viewer = TreeSequenceComparisonViewer(seq1, seq2, names1, names2, **kwargs)
    viewer.display()
    return viewer
