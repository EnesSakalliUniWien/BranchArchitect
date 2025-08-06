"""
BranchArchitect plotting utilities.
"""

# Import main plotting functions
from .circular_bezier_trees import plot_tree_row_with_beziers_and_distances
from .circular_tree import plot_circular_tree


# Import interactive viewers (optional, requires ipywidgets)
try:
    from .interactive_viewers import (
        InteractiveTreeViewer,
        EnhancedTreeViewer,
        TreeSequenceComparisonViewer,
        create_interactive_viewer,
        create_enhanced_viewer,
        create_comparison_viewer,
    )

    __all__ = [
        "plot_tree_row_with_beziers_and_distances",
        "plot_circular_tree",
        "plot_rectangular_tree",
        "plot_tree",
        "InteractiveTreeViewer",
        "EnhancedTreeViewer",
        "TreeSequenceComparisonViewer",
        "create_interactive_viewer",
        "create_enhanced_viewer",
        "create_comparison_viewer",
    ]
except ImportError:
    # ipywidgets not available, skip interactive viewers
    __all__ = [
        "plot_tree_row_with_beziers_and_distances",
        "plot_circular_tree",
        "plot_rectangular_tree",
        "plot_tree",
    ]
