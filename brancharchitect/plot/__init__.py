"""
BranchArchitect plotting utilities (lazy import).

This package avoids importing heavy optional dependencies at package import time
to ensure test environments without Cairo/matplotlib can import the package.

Import concrete submodules directly, e.g.:
    from brancharchitect.plot.tree_plot import plot_rectangular_tree_pair
    from brancharchitect.plot.circular_tree import plot_circular_tree

Interactive viewers and Cairo-dependent modules are intentionally not imported here.
"""

# Expose submodule names for discoverability; functions are available via
# direct submodule imports to avoid eager loading.
__all__ = [
    "tree_plot",
    "circular_tree",
    "rectangular_tree",
]
