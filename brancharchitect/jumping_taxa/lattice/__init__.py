"""Lattice package.

Note: Avoid importing submodules at package import time to prevent import cycles
with debug/core components. Import submodules directly where needed, e.g.:

from brancharchitect.jumping_taxa.lattice.compute_pivot_solutions_with_deletions import compute_pivot_solutions_with_deletions
"""

__all__: list[str] = []
