# Deprecated Lattice Files

**Date:** October 21, 2025

## Overview

This folder contains deprecated implementations that are no longer used in the active codebase. These files are kept for historical reference and comparison purposes only.

---

## Files

### 1. `frontiers.py`

**Status:** ❌ Completely unused
**Replacement:** `child_frontiers_refactored.py` (in parent directory)

**Description:**
Old frontier utilities implementation that was superseded by the refactored child frontiers approach.

**Contains:**
- `bottom_or_none()` - Extract unique minimal element
- `downset_under()` - Compute downset under frontier
- `map_frontier_to_minimals_by_shared_set()` - Map frontiers to minimal elements
- `compute_frontier_map()` - Build nested frontier mappings

**Why Deprecated:**
- Not imported anywhere in the codebase
- Functionality replaced by more modular approach in `child_frontiers_refactored.py`
- Old implementation strategy that didn't align with final design

---

### 2. `compute_child_frontiers.py`

**Status:** ❌ Duplicate of active implementation
**Replacement:** `child_frontiers_refactored.py` (in parent directory)

**Description:**
Nearly identical implementation to `child_frontiers_refactored.py`. Only minor differences in empty case handling (13 lines different).

**Contains:**
- `compute_child_frontiers()` - Main function for computing per-child frontiers

**Why Deprecated:**
- Almost exact duplicate of the active `child_frontiers_refactored.py`
- Not imported anywhere
- Created during development/refactoring but superseded by final version

**Differences from Active Version:**
```python
# This version (deprecated): Returns empty dict for empty children_to_process
if not children_to_process:
    return {}

# Active version: Creates TopToBottom entries for shared splits
if not children_to_process:
    child_frontiers: dict[Partition, TopToBottom] = {}
    for partition in shared_splits:
        # Creates singleton entries...
```

---

### 3. `child_frontiers_original.py`

**Status:** ⚠️ Historical backup (contains known bugs)
**Replacement:** `child_frontiers_refactored.py` (in parent directory)

**Description:**
Original implementation before the self-covering frontiers fix. Explicitly kept as a backup for reference and comparison.

**File Header States:**
> "This version has a known issue where frontier splits not covered by any bottom are omitted from the bottom_to_frontiers mapping, which can cause incorrect jumping taxa solutions in some cases."

**Contains:**
- `compute_child_frontiers()` - Original buggy implementation

**Why Deprecated:**
- Contains known bugs that produce incorrect solutions
- Lacks self-covering frontier logic
- Explicitly marked in comments as buggy backup
- Kept only for historical comparison

**Known Issues:**
- Frontier splits not covered by any bottom are omitted
- Causes incorrect jumping taxa solutions in edge cases
- Missing self-covering logic for shared direct children

---

### 4. `edge_sorting_OLD_20251021.py`

**Status:** ❌ Redundant wrapper
**Replacement:** `edge_depth_ordering.py` (in parent directory)

**Description:**
Thin wrapper around `edge_depth_ordering.py` that provided no additional functionality. The file only re-exported `sort_lattice_edges_by_subset_hierarchy()` which is already available in the main module.

**Contains:**
- `sort_lattice_edges_by_subset_hierarchy()` - Wrapper function that simply called `compute_lattice_edge_depths()` from `edge_depth_ordering.py`

**Why Deprecated:**
- Redundant abstraction layer with no added value
- Only contained one function that duplicated code from `edge_depth_ordering.py`
- All functionality is directly available in `edge_depth_ordering.py`
- Simplified codebase by eliminating unnecessary indirection

**Previous Usage:**
- Was imported by `lattice_solver.py` → Updated to import from `edge_depth_ordering.py` instead

**Migration:**
```python
# BEFORE (using wrapper):
from brancharchitect.jumping_taxa.lattice.edge_sorting import (
    sort_lattice_edges_by_subset_hierarchy,
)

# AFTER (direct import):
from brancharchitect.jumping_taxa.lattice.edge_depth_ordering import (
    sort_lattice_edges_by_subset_hierarchy,
)
```

---

### 5. `lattice_solver_OLD_20251021.py`

**Status:** ❌ Duplicate of active implementation (renamed file)
**Replacement:** `pivot_edge_solver.py` (in parent directory)

**Description:**
Original file that was renamed to `pivot_edge_solver.py` during Phase 1 of the lattice module refactoring. This file contains nearly identical code but lacks the parsimony filtering optimization present in the active version.

**Contains:**
- `solve_lattice_edges()` - Process sub-lattices to find solutions
- `lattice_algorithm()` - Main entry point for jumping taxa detection

**Why Deprecated:**
- File was officially renamed to `pivot_edge_solver.py` in Phase 1 refactoring
- Contains outdated implementation (no parsimony filtering)
- Has inline `_popcount()` function definition inside loop (performance issue)
- Computes unused `rank_key` values (wasteful)
- Not imported anywhere in active codebase

**Key Differences from Active Version:**
- **Missing parsimony filtering**: Keeps ALL solutions per pivot edge, not just best-ranked
- **Inline function definition**: `_popcount()` redefined in every loop iteration
- **Manual rank computation**: Reimplements `compute_solution_rank_key()` inline
- **Different imports**: Uses old `lattice_solution.LatticeSolutions` instead of `solution_manager`

**Migration:**
```python
# OLD (deprecated):
from brancharchitect.jumping_taxa.lattice.lattice_solver import lattice_algorithm

# NEW (active):
from brancharchitect.jumping_taxa.lattice.pivot_edge_solver import lattice_algorithm
```

**Note:** One stale import exists in `notebooks/paper_plots/rotation_system.ipynb` but the notebook imports `iterate_lattice_algorithm` which doesn't exist in this file, so the import would fail anyway.

---

## Important Notes

⚠️ **DO NOT USE THESE FILES IN PRODUCTION CODE**

These implementations are:
- Not imported by any active code
- Not tested in the current test suite
- May contain bugs or outdated logic
- Kept only for reference

---

## Active Implementation

The current, correct implementation is:

**`child_frontiers_refactored.py`** (in parent directory)

This file contains:
- Fixed self-covering frontier logic
- Proper handling of shared direct children
- Correct bottom-to-frontier mappings
- Full test coverage

---

## Migration History

| Date         | Action                       | Files                                                                 |
| ------------ | ---------------------------- | --------------------------------------------------------------------- |
| Oct 21, 2025 | Moved duplicate solver file  | lattice_solver.py → lattice_solver_OLD_20251021.py                    |
| Oct 21, 2025 | Deprecated redundant wrapper | edge_sorting.py → edge_sorting_OLD_20251021.py                        |
| Oct 21, 2025 | Moved to deprecated/         | frontiers.py, compute_child_frontiers.py, child_frontiers_original.py |
| Oct 20, 2025 | Fixed self-covering bug      | child_frontiers_refactored.py created                                 |
| Earlier      | Original implementation      | child_frontiers_original.py                                           |

---

## If You Need to Reference These Files

These files can be useful for:
- Understanding the evolution of the algorithm
- Comparing old vs new implementation
- Debugging historical issues
- Academic/research documentation

**But remember:** Always use `child_frontiers_refactored.py` for actual computation!

---

## Related Documentation

- [Lattice Module Renaming](../../../../docs/LATTICE_MODULE_RENAMING.md)
- [Cleanup Candidates](../../../../docs/CLEANUP_CANDIDATES.md)
- [Build Pivot Lattices](../build_pivot_lattices.py) - Uses the active implementation

---

**Maintained by:** BranchArchitect Development Team
**Last Updated:** October 21, 2025
