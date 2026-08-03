"""Bridge to the SatuTe branch-saturation test (a mode of the iq-tree-satute fork's
iqtree3 binary), for the single-tree + single-alignment Information-Scope feature.

Distinct from msa_to_trees' iqtree3 usage: that module runs the bundled, stock
IQ-TREE binary to *infer* trees for sliding windows. SatuTe is a fork-specific
analysis mode that does not exist in the bundled binary, so it is resolved
separately (see `runner.get_satute_iqtree_exe`).
"""

from brancharchitect.satute_bridge.annotate import attach_saturation_annotations
from brancharchitect.satute_bridge.runner import SatuteRunResult, run_satute
from brancharchitect.satute_bridge.site_windows import (
    DEFAULT_STEP,
    DEFAULT_WINDOW_SIZE,
    compute_windows_for_branches,
    summarize_windows_across_branches,
)
from brancharchitect.satute_bridge.stat_parser import SatuteBranchRow, parse_sat_stat
from brancharchitect.satute_bridge.unrooting import to_unrooted_newick

__all__ = [
    "DEFAULT_STEP",
    "DEFAULT_WINDOW_SIZE",
    "SatuteBranchRow",
    "SatuteRunResult",
    "attach_saturation_annotations",
    "compute_windows_for_branches",
    "parse_sat_stat",
    "run_satute",
    "summarize_windows_across_branches",
    "to_unrooted_newick",
]
