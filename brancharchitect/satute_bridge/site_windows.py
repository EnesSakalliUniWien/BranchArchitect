"""Sliding-window saturation from SatuTe's per-site contributions.

The manuscript's Tree-of-Life result is a *windowed* finding: pooled over a
whole alignment a branch can read as informative while individual 36-site
windows are saturated. Re-running SatuTe per window is infeasible (one ToL run
is ~6 minutes), so windows are aggregated from the per-site contributions a
single run emits via ``--satute-sites``.

Per window the statistic is formed exactly as the branch-level one is in
``finalizeSatuTeResult`` (tree/satute/satute_statistics.cpp), just over the
window's sites instead of all of them::

    coherence = mean(site contributions in window)
    se        = sqrt(branch variance / window site count)
    z         = coherence / se
    p         = Q(z)                      # upper tail
    saturated = p > alpha                 # informative if p <= alpha

The branch-level per-site variance (``satVar``) is reused; it is a property of
the branch, not of the window.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Manuscript sliding-window parameters (SatuTe Tree-of-Life analyses).
DEFAULT_WINDOW_SIZE = 36
DEFAULT_STEP = 1


@dataclass
class SaturationWindow:
    start: int  # 1-based inclusive alignment column
    end: int  # 1-based inclusive alignment column
    coherence: float
    z_score: float
    p_value: float
    saturated: bool


@dataclass
class BranchWindows:
    branch_id: int
    formula: str
    split: str
    windows: List[SaturationWindow]

    @property
    def saturated_count(self) -> int:
        return sum(1 for w in self.windows if w.saturated)

    @property
    def saturated_fraction(self) -> float:
        return self.saturated_count / len(self.windows) if self.windows else 0.0


def _normal_sf(z: float) -> float:
    """Upper-tail standard normal probability, matching gsl_cdf_ugaussian_Q."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def parse_sat_site(
    site_file: str | Path,
    formula: str = "eigenvalue_weighted",
) -> Dict[int, List[float]]:
    """Read .sat.site into {branch_id: [site contributions by column]}.

    Sites are 1-based in the file and stored 0-indexed here. Splits are not
    carried in this file (they would repeat a long taxon list on every row);
    look the branch ID up in .sat.stat instead.
    """
    by_branch: Dict[int, Dict[int, float]] = defaultdict(dict)

    with open(site_file, "r", encoding="utf-8", newline="") as handle:
        data_lines = (line for line in handle if not line.startswith("#"))
        for record in csv.DictReader(data_lines, delimiter="\t"):
            if record.get("Formula") != formula:
                continue
            branch_id = int(record["ID"])
            by_branch[branch_id][int(record["Site"]) - 1] = float(record["SiteCoherence"])

    result: Dict[int, List[float]] = {}
    for branch_id, site_map in by_branch.items():
        length = max(site_map) + 1
        result[branch_id] = [site_map.get(index, 0.0) for index in range(length)]
    return result


def compute_branch_windows(
    site_values: List[float],
    variance: float,
    alpha: float = 0.05,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP,
) -> List[SaturationWindow]:
    """Slide a window over one branch's per-site contributions."""
    windows: List[SaturationWindow] = []
    n_sites = len(site_values)
    if n_sites < window_size or not variance > 0:
        return windows

    for start in range(0, n_sites - window_size + 1, step):
        chunk = site_values[start : start + window_size]
        # Sites with no contribution (skipped by SatuTe, or all-gap) carry 0
        # and are excluded rather than diluting the window mean.
        contributing = [value for value in chunk if value != 0.0]
        if not contributing:
            continue

        coherence = sum(contributing) / len(contributing)
        se = math.sqrt(variance / len(contributing))
        if not se > 0:
            continue
        z_score = coherence / se
        p_value = _normal_sf(z_score)
        windows.append(
            SaturationWindow(
                start=start + 1,
                end=start + window_size,
                coherence=coherence,
                z_score=z_score,
                p_value=p_value,
                saturated=p_value > alpha,
            )
        )
    return windows


def summarize_windows_across_branches(
    branch_windows: List[BranchWindows],
) -> List[Dict[str, float]]:
    """Collapse per-branch windows into one whole-alignment track.

    Each entry covers one window position and reports how many of the tested
    branches are saturated there. This is what makes the manuscript's point
    visible without picking a branch first: a branch can be decisively
    "informative" pooled over the whole alignment while a large share of its
    individual windows are saturated.
    """
    by_position: Dict[Tuple[int, int], List[bool]] = defaultdict(list)
    for entry in branch_windows:
        for window in entry.windows:
            by_position[(window.start, window.end)].append(window.saturated)

    track: List[Dict[str, float]] = []
    for (start, end), flags in sorted(by_position.items()):
        total = len(flags)
        saturated = sum(1 for flag in flags if flag)
        track.append(
            {
                "start": start,
                "end": end,
                "branches": total,
                "saturated_branches": saturated,
                "saturated_fraction": (saturated / total) if total else 0.0,
            }
        )
    return track


def compute_windows_for_branches(
    site_file: str | Path,
    variance_by_branch: Dict[int, float],
    alpha: float = 0.05,
    formula: str = "eigenvalue_weighted",
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP,
    split_by_branch: Dict[int, str] | None = None,
) -> List[BranchWindows]:
    """Windowed saturation for every branch present in the .sat.site file."""
    parsed = parse_sat_site(site_file, formula=formula)
    splits = split_by_branch or {}
    results: List[BranchWindows] = []
    for branch_id, site_values in sorted(parsed.items()):
        split = splits.get(branch_id, "")
        variance = variance_by_branch.get(branch_id)
        if variance is None:
            continue
        windows = compute_branch_windows(
            site_values,
            variance,
            alpha=alpha,
            window_size=window_size,
            step=step,
        )
        results.append(
            BranchWindows(
                branch_id=branch_id,
                formula=formula,
                split=split,
                windows=windows,
            )
        )
    return results
