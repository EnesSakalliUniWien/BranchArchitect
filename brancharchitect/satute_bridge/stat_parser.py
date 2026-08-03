"""Parser for SatuTe's <prefix>.sat.stat output.

Column order is fixed by satute_report_writer.cpp in the iq-tree-satute fork:
ID, Formula, RateCategory, RateMultiplier, RateSites, LeftTaxa, RightTaxa,
ValidSites, SkippedSites, satC, satVar, satSE, satZ, satP, Alpha,
AlphaTaxonBonf, Decision, DecisionTaxonBonf, FDR_BY, DecisionFDR, Label,
Length, EffectiveLength, InformationFraction, SaturationIndex, Split, Modes,
Eigenvalues, Weights.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

# The pooled, eigenvalue_weighted row is the one branch-level statistic surfaced
# to phylo-movies today; per-rate-category rows and the alternate "dominant"
# formula are parsed but filtered out by parse_sat_stat.
POOLED_RATE_CATEGORY = "pooled"
DEFAULT_FORMULA = "eigenvalue_weighted"


@dataclass
class SatuteBranchRow:
    branch_id: int
    formula: str
    decision: str
    decision_fdr: str
    z_score: float
    p_value: float
    variance: float
    length: float
    split: str  # comma-separated taxon names, smaller side of the bipartition


def parse_sat_stat(
    stat_file: str | Path,
    formula: str = DEFAULT_FORMULA,
) -> List[SatuteBranchRow]:
    """Parse a .sat.stat file, keeping only the pooled row per branch/formula."""
    rows: List[SatuteBranchRow] = []
    with open(stat_file, "r", encoding="utf-8", newline="") as handle:
        # The file opens with a '#'-prefixed comment block documenting each
        # column before the actual tab-separated header row.
        data_lines = (line for line in handle if not line.startswith("#"))
        reader = csv.DictReader(data_lines, delimiter="\t")
        for record in reader:
            if record.get("RateCategory") != POOLED_RATE_CATEGORY:
                continue
            if record.get("Formula") != formula:
                continue
            rows.append(
                SatuteBranchRow(
                    branch_id=int(record["ID"]),
                    formula=record["Formula"],
                    decision=record["Decision"],
                    decision_fdr=record["DecisionFDR"],
                    z_score=float(record["satZ"]),
                    p_value=float(record["satP"]),
                    variance=float(record["satVar"]),
                    length=float(record["Length"]),
                    split=record["Split"],
                )
            )
    return rows
