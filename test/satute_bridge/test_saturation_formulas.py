"""Both SatuTe formulas must reach the frontend, not just the default one.

SatuTe computes `dominant` and `eigenvalue_weighted` for every branch. The
Information-Scope UI needs to let the user switch between them, which means
both sets of results have to be attached to the tree, distinguishable by
formula.
"""

from pathlib import Path

import pytest

from brancharchitect.io import parse_newick
from brancharchitect.satute_bridge import attach_saturation_annotations, parse_sat_stat

DATA = Path(__file__).parent / "data"
STAT_FILE = DATA / "two_formula.sat.stat"
NEWICK = "(A:0.1,B:0.1,(C:0.1,D:0.1):0.1);"


def test_parse_sat_stat_can_select_either_formula():
    weighted = parse_sat_stat(STAT_FILE, formula="eigenvalue_weighted")
    dominant = parse_sat_stat(STAT_FILE, formula="dominant")

    assert weighted, "expected eigenvalue_weighted rows"
    assert dominant, "expected dominant rows"
    assert {row.formula for row in weighted} == {"eigenvalue_weighted"}
    assert {row.formula for row in dominant} == {"dominant"}
    # Same branches, different statistics.
    assert {row.branch_id for row in weighted} == {row.branch_id for row in dominant}
    assert [row.z_score for row in weighted] != [row.z_score for row in dominant]


def test_both_formulas_attach_side_by_side():
    root = parse_newick(NEWICK)
    weighted = parse_sat_stat(STAT_FILE, formula="eigenvalue_weighted")
    dominant = parse_sat_stat(STAT_FILE, formula="dominant")

    attach_saturation_annotations(root, weighted)
    attach_saturation_annotations(root, dominant)

    annotated = [node for node in root.traverse() if node.values.get("saturation_decision")]
    assert annotated, "no node received saturation annotations"

    node = annotated[0]
    # Both formulas must survive; attaching the second must not overwrite the
    # first, or the UI can only ever show one of them.
    assert node.values.get("saturation_eigenvalue_weighted_decision") is not None
    assert node.values.get("saturation_dominant_decision") is not None


def test_decisions_other_than_informative_are_preserved():
    """`saturated` and `undefined_variance` must round-trip, not be dropped."""
    rows = parse_sat_stat(STAT_FILE, formula="dominant")
    decisions = {row.decision for row in rows}
    assert "saturated" in decisions
