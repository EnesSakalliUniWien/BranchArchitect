"""Join SatuTe's per-branch results back onto a BranchArchitect tree.

SatuTe reports a branch by its bipartition (the `Split` column: comma-joined
taxon names on the smaller side), not by any id BranchArchitect assigns — the
two tools ran the tree through completely different code paths, so ids can't
be trusted to line up. The bipartition is the only stable join key, and since
either endpoint's own `split_indices` could be the smaller *or* larger side of
a given branch depending on rooting, both this tree and SatuTe's rows are
canonicalised the same way (smaller-side, tie-broken by min()) before matching
— this makes the join independent of which side either tool happened to
report, and independent of SatuTe's own tie-break convention.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from brancharchitect.satute_bridge.stat_parser import SatuteBranchRow
from brancharchitect.tree import Node

IndexTuple = Tuple[int, ...]


def _normalize_indices(indices: Iterable[int]) -> IndexTuple:
    return tuple(sorted({int(index) for index in indices}))


def _canonical_split_key(
    indices: Iterable[int],
    all_taxa_indices: IndexTuple,
) -> IndexTuple:
    split = _normalize_indices(indices)
    if not split or not all_taxa_indices or split == all_taxa_indices:
        return ()

    split_set = set(split)
    complement = tuple(index for index in all_taxa_indices if index not in split_set)
    if not complement:
        return ()
    if len(split) < len(complement):
        return split
    if len(complement) < len(split):
        return complement
    return min(split, complement)


def _build_canonical_node_map(
    root: Node,
    all_taxa_indices: IndexTuple,
) -> Dict[IndexTuple, Node]:
    canonical_map: Dict[IndexTuple, Node] = {}
    for node in root.traverse():
        # SatuTe tests pendant (leaf) branches too, not just internal ones —
        # a leaf's own split_indices (its single taxon) is a valid branch key.
        if node.parent is None:
            continue
        key = _canonical_split_key(node.split_indices.indices, all_taxa_indices)
        if key:
            canonical_map[key] = node
    return canonical_map


def attach_saturation_annotations(
    root: Node,
    rows: List[SatuteBranchRow],
    set_default: bool = True,
) -> List[str]:
    """Attach SatuTe results onto matching nodes as flat `node.values` entries.

    These flow through the existing generic annotation pipeline
    (`build_branch_annotation_fields`'s metadata fallback) with zero changes to
    that shared module — each key becomes its own `role: "metadata"` field.

    SatuTe computes every branch under both the `dominant` and
    `eigenvalue_weighted` formulas, and the UI lets the user switch between
    them, so keys are namespaced by formula
    (``saturation_<formula>_<field>``). Without that namespacing a second call
    for the other formula silently overwrites the first.

    `set_default` additionally writes the un-namespaced ``saturation_<field>``
    keys, so exactly one formula is the one shown before the user picks.

    Returns the list of SatuTe `Split` values that could not be matched to any
    branch in this tree, so callers can surface a clear warning rather than
    silently dropping rows (e.g. a taxon-name mismatch between the tree and
    the SatuTe run's alignment).
    """
    all_taxa_indices = _normalize_indices(root.taxa_encoding.values())
    canonical_map = _build_canonical_node_map(root, all_taxa_indices)

    unmatched: List[str] = []
    for row in rows:
        taxon_names = [name for name in row.split.split(",") if name]
        try:
            row_indices = [root.taxa_encoding[name] for name in taxon_names]
        except KeyError:
            unmatched.append(row.split)
            continue

        key = _canonical_split_key(row_indices, all_taxa_indices)
        node = canonical_map.get(key) if key else None
        if node is None:
            unmatched.append(row.split)
            continue

        fields = {
            "p_value": row.p_value,
            "z_score": row.z_score,
            "decision": row.decision,
            "decision_fdr": row.decision_fdr,
            "formula": row.formula,
        }
        for name, value in fields.items():
            node.values[f"saturation_{row.formula}_{name}"] = value
            if set_default:
                node.values[f"saturation_{name}"] = value

    return unmatched
