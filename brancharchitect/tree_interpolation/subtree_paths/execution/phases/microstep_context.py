from __future__ import annotations

from dataclasses import dataclass

from brancharchitect.elements.partition import Partition

from ..grouping import get_group_for_mover
from ...planning import PivotTransitionStep


@dataclass(frozen=True, slots=True)
class SelectionPaths:
    """Normalized collapse and expand paths for one planner-selected mover."""

    subtree: Partition
    collapse_paths: list[Partition]
    expand_paths: list[Partition]

    @classmethod
    def from_transition_step(cls, step: PivotTransitionStep) -> "SelectionPaths":
        return cls(
            subtree=step.subtree,
            collapse_paths=list(step.collapse_path),
            expand_paths=list(step.expand_path),
        )

    @property
    def has_collapse_work(self) -> bool:
        return bool(self.collapse_paths)

    @property
    def has_expand_work(self) -> bool:
        return bool(self.expand_paths)


@dataclass(frozen=True, slots=True)
class PhaseHighlightGroups:
    """Phase-specific active mover groups emitted for renderer highlights."""

    collapse: list[Partition]
    reorder: list[Partition]
    expand: list[Partition]


def build_phase_highlight_groups(
    subtree: Partition,
    collapse_sibling_groups: dict[Partition, list[Partition]] | None,
    expand_sibling_groups: dict[Partition, list[Partition]] | None,
) -> PhaseHighlightGroups:
    collapse_group = [subtree]
    if collapse_sibling_groups:
        collapse_group = get_group_for_mover(subtree, collapse_sibling_groups)

    expand_group = [subtree]
    if expand_sibling_groups:
        expand_group = get_group_for_mover(subtree, expand_sibling_groups)

    reorder_group: list[Partition] = []
    seen_reorder_highlights: set[Partition] = set()
    for sibling_groups in (collapse_sibling_groups, expand_sibling_groups):
        if not sibling_groups:
            continue
        for mover in get_group_for_mover(subtree, sibling_groups):
            if mover in seen_reorder_highlights:
                continue
            reorder_group.append(mover)
            seen_reorder_highlights.add(mover)

    if not reorder_group:
        reorder_group = [subtree]

    return PhaseHighlightGroups(
        collapse=collapse_group,
        reorder=reorder_group,
        expand=expand_group,
    )
