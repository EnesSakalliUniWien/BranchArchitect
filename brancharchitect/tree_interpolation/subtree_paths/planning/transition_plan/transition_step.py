"""Typed transition-plan entries for one pivot mover."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TypeAlias

from brancharchitect.elements.partition import Partition


@dataclass(frozen=True, slots=True)
class PivotTransitionStep:
    """Ordered collapse and expand work assigned to one driver subtree."""

    subtree: Partition
    collapse_path: tuple[Partition, ...]
    expand_path: tuple[Partition, ...]


PivotTransitionPlan: TypeAlias = OrderedDict[Partition, PivotTransitionStep]
