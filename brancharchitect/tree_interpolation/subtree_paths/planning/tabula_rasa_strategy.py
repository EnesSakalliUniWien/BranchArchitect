from typing import TYPE_CHECKING
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet

if TYPE_CHECKING:
    from .pivot_split_registry import PivotSplitRegistry


def get_tabula_rasa_collapse_splits(
    state: "PivotSplitRegistry",
) -> PartitionSet[Partition]:
    """
    Legacy helper for the old clean-slate collapse strategy.

    The active planner uses stepwise collapse paths. This module remains only for
    compatibility with experimental callers.

    Side Effects:
        - Sets state.first_subtree_processed to True.
    """
    if not state.first_subtree_processed:
        state.first_subtree_processed = True
        # Return ALL collapse splits - legacy clean-slate behavior.
        return state.all_collapsible_splits.copy()

    return PartitionSet(encoding=state.encoding)
