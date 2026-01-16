from typing import TYPE_CHECKING
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet

if TYPE_CHECKING:
    from .pivot_split_registry import PivotSplitRegistry


def get_tabula_rasa_collapse_splits(
    state: "PivotSplitRegistry",
) -> PartitionSet[Partition]:
    """
    Get ALL collapse splits for tabula rasa strategy.

    If this is the first subtree, return all collapsible splits to perform a "Big Bang"
    collapse (clearing the canvas).
    Otherwise return empty set.

    Side Effects:
        - Sets state.first_subtree_processed to True.
    """
    if not state.first_subtree_processed:
        state.first_subtree_processed = True
        # Return ALL collapse splits - complete tabula rasa
        return state.all_collapsible_splits.copy()

    return PartitionSet(encoding=state.encoding)
