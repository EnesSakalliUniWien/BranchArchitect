"""Serialization utilities for converting Partition objects to JSON-serializable formats."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from brancharchitect.elements.partition import Partition


def serialize_partition_to_indices(
    partition: Optional["Partition"],
) -> Optional[List[int]]:
    """Convert a Partition object to its indices list for JSON serialization.

    Args:
        partition: A Partition object or None

    Returns:
        List of indices if partition is not None, otherwise None
    """
    return list(partition.indices) if partition is not None else None


def serialize_partition_dict_to_indices(
    partition_dict: Dict[Any, Any],
) -> Dict[str, Any]:
    """Convert a dictionary with Partition keys to index-based keys.

    Args:
        partition_dict: Dictionary with Partition objects as keys

    Returns:
        Dictionary with string keys representing serialized Partition indices
    """

    def _serialize_value(val: Any) -> Any:
        # Partition-like objects: have 'indices'
        if hasattr(val, "indices"):
            return serialize_partition_to_indices(val)  # -> List[int]
        # Lists or other iterables of partitions
        if isinstance(val, list):
            out: List[Any] = []
            for item in val:
                out.append(_serialize_value(item))
            return out
        # Nested dicts: recursively serialize keys and values
        if isinstance(val, dict):
            return serialize_partition_dict_to_indices(val)
        # Primitive or unknown: pass through
        return val

    return {
        str(serialize_partition_to_indices(key)): _serialize_value(value)
        for key, value in partition_dict.items()
    }
