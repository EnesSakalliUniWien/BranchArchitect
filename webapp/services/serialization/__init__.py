"""
Serialization utilities for converting complex objects to JSON-serializable formats.
"""

from webapp.services.serialization.partition import (
    serialize_partition_to_indices,
    serialize_partition_dict_to_indices,
)

__all__ = [
    "serialize_partition_to_indices",
    "serialize_partition_dict_to_indices",
]
