"""
Custom exceptions for the jumping taxa analysis module.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from brancharchitect.elements.partition import Partition


class JumpingTaxaError(Exception):
    """Base exception for jumping taxa analysis errors."""

    pass


class LatticeConstructionError(JumpingTaxaError):
    """Raised when lattice construction encounters an inconsistency or error condition."""

    @staticmethod
    def raise_missing_node(
        split: Partition, is_missing_in_tree1: bool, is_missing_in_tree2: bool
    ) -> NoReturn:
        """
        Raises a LatticeConstructionError when a common split cannot be found in one or both trees.

        Args:
            split: The split (Partition) that could not be found
            is_missing_in_tree1: True if the split is missing in tree 1
            is_missing_in_tree2: True if the split is missing in tree 2

        Raises:
            LatticeConstructionError: Always raised with detailed error information
        """
        from brancharchitect.logger.debug import jt_logger

        missing_node_error_message = (
            f"Failed to find node for common split {split.bipartition()}. "
            f"Missing in tree 1: {is_missing_in_tree1}, Missing in tree 2: {is_missing_in_tree2}. "
            f"This indicates an inconsistency between split enumeration and node indexing."
        )
        if not jt_logger.disabled:
             jt_logger.error(missing_node_error_message)
        raise LatticeConstructionError(missing_node_error_message)


class TreeEncodingError(JumpingTaxaError):
    """Raised when tree encoding issues are detected."""

    pass


class SplitLookupError(JumpingTaxaError):
    """Raised when split lookups fail."""

    pass
