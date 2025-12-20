"""Data models for the split alignment module."""

from enum import Enum
from typing import Optional


class SequenceType(Enum):
    """Enumeration of supported sequence types."""

    NUCLEOTIDE = 1
    AMINO_ACID = 2
    OTHER = 3


class WindowInfo:
    """
    Information about a sub-alignment window.

    Attributes:
        count: The sequential number of the window.
        start_pos: The start position of the window (0-indexed).
        mid_pos: The middle position of the window (0-indexed).
        end_pos: The end position of the window (0-indexed).
        length: The number of positions in the window.
        name: The name of the window (defaults to 1-indexed midpoint).
        reverse_complement: Whether to use the reverse complement.
    """

    def __init__(
        self,
        count: int,
        start_pos: int,
        end_pos: int,
        mid_pos: Optional[int] = None,
        name: Optional[str] = None,
        reverse_complement: bool = False,
    ):
        """
        Initialize a WindowInfo object.

        Args:
            count: The sequential number of the window.
            start_pos: The start position (0-indexed).
            end_pos: The end position (0-indexed).
            mid_pos: The middle position (0-indexed). Auto-calculated if None.
            name: The name of the window. Defaults to 1-indexed midpoint.
            reverse_complement: Whether to use the reverse complement.
        """
        self.count = count
        self.start_pos = start_pos
        self.mid_pos = (
            mid_pos if mid_pos is not None else int((end_pos + start_pos) / 2)
        )
        self.end_pos = end_pos
        self.length = end_pos - start_pos
        self.name = str(self.mid_pos + 1) if name is None else name
        self.reverse_complement = reverse_complement

    @property
    def start_pos_display(self) -> int:
        """Get 1-indexed start position for display."""
        return self.start_pos + 1

    @property
    def mid_pos_display(self) -> int:
        """Get 1-indexed middle position for display."""
        return self.mid_pos + 1

    def get_display_string(self) -> str:
        """
        Get 1-based information of the window as tab-separated values.

        Returns:
            Tab-separated string with count, start, mid, end, length, and name.
        """
        return (
            f"{self.count}\t{self.start_pos_display}\t{self.mid_pos_display}\t"
            f"{self.end_pos}\t{self.length}\t{self.name}"
        )

    def __repr__(self) -> str:
        """String representation with 0-indexed positions."""
        return (
            f"{self.count}\t{self.start_pos}\t{self.mid_pos}\t"
            f"{self.end_pos}\t{self.length}\t{self.name}"
        )
