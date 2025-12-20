"""Value object for tree pair identification."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PairKey:
    """
    Immutable identifier for a tree pair in an interpolation sequence.

    Represents the relationship between two consecutive trees (source -> target)
    in a phylogenetic tree sequence. Used as a type-safe key for accessing
    pair-specific interpolation data.

    Attributes:
        source_index: Zero-based index of the source tree in the original sequence
        target_index: Zero-based index of the target tree in the original sequence

    Example:
        >>> key = PairKey(0, 1)  # Represents T0 -> T1
        >>> str(key)
        'pair_0_1'
        >>> key.source_index
        0
    """

    source_index: int
    target_index: int

    def __post_init__(self) -> None:
        """Validate pair indices."""
        if self.source_index < 0:
            raise ValueError(
                f"source_index must be non-negative, got {self.source_index}"
            )
        if self.target_index < 0:
            raise ValueError(
                f"target_index must be non-negative, got {self.target_index}"
            )
        if self.target_index != self.source_index + 1:
            raise ValueError(
                f"target_index must be source_index + 1 (consecutive pairs only). "
                f"Got source={self.source_index}, target={self.target_index}"
            )

    def __str__(self) -> str:
        """Return standard string representation for backwards compatibility."""
        return f"pair_{self.source_index}_{self.target_index}"

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"PairKey(source={self.source_index}, target={self.target_index})"

    @classmethod
    def from_index(cls, pair_index: int) -> PairKey:
        """
        Create a PairKey from a zero-based pair index.

        Args:
            pair_index: The sequential pair number (0 for first pair, 1 for second, etc.)

        Returns:
            PairKey representing the pair at that position

        Example:
            >>> PairKey.from_index(0)  # First pair: T0 -> T1
            PairKey(source=0, target=1)
            >>> PairKey.from_index(2)  # Third pair: T2 -> T3
            PairKey(source=2, target=3)
        """
        if pair_index < 0:
            raise ValueError(f"pair_index must be non-negative, got {pair_index}")
        return cls(source_index=pair_index, target_index=pair_index + 1)

    @classmethod
    def from_string(cls, key_string: str) -> PairKey:
        """
        Parse a PairKey from its string representation.

        Args:
            key_string: String in format "pair_X_Y" where X and Y are integers

        Returns:
            PairKey object

        Raises:
            ValueError: If string format is invalid

        Example:
            >>> PairKey.from_string("pair_0_1")
            PairKey(source=0, target=1)
        """
        if not key_string.startswith("pair_"):
            raise ValueError(f"Invalid pair key format: {key_string}")

        parts = key_string.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid pair key format: {key_string}")

        try:
            source = int(parts[1])
            target = int(parts[2])
            return cls(source_index=source, target_index=target)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid pair key format: {key_string}") from e
