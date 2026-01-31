"""Pipeline result dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineResult:
    """Result from running the tree inference pipeline."""

    tree_file_path: Path
    """Path to the generated tree file containing all trees."""

    total_taxa: int
    """Total number of taxa in the original alignment."""

    kept_taxa: int
    """Number of taxa kept (valid in all windows)."""

    dropped_taxa: list[str] = field(default_factory=list)
    """List of taxa IDs that were dropped due to invalid data in some windows."""

    dropped_taxa_reasons: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of dropped taxa ID to list of window names where they had invalid data."""

    num_windows: int = 0
    """Number of windows generated."""

    @property
    def has_dropped_taxa(self) -> bool:
        """Whether any taxa were dropped."""
        return len(self.dropped_taxa) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tree_file_path": str(self.tree_file_path),
            "total_taxa": self.total_taxa,
            "kept_taxa": self.kept_taxa,
            "dropped_taxa": self.dropped_taxa,
            "dropped_taxa_reasons": self.dropped_taxa_reasons,
            "num_windows": self.num_windows,
            "has_dropped_taxa": self.has_dropped_taxa,
        }
