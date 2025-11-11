__all__ = [
    "call_jumping_taxa",
    "LatticeConstructionError",
    "JumpingTaxaError",
    "TreeEncodingError",
    "SplitLookupError",
]

from brancharchitect.jumping_taxa.api import call_jumping_taxa
from brancharchitect.jumping_taxa.exceptions import (
    JumpingTaxaError,
    LatticeConstructionError,
    TreeEncodingError,
    SplitLookupError,
)


# Use this pattern to avoid circular imports during static analysis
def __getattr__(name: str):
    if name == "call_jumping_taxa":
        return call_jumping_taxa
    elif name in (
        "JumpingTaxaError",
        "LatticeConstructionError",
        "TreeEncodingError",
        "SplitLookupError",
    ):
        from brancharchitect.jumping_taxa import exceptions

        return getattr(exceptions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
