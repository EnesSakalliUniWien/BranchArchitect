"""Text formatting utilities for logging."""

import re
from typing import Any, Iterable, Set


def format_set(s: Set[Any]) -> str:
    """Format set for consistent display."""
    if not s:
        return "∅"
    return "{" + ", ".join(str(x) for x in sorted(s)) + "}"


def beautify_frozenset(obj: Any) -> str:
    """Convert frozenset objects to beautiful mathematical notation."""
    if obj is None:
        return "∅"

    # Convert to string first
    s = str(obj)

    # Replace frozenset({...}) with just {...}
    s = re.sub(r"frozenset\(\{(.+?)\}\)", r"{\1}", s)

    # Replace nested frozensets recursively
    while "frozenset" in s:
        s = re.sub(r"frozenset\(\{(.+?)\}\)", r"{\1}", s)

    # Replace double parentheses around single elements
    s = re.sub(r"\(\((.+?)\)\)", r"(\1)", s)

    # Clean up any remaining artifacts
    s = s.replace("()", "∅")
    s = s.replace("{}", "∅")

    return s


def format_partition(part: Any) -> str:
    """Format a Partition (or its tuple) as '(a, b, ...)'.

    If a Partition object exposes `reverse_encoding` and `indices`, map indices to
    human-readable taxon names using `reverse_encoding`.
    """
    # Prefer pretty names when available
    if hasattr(part, "reverse_encoding") and hasattr(part, "indices"):
        try:
            reverse_encoding = getattr(part, "reverse_encoding")
            indices: Iterable[Any] = getattr(part, "indices")
            taxa_names = sorted(reverse_encoding.get(i, str(i)) for i in indices)
            return "(" + ", ".join(taxa_names) + ")"
        except Exception:
            # Fallback to generic representation
            pass

    # Default behavior for non-Partition objects or if reverse_encoding fails
    try:
        values = tuple(part)  # works if part is iterable (like Partition)
    except TypeError:
        values = part
    return "(" + ", ".join(str(x) for x in values) + ")"


def format_partition_set(ps: Iterable[Any]) -> str:
    """Format a PartitionSet as a brace-enclosed, comma-separated list of partitions."""
    try:
        parts = sorted(ps, key=lambda p: format_partition(p))
    except Exception:
        # Robust fallback: best-effort formatting without sorting
        parts = list(ps)
    return "{" + ", ".join(format_partition(p) for p in parts) + "}"
