"""Text formatting utilities for logging."""

import re
from typing import Any, Set

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


def format_partition(part):
    """Format a Partition (or its tuple representation) as '(a, b, ...)'."""
    try:
        values = tuple(part)  # works if part is iterable (like Partition)
    except TypeError:
        values = part
    return "(" + ", ".join(str(x) for x in values) + ")"


def format_partition_set(ps):
    """Format a PartitionSet as a brace-enclosed, comma-separated list of partitions."""
    parts = sorted(ps, key=lambda p: format_partition(p))
    return "{" + ", ".join(format_partition(p) for p in parts) + "}"