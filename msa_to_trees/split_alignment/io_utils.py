"""Utilities for loading and validating alignment files."""

from pathlib import Path
from typing import Optional

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment

from .constants import SUPPORTED_ALIGNMENT_FORMATS


def load_alignment(
    file_path: Path | str,
    format_hint: Optional[str] = None,
) -> tuple[MultipleSeqAlignment, str]:
    """
    Load an alignment file and auto-detect its format.

    Attempts to parse the alignment using various supported formats.
    Raises ValueError if the file cannot be parsed in any format.

    Args:
        file_path: Path to the alignment file.
        format_hint: Optional format specifier to skip auto-detection.

    Returns:
        Tuple of (alignment object, detected format string).

    Raises:
        ValueError: If the file cannot be parsed in any supported format.
    """
    alignment: Optional[MultipleSeqAlignment] = None
    detected_format = ""

    if format_hint:
        # Use provided format
        detected_format = format_hint
        alignment = AlignIO.read(file_path, detected_format)
    else:
        # Try each format until one works
        for seq_format in SUPPORTED_ALIGNMENT_FORMATS:
            try:
                alignment = AlignIO.read(file_path, seq_format)
                detected_format = seq_format
                break
            except Exception:
                continue

    if alignment is None:
        raise ValueError(
            f"Unable to parse alignment file. Tried formats: {SUPPORTED_ALIGNMENT_FORMATS}"
        )

    return alignment, detected_format


def sanitize_sequence_ids(
    alignment: MultipleSeqAlignment,
    char_replacements: dict[str, str],
) -> None:
    """
    Replace illegal characters in sequence IDs (in-place modification).

    This ensures compatibility with tools like RAxML that have restrictions
    on sequence identifier characters.

    Args:
        alignment: The alignment to modify.
        char_replacements: Dictionary mapping characters to their replacements.
    """
    translation_table = str.maketrans(char_replacements)
    for seq in alignment:
        seq.id = seq.id.translate(translation_table)
