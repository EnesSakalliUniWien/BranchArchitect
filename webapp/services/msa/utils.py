"""
Simplified MSA utilities - only what's needed for tree processing.
"""

import re
from io import StringIO
from logging import Logger
from typing import Any, Dict, Optional

import skbio


def get_alignment_length(msa_content: str) -> Optional[int]:
    """
    Extract alignment length from MSA content.

    Supports FASTA, PHYLIP, and Clustal formats.

    Args:
        msa_content: Raw MSA content as string.

    Returns:
        Alignment length if parseable, None otherwise.
    """
    if not msa_content:
        return None

    lines = msa_content.strip().split("\n")

    # FASTA format
    if any(line.startswith(">") for line in lines):
        sequences: list[str] = []
        current_seq = ""
        for line in lines:
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            sequences.append(current_seq)

        return len(sequences[0]) if sequences else None

    # PHYLIP format
    elif re.match(r"^\s*\d+\s+\d+", lines[0]):
        header_match = re.match(r"^\s*(\d+)\s+(\d+)", lines[0])
        if header_match:
            return int(header_match.group(2))

    # Clustal format
    elif any("CLUSTAL" in line.upper() for line in lines[:3]):
        for line in lines:
            if (
                line.strip()
                and not line.startswith("CLUSTAL")
                and not line.strip().startswith("*")
            ):
                parts = line.split()
                if len(parts) >= 2:
                    return len("".join(parts[1:]))

    return None


class WindowParameters:
    """Simple window parameters class."""

    def __init__(self, window_size: int, step_size: int):
        self.window_size = window_size
        self.step_size = step_size
        self.is_overlapping = step_size < window_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "step_size": self.step_size,
            "is_overlapping": self.is_overlapping,
        }


def infer_window_parameters(num_trees: int, alignment_length: int) -> WindowParameters:
    """
    Infer sliding window parameters from tree count and alignment length.

    Args:
        num_trees: Number of trees in the dataset.
        alignment_length: Length of the MSA alignment.

    Returns:
        WindowParameters with inferred values.
    """
    if num_trees <= 1:
        return WindowParameters(alignment_length, alignment_length)

    window_size = max(1, alignment_length // num_trees)
    step_size = max(1, alignment_length // num_trees)

    return WindowParameters(window_size, step_size)


def msa_to_dict(msa_content: str) -> Dict[str, str]:
    """
    Parse MSA content (FASTA) into a dictionary.

    Args:
        msa_content: Raw MSA content in FASTA format.

    Returns:
        Dictionary mapping sequence IDs to sequences.
    """
    try:
        msa = skbio.io.read(StringIO(msa_content), format="fasta")  # type: ignore
        return {seq.metadata["id"]: str(seq) for seq in msa}  # type: ignore
    except Exception:
        return {}


def process_msa_data(
    msa_content: Optional[str],
    num_trees: int,
    window_size: int = 1,
    step_size: int = 1,
    logger: Optional[Logger] = None,
) -> Dict[str, Any]:
    """
    Process MSA content and determine effective window parameters.

    Behavior:
    - If both window_size and step_size are 1, infer parameters from the MSA
      alignment length and number of trees (when MSA is provided and parsable).
    - Otherwise, echo back the provided window_size/step_size as the effective
      values so the API response reflects user input.

    Args:
        msa_content: Raw MSA content (optional).
        num_trees: Number of trees in the dataset.
        window_size: User-provided window size.
        step_size: User-provided step size.
        logger: Optional logger for debug output.

    Returns:
        Dictionary with inferred parameters and parsed MSA data.
    """
    # If no MSA provided, still surface the provided parameters
    if not msa_content:
        return {
            "inferred_window_size": window_size,
            "inferred_step_size": step_size,
            "windows_are_overlapping": (step_size < window_size),
            "alignment_length": None,
            "msa_dict": None,
        }

    alignment_length = get_alignment_length(msa_content)
    if not alignment_length:
        if logger:
            logger.warning("Could not determine alignment length from MSA content")
        return {
            "inferred_window_size": window_size,
            "inferred_step_size": step_size,
            "windows_are_overlapping": (step_size < window_size),
            "alignment_length": None,
            "msa_dict": None,
        }

    if logger:
        logger.info(f"MSA alignment length: {alignment_length}")

    # Determine effective parameters: infer only when both are 1
    if window_size == 1 and step_size == 1:
        window_params = infer_window_parameters(num_trees, alignment_length)
        if logger:
            logger.info(f"Inferred window parameters: {window_params.to_dict()}")
        effective_window_size = window_params.window_size
        effective_step_size = window_params.step_size
        overlapping = window_params.is_overlapping
    else:
        effective_window_size = window_size
        effective_step_size = step_size
        overlapping = step_size < window_size

    msa_dict = msa_to_dict(msa_content)

    return {
        "inferred_window_size": effective_window_size,
        "inferred_step_size": effective_step_size,
        "windows_are_overlapping": overlapping,
        "alignment_length": alignment_length,
        "msa_dict": msa_dict,
    }
