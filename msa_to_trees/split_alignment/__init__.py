"""
Split alignment module for creating sub-alignments from multiple sequence alignments.

This module provides functionality to split a multiple sequence alignment into
several sub-alignments using either a sliding window approach or custom ranges
specified in a CSV file.
"""

from .models import SequenceType, WindowInfo
from .io_utils import load_alignment
from .windowing import (
    create_windows_from_parameters,
    create_windows_from_file,
    apply_sliding_window,
)
from .sequence_analysis import detect_sequence_type, filter_ambiguous_sequences

__all__ = [
    "SequenceType",
    "WindowInfo",
    "load_alignment",
    "create_windows_from_parameters",
    "create_windows_from_file",
    "apply_sliding_window",
    "detect_sequence_type",
    "filter_ambiguous_sequences",
]
