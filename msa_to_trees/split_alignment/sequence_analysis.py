"""Sequence analysis and filtering utilities."""

import re

from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord

from .constants import (
    NUCLEOTIDE_CHARACTERS,
    AMINO_ACID_CHARACTERS,
    AMBIGUOUS_NUCLEOTIDE_PATTERN,
    AMBIGUOUS_AMINO_ACID_PATTERN,
    AMBIGUOUS_OTHER_PATTERN,
)
from .models import SequenceType


def detect_sequence_type(alignment: MultipleSeqAlignment) -> SequenceType:
    """
    Detect the type of sequences in an alignment.

    Examines a sample of sequences to determine if they are nucleotides,
    amino acids, or other. Uses IUPAC character sets for detection.

    Args:
        alignment: The alignment to analyze.

    Returns:
        The detected sequence type.
    """
    # Build regex patterns for valid characters
    nuc_pattern = f"[^{'|'.join(NUCLEOTIDE_CHARACTERS)}]"
    amino_pattern = f"[^{'|'.join(AMINO_ACID_CHARACTERS)}]"

    # Sample first 10 sequences or all if fewer
    sample_size = min(len(alignment), 10)
    sequences: list[SeqRecord] = [alignment[i] for i in range(sample_size)]  # type: ignore
    sample_text = "".join(str(seq.seq) for seq in sequences).upper()

    # Check for invalid characters
    has_non_nucleotide = re.search(nuc_pattern, sample_text) is not None
    has_non_amino = re.search(amino_pattern, sample_text) is not None

    if not has_non_nucleotide:
        return SequenceType.NUCLEOTIDE
    elif not has_non_amino:
        return SequenceType.AMINO_ACID
    else:
        return SequenceType.OTHER


def get_ambiguous_pattern(sequence_type: SequenceType) -> str:
    """
    Get the regex pattern for ambiguous characters based on sequence type.

    Args:
        sequence_type: The type of sequences.

    Returns:
        Regex pattern string for matching sequences with only ambiguous characters.

    Raises:
        ValueError: If sequence type is invalid.
    """
    match sequence_type:
        case SequenceType.NUCLEOTIDE:
            return AMBIGUOUS_NUCLEOTIDE_PATTERN
        case SequenceType.AMINO_ACID:
            return AMBIGUOUS_AMINO_ACID_PATTERN
        case SequenceType.OTHER:
            return AMBIGUOUS_OTHER_PATTERN
        case _:
            raise ValueError(f"Invalid sequence type: {sequence_type}")


def filter_ambiguous_sequences(
    alignment: MultipleSeqAlignment,
    sequence_type: SequenceType,
) -> tuple[MultipleSeqAlignment, list[str]]:
    """
    Remove sequences that contain only ambiguous characters.

    Args:
        alignment: The alignment to filter.
        sequence_type: The type of sequences (determines ambiguous characters).

    Returns:
        Tuple of (filtered alignment, list of removed sequence IDs).
    """
    pattern = re.compile(get_ambiguous_pattern(sequence_type))

    kept_sequences: list[SeqRecord] = []
    removed_ids: list[str] = []

    for seq_record in alignment:
        if pattern.match(str(seq_record.seq)) is None:
            # Sequence has non-ambiguous characters, keep it
            kept_sequences.append(seq_record)
        else:
            # Sequence is entirely ambiguous, remove it
            removed_ids.append(seq_record.id)

    filtered_alignment = MultipleSeqAlignment(kept_sequences)
    return filtered_alignment, removed_ids
