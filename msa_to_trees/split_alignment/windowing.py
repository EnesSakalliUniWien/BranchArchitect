"""Window generation and sliding window operations."""

from math import ceil
from pathlib import Path
from typing import Generator, Iterable

from Bio.Align import MultipleSeqAlignment

from .models import WindowInfo, SequenceType


def create_windows_from_parameters(
    window_size: int,
    step_size: int,
    alignment_length: int,
) -> Generator[WindowInfo, None, None]:
    """
    Generate sliding windows from size and step parameters.

    Creates overlapping windows across the alignment. For even window sizes,
    the left side will be one position longer than the right side.

    Args:
        window_size: Size of each window.
        step_size: Distance between window centers.
        alignment_length: Total length of the alignment.

    Yields:
        WindowInfo objects for each window position.
    """
    left_half = int(window_size / 2)
    right_half = ceil(window_size / 2)

    for count, center_pos in enumerate(range(0, alignment_length, step_size)):
        yield WindowInfo(
            count=count,
            start_pos=max(0, center_pos - left_half),
            mid_pos=center_pos,
            end_pos=min(alignment_length, center_pos + right_half),
        )


def create_windows_from_file(
    csv_path: Path,
    one_based: bool = False,
) -> Generator[WindowInfo, None, None]:
    """
    Generate windows from ranges specified in a CSV file.

    CSV format:
    - Column 1: Start position
    - Column 2: End position
    - Column 3 (optional): Window name

    If start > end, the positions are swapped and reverse_complement is set
    to True (for nucleotide sequences).

    Args:
        csv_path: Path to CSV file with window ranges.
        one_based: Whether CSV positions are 1-indexed (converts to 0-indexed).

    Yields:
        WindowInfo objects for each range in the file.

    Raises:
        ValueError: If CSV format is invalid or positions are not integers.
    """
    # Parse CSV file
    with open(csv_path, "r") as f:
        file_content = f.read()

    rows = [line.split(",") for line in file_content.strip().split("\n")]
    num_columns = len(rows[0])

    if num_columns < 2:
        raise ValueError(
            "CSV file must have at least 2 columns: start_pos, end_pos. "
            "Optional 3rd column: window_name"
        )

    # Parse and validate position columns
    try:
        parsed_rows: list[list[int | str]] = [
            [int(row[0]), int(row[1])] + row[2:] for row in rows
        ]
    except ValueError as e:
        raise ValueError(f"First two columns must be integer positions: {e}") from e

    # Generate windows with optional names
    for count, row in enumerate(parsed_rows):
        start: int = int(row[0])
        end: int = int(row[1])
        name: str | None = str(row[2]) if num_columns > 2 else None

        # Convert from 1-indexed to 0-indexed if needed
        if one_based:
            start -= 1
            # end stays the same (exclusive end in Python slicing)

        # Calculate midpoint
        mid = int((start + end) / 2)

        # Handle reverse complement case
        if start > end:
            # Swap positions and flag for reverse complement
            yield WindowInfo(
                count=count,
                start_pos=end,
                mid_pos=mid,
                end_pos=start,
                name=name,
                reverse_complement=True,
            )
        else:
            yield WindowInfo(
                count=count,
                start_pos=start,
                mid_pos=mid,
                end_pos=end,
                name=name,
            )


def apply_sliding_window(
    alignment: MultipleSeqAlignment,
    windows: Iterable[WindowInfo],
    sequence_type: SequenceType,
) -> Generator[tuple[MultipleSeqAlignment, WindowInfo], None, None]:
    """
    Extract sub-alignments for each window from the full alignment.

    For nucleotide sequences with reverse_complement=True, computes the
    reverse complement of the sub-alignment.

    Args:
        alignment: The full multiple sequence alignment.
        windows: Iterable of window specifications.
        sequence_type: Type of sequences (affects reverse complement handling).

    Yields:
        Tuples of (sub-alignment, window_info) for each window.
    """
    for window in windows:
        sub_alignment: MultipleSeqAlignment = alignment[
            :, window.start_pos : window.end_pos
        ]  # type: ignore

        if window.reverse_complement and sequence_type == SequenceType.NUCLEOTIDE:
            # Apply reverse complement to all sequences
            for seq_record in sub_alignment:  # type: ignore
                seq_record.seq = seq_record.seq.reverse_complement()

        yield sub_alignment, window
