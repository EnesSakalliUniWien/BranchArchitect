#!/usr/bin/env python3
"""
Command-line interface for splitting multiple sequence alignments.

Splits a multiple sequence alignment into sub-alignments using either a
sliding window approach or custom ranges from a CSV file.
"""

import argparse
from pathlib import Path
from typing import Dict, List

from Bio import AlignIO

from .constants import ILLEGAL_ID_CHARACTERS
from .io_utils import load_alignment, sanitize_sequence_ids
from .sequence_analysis import detect_sequence_type, filter_ambiguous_sequences
from .validators import PositiveIntegerAction
from .windowing import (
    apply_sliding_window,
    create_windows_from_file,
    create_windows_from_parameters,
)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure and return the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        help="Path to the alignment file",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="Output directory for sub-alignments",
        required=True,
        type=Path,
    )

    # Windowing options
    windowing_group = parser.add_argument_group("windowing options")
    windowing_group.add_argument(
        "-w",
        "--window-size",
        help="Size of each window (default: 300)",
        default=300,
        type=int,
        action=PositiveIntegerAction,
    )
    windowing_group.add_argument(
        "-s",
        "--step-size",
        help="Distance between window centers (default: 25)",
        default=25,
        type=int,
        action=PositiveIntegerAction,
    )
    windowing_group.add_argument(
        "--split-file",
        help=(
            "CSV file with custom ranges (columns: start,end[,name]). "
            "If start > end for nucleotides, uses reverse complement."
        ),
        type=Path,
    )
    windowing_group.add_argument(
        "-1",
        "--one-based",
        dest="one_based",
        help="Treat CSV ranges as 1-indexed (default: 0-indexed)",
        action="store_true",
    )

    # Filtering options
    filtering_group = parser.add_argument_group("filtering options")
    filtering_group.add_argument(
        "--keep-ambiguous",
        help="Keep sequences with only ambiguous characters",
        action="store_true",
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "-f",
        "--force",
        help="Overwrite existing output files",
        action="store_true",
    )
    output_group.add_argument(
        "-l",
        "--log",
        help="Generate log files (windows.log, removed_sequences.log)",
        action="store_true",
    )

    return parser


def write_log_files(
    output_dir: Path,
    windows_log: List[str],
    removed_sequences: Dict[str, List[str]],
) -> None:
    """
    Write log files for windows and removed sequences.

    Args:
        output_dir: Directory to write log files to.
        windows_log: List of window information strings.
        removed_sequences: Dict mapping window names to removed sequence IDs.
    """
    # Write windows log
    windows_log_path = output_dir / "windows.log"
    with open(windows_log_path, "w") as f:
        f.write("\n".join(windows_log))

    # Write removed sequences log if any
    if removed_sequences:
        removed_log_path = output_dir / "removed_sequences.log"
        entries = [
            f"{window}\t{','.join(seq_ids)}"
            for window, seq_ids in removed_sequences.items()
        ]
        with open(removed_log_path, "w") as f:
            f.write("\n".join(entries))


def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Load and prepare alignment
    print(f"Loading alignment from {args.input}...")
    alignment, detected_format = load_alignment(args.input)
    print(f"Detected format: {detected_format}")
    print(f"Alignment length: {alignment.get_alignment_length()}")
    print(f"Number of sequences: {len(alignment)}")

    # Sanitize sequence IDs for compatibility
    sanitize_sequence_ids(alignment, ILLEGAL_ID_CHARACTERS)

    # Detect sequence type
    sequence_type = detect_sequence_type(alignment)
    print(f"Detected sequence type: {sequence_type.name}")

    # Generate windows
    alignment_length = alignment.get_alignment_length()
    if args.split_file:
        print(f"Loading windows from {args.split_file}...")
        windows = create_windows_from_file(args.split_file, args.one_based)
    else:
        print(
            f"Generating sliding windows (size={args.window_size}, step={args.step_size})..."
        )
        windows = create_windows_from_parameters(
            args.window_size,
            args.step_size,
            alignment_length,
        )

    # Create output directory
    args.output_directory.mkdir(parents=True, exist_ok=True)

    # Process windows
    windows_log = ["count\tstart\tmid\tend\twin_len\tname"]
    removed_sequences: Dict[str, List[str]] = {}
    num_windows = 0

    print("Processing windows...")
    for sub_alignment, window in apply_sliding_window(
        alignment, windows, sequence_type
    ):
        num_windows += 1
        windows_log.append(window.get_display_string())

        # Prepare output path
        output_path = (args.output_directory / window.name).with_suffix(".fasta")

        # Check for existing files
        if output_path.exists() and not args.force:
            raise FileExistsError(
                f"Output file {output_path} already exists. Use --force to overwrite."
            )

        # Filter ambiguous sequences if requested
        if not args.keep_ambiguous:
            sub_alignment, removed_ids = filter_ambiguous_sequences(
                sub_alignment,
                sequence_type,
            )
            if removed_ids:
                removed_sequences[window.name] = removed_ids

        # Write sub-alignment
        AlignIO.write(sub_alignment, output_path, "fasta-2line")

    print(f"Generated {num_windows} sub-alignments in {args.output_directory}")

    # Write log files if requested
    if args.log:
        print("Writing log files...")
        write_log_files(args.output_directory, windows_log, removed_sequences)

    if removed_sequences:
        total_removed = sum(len(ids) for ids in removed_sequences.values())
        print(
            f"Removed {total_removed} ambiguous sequences across {len(removed_sequences)} windows"
        )

    print("Done!")


if __name__ == "__main__":
    main()
