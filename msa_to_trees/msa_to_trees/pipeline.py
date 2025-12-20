"""
A Python script to automate a sliding window phylogenetic analysis.
1. It generates overlapping window alignments from a single input MSA.
2. It then runs FastTree on each window to generate a phylogenetic tree.
3. Finally, it concatenates all resulting trees into a single output file.
"""

import argparse
import os
import subprocess
import glob
import concurrent.futures
from pathlib import Path
from math import ceil
from typing import Generator, Optional
from uuid import uuid4

# BioPython is required for this script
try:
    from Bio import AlignIO
    from Bio.Align import MultipleSeqAlignment
except ImportError:
    print("Error: BioPython is required but not installed.")
    print("Please install it by running: poetry add biopython")
    exit(1)


class WindowInfo:
    """A simple data class to hold window information."""

    def __init__(self, count: int, strtpos: int, endpos: int, midpos: int) -> None:
        self.count = count
        self.strtpos = strtpos
        self.endpos = endpos
        self.midpos = midpos
        # The name for the window file, based on the 1-based midpoint
        self.name = str(midpos + 1)


def get_windows_from_parameters(
    window_size: int,
    step_size: int,
    alignment_length: int,
) -> Generator[WindowInfo, None, None]:
    """
    Generates window information for a sliding window analysis.
    """
    leftwin = int(window_size / 2)
    rightwin = ceil(window_size / 2)

    for count, i in enumerate(range(0, alignment_length, step_size)):
        strtpos = max(0, i - leftwin)
        endpos = min(alignment_length, i + rightwin)
        yield WindowInfo(count=count, strtpos=strtpos, midpos=i, endpos=endpos)


def run_fasttree(alignment_file: str) -> str:
    """
    Runs FastTree on a single alignment file and returns the Newick tree string.
    Forces single-threaded execution to allow efficient parallel processing of multiple windows.
    """
    # Force FastTree to use a single thread per process to avoid oversubscription
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    cmd = ["fasttree", "-nt", "-quiet", alignment_file]

    try:
        # Run FastTree and capture its output (the Newick tree)
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env
        )
        return result.stdout

    except FileNotFoundError:
        raise RuntimeError("FastTree command not found. Please install FastTree.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FastTree failed on {alignment_file}: {e.stderr}")


def run_pipeline(
    input_file: str,
    output_directory: str,
    window_size: int,
    step_size: int,
    output_tree_filename: Optional[str] = None,
) -> Path:
    """Main function to run the analysis pipeline."""

    # --- Setup Paths and Directories ---
    output_dir = Path(output_directory)
    windows_dir = output_dir / "windows"
    trees_dir = output_dir / "trees"

    print("Creating output directories...")
    windows_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Generate Windowed Alignments ---
    print(f"Loading alignment from {input_file}...")
    try:
        alignment = AlignIO.read(input_file, "fasta")  # Assuming fasta for simplicity
    except Exception as e:
        print(f"Error reading alignment file: {e}")
        exit(1)

    alignment_length = alignment.get_alignment_length()
    print(
        f"Generating sliding window alignments (WinSize: {window_size}, Step: {step_size})..."
    )

    windows = get_windows_from_parameters(window_size, step_size, alignment_length)

    for window in windows:
        # Slice the alignment to get the window
        sub_alignment: MultipleSeqAlignment = alignment[
            :, window.strtpos : window.endpos
        ]  # type: ignore

        # Define the output path for the window fasta file
        window_fasta_path = windows_dir / f"{window.name}.fasta"

        # Write the sub-alignment to a new FASTA file
        AlignIO.write(sub_alignment, window_fasta_path, "fasta")

    print(f"Window generation complete. Files are in {windows_dir}")

    # --- Step 2: Infer Trees for each Window ---
    print("Running FastTree on each window...")

    # Clear any previously generated tree files
    for old_tree in glob.glob(str(trees_dir / "*.newick")):
        os.remove(old_tree)

    master_tree_file = trees_dir / (
        output_tree_filename if output_tree_filename else f"{uuid4().hex}.newick"
    )

    fasta_files = sorted(
        glob.glob(str(windows_dir / "*.fasta")), key=lambda x: int(Path(x).stem)
    )

    # Use ProcessPoolExecutor to run FastTree in parallel
    max_workers = os.cpu_count()
    print(f"Parallelizing tree inference across {max_workers} cores...")

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Map returns results in the order of the input iterable
            results = list(executor.map(run_fasttree, fasta_files))

        # Write all trees to the master file
        with open(master_tree_file, "w") as f:
            for tree_newick in results:
                f.write(tree_newick)

    except RuntimeError as e:
        print(f"\nError during tree inference: {e}")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        exit(1)

    print(
        f"Tree inference complete. All trees are concatenated into {master_tree_file}"
    )
    print("Analysis finished successfully.")

    return master_tree_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a sliding window analysis using FastTree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input multiple sequence alignment file (FASTA format).",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        required=True,
        help="Directory to store all output files (windows and trees).",
    )
    parser.add_argument(
        "-w",
        "--window-size",
        type=int,
        default=200,
        help="The size of each sliding window.",
    )
    parser.add_argument(
        "-s",
        "--step-size",
        type=int,
        default=50,
        help="The step size to move the window across the alignment.",
    )

    args = parser.parse_args()
    run_pipeline(
        input_file=args.input,
        output_directory=args.output_directory,
        window_size=args.window_size,
        step_size=args.step_size,
    )
