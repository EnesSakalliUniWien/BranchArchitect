"""
A Python script to automate a sliding window phylogenetic analysis.
1. It generates overlapping window alignments from a single input MSA.
2. It then runs the configured inference engine on each window to generate a phylogenetic tree.
3. Finally, it concatenates all resulting trees into a single output file.

Taxa consistency is enforced: only taxa with valid (non-gap/ambiguous) data
in ALL windows are included, ensuring all trees have the same taxa set.
"""

import argparse
import concurrent.futures
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional, cast
from uuid import uuid4

from split_alignment.constants import AMBIGUOUS_NUCLEOTIDE_PATTERN
from split_alignment.models import WindowInfo
from split_alignment.windowing import create_windows_from_parameters

from .result import PipelineResult

# Module logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FastTreeConfig:
    """
    Configuration for FastTree model settings.

    Attributes:
        use_gtr: Use GTR (General Time Reversible) model instead of JC.
        use_gamma: Use gamma rate heterogeneity for rate variation across sites.
        use_pseudo: Use pseudocounts for sequences with gaps/little overlap.
        no_ml: Disable ML NNI moves to produce fully bifurcating trees.
    """

    # Defaults tuned for speed: start simple, allow callers to opt into heavier models.
    use_gtr: bool = False
    use_gamma: bool = False
    use_pseudo: bool = False
    no_ml: bool = True

    @property
    def description(self) -> str:
        """Human-readable description of the model configuration."""
        parts = []
        if self.use_gtr:
            parts.append("GTR")
        if self.use_gamma:
            parts.append("gamma")
        if self.use_pseudo:
            parts.append("pseudo")
        if self.no_ml:
            parts.append("no-ML")
        return " + ".join(parts) if parts else "JC (fast default)"

    def build_command_args(self) -> list[str]:
        """Build FastTree command line arguments for this configuration."""
        args = []
        if self.use_gtr:
            args.append("-gtr")
        if self.use_gamma:
            args.append("-gamma")
        if self.use_pseudo:
            args.append("-pseudo")
        if self.no_ml:
            args.append("-noml")
        return args


@dataclass(frozen=True)
class IQTreeConfig:
    """
    Configuration for IQ-TREE model settings.

    IQ-TREE is bundled in frozen application builds. Set IQTREE_PATH to
    override executable discovery, otherwise iqtree3, iqtree2, or iqtree is
    used from PATH.
    """

    use_gtr: bool = True
    use_gamma: bool = True
    threads: int = 1
    model: str | None = None
    fast_search: bool = True
    support_mode: str = "none"
    ufboot_replicates: int = 1000
    sh_alrt_replicates: int = 1000
    bnni: bool = False

    @property
    def description(self) -> str:
        return f"IQ-TREE {self.model_name}"

    @property
    def model_name(self) -> str:
        if self.model:
            return self.model
        base_model = "GTR" if self.use_gtr else "JC"
        if self.use_gamma:
            return f"{base_model}+G"
        return base_model

    def build_command_args(self, alignment_file: str, prefix: Path) -> list[str]:
        """Build IQ-TREE command line arguments for this configuration."""
        supports_ufboot = self.support_mode in {"ufboot", "sh_alrt_ufboot"}
        args = [
            "-s",
            alignment_file,
            "-st",
            "DNA",
            "-m",
            self.model_name,
            "-nt",
            str(self.threads),
        ]
        if self.fast_search and not supports_ufboot:
            args.append("-fast")
        if supports_ufboot:
            args.extend(["-B", str(self.ufboot_replicates)])
        if self.support_mode in {"sh_alrt", "sh_alrt_ufboot"}:
            args.extend(["-alrt", str(self.sh_alrt_replicates)])
        if self.bnni and self.support_mode != "none":
            args.append("-bnni")
        args.extend(
            [
                "-quiet",
                "-redo",
                "-pre",
                str(prefix),
            ]
        )
        return args


TreeInferenceConfig = FastTreeConfig | IQTreeConfig


# BioPython is required for this script
try:
    from Bio import AlignIO
    from Bio.Align import MultipleSeqAlignment
except ImportError:
    # BioPython is required
    raise ImportError(
        "Error: BioPython is required but not installed. "
        "Please install it by running: poetry add biopython"
    )


# Use the ambiguous pattern from split_alignment, compiled for efficiency
_AMBIGUOUS_PATTERN = re.compile(AMBIGUOUS_NUCLEOTIDE_PATTERN, re.IGNORECASE)


def is_sequence_valid(sequence: str) -> bool:
    """
    Check if a sequence contains valid (non-gap/ambiguous) characters.

    Uses the nucleotide ambiguity pattern from split_alignment.constants.

    Args:
        sequence: The sequence string to check.

    Returns:
        True if the sequence has at least one valid character, False otherwise.
    """
    return _AMBIGUOUS_PATTERN.match(sequence) is None


def generate_filtered_window_alignments(
    alignment: MultipleSeqAlignment,
    windows_list: list[WindowInfo],
    kept_taxa_ids: set[str],
    windows_dir: Path,
) -> None:
    """
    Generate filtered windowed alignments from an MSA.

    For each window, slices the alignment, filters to keep only taxa that are
    valid across all windows, and writes the result to a FASTA file.

    Args:
        alignment: The full multiple sequence alignment.
        windows_list: List of WindowInfo objects defining the windows.
        kept_taxa_ids: Set of taxa IDs to keep (valid in all windows).
        windows_dir: Directory to write the window FASTA files.
    """
    for window in windows_list:
        # Slice the alignment to get the window
        sub_alignment: MultipleSeqAlignment = alignment[
            :, window.start_pos : window.end_pos
        ]  # type: ignore

        # Filter to keep only consistent taxa
        filtered_records = [
            record for record in sub_alignment if record.id in kept_taxa_ids
        ]
        filtered_alignment = MultipleSeqAlignment(filtered_records)

        # Define the output path for the window fasta file
        window_fasta_path = windows_dir / f"{window.name}.fasta"

        # Write the filtered sub-alignment to a new FASTA file
        AlignIO.write(filtered_alignment, window_fasta_path, "fasta")


def scan_invalid_taxa(
    alignment: MultipleSeqAlignment,
    windows_list: list[WindowInfo],
) -> dict[str, list[str]]:
    """
    Pre-scan all windows to find taxa with invalid sequences.

    A taxon is considered invalid in a window if its sequence contains only
    gaps or ambiguous characters.

    Args:
        alignment: The full multiple sequence alignment.
        windows_list: List of WindowInfo objects defining the windows.

    Returns:
        Dictionary mapping taxa IDs to lists of window names where they are invalid.
    """
    invalid_taxa_windows: dict[str, list[str]] = {}

    for window in windows_list:
        sub_alignment: MultipleSeqAlignment = alignment[
            :, window.start_pos : window.end_pos
        ]  # type: ignore

        for record in sub_alignment:
            seq_str = str(record.seq)
            if not is_sequence_valid(seq_str):
                if record.id not in invalid_taxa_windows:
                    invalid_taxa_windows[record.id] = []
                invalid_taxa_windows[record.id].append(window.name)

    return invalid_taxa_windows


def infer_trees_parallel(
    windows_dir: Path,
    trees_dir: Path,
    output_tree_filename: str | None,
    config: TreeInferenceConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    """
    Run tree inference in parallel on all window alignments.

    Args:
        windows_dir: Directory containing window FASTA files.
        trees_dir: Directory to store tree files.
        output_tree_filename: Optional custom filename for output tree file.
        config: Tree inference model configuration.

    Returns:
        Path to the master tree file containing all trees.
    """
    # Clear any previously generated tree files
    for old_tree in glob.glob(str(trees_dir / "*.newick")):
        os.remove(old_tree)

    master_tree_file = trees_dir / (
        output_tree_filename if output_tree_filename else f"{uuid4().hex}.newick"
    )

    all_fasta_files: list[str] = glob.glob(str(windows_dir / "*.fasta"))
    fasta_files = sorted(all_fasta_files, key=lambda x: int(Path(x).stem))

    # Detect if running in PyInstaller frozen executable
    is_frozen = getattr(sys, "frozen", False)

    if is_frozen:
        # In frozen executables, multiprocessing spawn doesn't work reliably
        # due to how PyInstaller packages the application. Run sequentially
        # to ensure stability. Performance impact is acceptable for typical
        # alignment sizes in interactive usage.
        results = []
        total = len(fasta_files)
        for completed, fasta_file in enumerate(fasta_files, start=1):
            results.append(_run_tree_inference(fasta_file, config))
            if progress_callback:
                progress_callback(completed, total, fasta_file)
    else:
        # Use ProcessPoolExecutor to run tree inference in parallel
        max_workers = os.cpu_count()

        # Create a partial function with config bound
        inference_runner = partial(_run_tree_inference, config=config)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_index = {
                executor.submit(inference_runner, fasta_file): index
                for index, fasta_file in enumerate(fasta_files)
            }
            results_by_index: list[str | None] = [None] * len(fasta_files)
            total = len(fasta_files)

            for completed, future in enumerate(
                concurrent.futures.as_completed(future_to_index), start=1
            ):
                index = future_to_index[future]
                results_by_index[index] = future.result()
                if progress_callback:
                    progress_callback(completed, total, fasta_files[index])

            results = [result for result in results_by_index if result is not None]

    # Write all trees to the master file
    with open(master_tree_file, "w") as f:
        for tree_newick in results:
            f.write(tree_newick)

    return master_tree_file


def _run_tree_inference(alignment_file: str, config: TreeInferenceConfig) -> str:
    """Run the configured tree inference engine for one alignment."""
    if isinstance(config, IQTreeConfig):
        return run_iqtree(alignment_file, config)
    return run_fasttree(alignment_file, config)


def _get_fasttree_exe() -> str:
    """
    Determine the path to the FastTree executable.
    Prioritizes environment variable > Bundled binary (frozen) > System PATH.
    """
    # 1. Environment variable override
    if "FASTTREE_PATH" in os.environ:
        return os.environ["FASTTREE_PATH"]

    # 2. Check for bundled binary in PyInstaller frozen app
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            # One-file mode
            base_path = Path(sys._MEIPASS)
        else:
            # One-dir mode (macOS/Linux executable location)
            base_path = Path(sys.executable).parent

        system = platform.system().lower()
        if system == "darwin":
            platform_dir = "darwin"
            exe_name = "fasttree"
        elif system == "windows":
            platform_dir = "win32"
            exe_name = "fasttree.exe"
        else:
            platform_dir = "linux"
            exe_name = "fasttree"

        bundled_exe = base_path / "bin" / platform_dir / exe_name

        if bundled_exe.exists():
            return str(bundled_exe)

    # 3. Fallback to system PATH
    return "fasttree"


def _get_iqtree_exe() -> str:
    """
    Determine the path to the IQ-TREE executable.
    Prioritizes environment variable > bundled binary > common system PATH executable names.
    """
    if "IQTREE_PATH" in os.environ:
        return os.environ["IQTREE_PATH"]

    system = platform.system().lower()
    if system == "darwin":
        platform_dir = "darwin"
        exe_names = ("iqtree3", "iqtree2")
    elif system == "windows":
        platform_dir = "win32"
        exe_names = ("iqtree3.exe", "iqtree2.exe")
    else:
        platform_dir = "linux"
        exe_names = ("iqtree3", "iqtree2")

    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            base_path = Path(sys._MEIPASS)
        else:
            base_path = Path(sys.executable).parent

        for exe_name in exe_names:
            bundled_exe = base_path / "bin" / platform_dir / exe_name
            if bundled_exe.exists():
                return str(bundled_exe)

    source_bundle_base = Path(__file__).resolve().parents[2]
    for exe_name in exe_names:
        bundled_exe = source_bundle_base / "bin" / platform_dir / exe_name
        if bundled_exe.exists():
            return str(bundled_exe)

    for executable in ("iqtree3", "iqtree2", "iqtree"):
        resolved = shutil.which(executable)
        if resolved:
            return resolved

    return "iqtree3"


def _format_process_output(stdout: str | None, stderr: str | None) -> str:
    output_parts = []
    if stderr:
        output_parts.append(f"stderr: {stderr.strip()}")
    if stdout:
        output_parts.append(f"stdout: {stdout.strip()}")
    return "\n".join(output_parts) if output_parts else "no output captured"


def run_fasttree(
    alignment_file: str,
    config: FastTreeConfig | None = None,
) -> str:
    """
    Runs FastTree on a single alignment file and returns the Newick tree string.
    Forces single-threaded execution to allow efficient parallel processing of multiple windows.

    Args:
        alignment_file: Path to the FASTA alignment file.
        config: FastTree model configuration. Defaults to FastTreeConfig() if None.

    Returns:
        Newick tree string.
    """
    if config is None:
        config = FastTreeConfig()

    # Force FastTree to use a single thread per process to avoid oversubscription
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    # Build command with options:
    # -nt: nucleotide data (DNA, not protein)
    # Additional options from config (GTR, gamma, noml)
    fasttree_executable = _get_fasttree_exe()
    cmd = [
        fasttree_executable,
        "-nt",
        "-quiet",
        *config.build_command_args(),
        alignment_file,
    ]

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


def run_iqtree(
    alignment_file: str,
    config: IQTreeConfig | None = None,
) -> str:
    """
    Runs IQ-TREE on a single alignment file and returns the Newick tree string.

    IQ-TREE writes its tree to <prefix>.treefile, so this runner creates a
    deterministic per-window prefix next to the alignment and reads that file.
    """
    if config is None:
        config = IQTreeConfig()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config.threads)

    alignment_path = Path(alignment_file)
    prefix = alignment_path.parent / f"{alignment_path.stem}.iqtree"
    iqtree_executable = _get_iqtree_exe()
    cmd = [
        iqtree_executable,
        *config.build_command_args(alignment_file, prefix),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        treefile = Path(f"{prefix}.treefile")
        if not treefile.exists():
            raise RuntimeError(
                f"IQ-TREE finished but did not produce expected tree file: {treefile}"
            )
        return treefile.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "IQ-TREE executable was not found. "
            f"Tried: {iqtree_executable}. "
            "Phylo-Movies normally uses bundled iqtree3; set IQTREE_PATH to a "
            "valid IQ-TREE executable only if you want to override the bundle."
        ) from exc
    except PermissionError as exc:
        raise RuntimeError(
            "IQ-TREE executable is not runnable. "
            f"Tried: {iqtree_executable}. "
            "Check that the bundled binary exists and has executable permissions, "
            "or set IQTREE_PATH to a runnable IQ-TREE executable."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not start IQ-TREE executable {iqtree_executable}: {exc}"
        ) from exc
    except subprocess.CalledProcessError as e:
        output = _format_process_output(e.stdout, e.stderr)
        raise RuntimeError(
            f"IQ-TREE failed on {alignment_file} using {iqtree_executable} "
            f"with exit code {e.returncode}.\n{output}"
        ) from e


def load_alignment(
    source: str | Path,
    content: str | None = None,
) -> MultipleSeqAlignment:
    """
    Load a multiple sequence alignment from a file or string content.

    This function provides flexibility for both CLI (file-based) and
    webservice (in-memory content) use cases.

    Args:
        source: Path to alignment file (used if content is None).
        content: Raw MSA content as a string. If provided, source is ignored
                 for reading but may be used for format detection.

    Returns:
        Loaded MultipleSeqAlignment object.

    Raises:
        RuntimeError: If the alignment cannot be parsed.
    """
    from io import StringIO

    try:
        if content is not None:
            # Parse directly from string content (webservice path)
            return cast(MultipleSeqAlignment, AlignIO.read(StringIO(content), "fasta"))
        else:
            # Read from file path (CLI path)
            return cast(
                MultipleSeqAlignment,
                AlignIO.read(source, "fasta"),
            )
    except Exception as e:
        raise RuntimeError(f"Error reading alignment: {e}")


def run_pipeline(
    input_file: str | None,
    output_directory: str,
    window_size: int,
    step_size: int,
    output_tree_filename: Optional[str] = None,
    fasttree_config: TreeInferenceConfig | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    stage_progress_callback: Optional[Callable[[float, str], None]] = None,
    msa_content: str | None = None,
) -> PipelineResult:
    """
    Main function to run the analysis pipeline.

    Ensures taxa consistency across all trees by pre-scanning all windows
    and only keeping taxa that have valid (non-gap/ambiguous) data in ALL windows.

    Args:
        input_file: Path to the input MSA file (FASTA format). Can be None if
                    msa_content is provided.
        output_directory: Directory to store output files.
        window_size: Size of sliding window.
        step_size: Step size for sliding window.
        output_tree_filename: Optional custom filename for the output tree file.
        fasttree_config: Tree inference configuration. Defaults to IQTreeConfig().
        progress_callback: Optional callback for logging messages (for webapp integration).
        stage_progress_callback: Optional structured progress callback receiving
            pipeline-local percent and message.
        msa_content: Raw MSA content as a string. If provided, input_file is not read
                     (webservice use case - avoids unnecessary disk I/O).

    Returns:
        PipelineResult with path to tree file and info about dropped taxa.
    """
    if fasttree_config is None:
        fasttree_config = IQTreeConfig()

    def log(msg: str) -> None:
        """Log message to both logger and optional callback."""
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    def report_stage(pct: float, msg: str) -> None:
        if stage_progress_callback:
            stage_progress_callback(pct, msg)

    # --- Setup Paths and Directories ---
    output_dir = Path(output_directory)
    windows_dir = output_dir / "windows"
    trees_dir = output_dir / "trees"

    windows_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load alignment and get all taxa ---
    # Supports both file path (CLI) and in-memory content (webservice)
    alignment = load_alignment(input_file or "", content=msa_content)

    alignment_length = alignment.get_alignment_length()
    all_taxa_ids: list[str] = [record.id for record in alignment]
    total_taxa = len(all_taxa_ids)

    log(f"Alignment: {total_taxa} taxa, {alignment_length} positions")
    report_stage(5, f"Loaded alignment: {total_taxa} taxa, {alignment_length} positions")

    # --- Step 2: Pre-scan all windows to find taxa valid in ALL windows ---
    # This ensures all trees have the exact same taxa set
    windows_list = list(
        create_windows_from_parameters(window_size, step_size, alignment_length)
    )

    num_windows = len(windows_list)
    log(f"Pre-scanning {num_windows} windows for taxa consistency...")
    report_stage(10, f"Pre-scanning {num_windows} windows for taxa consistency...")

    # Track which taxa are invalid in which windows
    invalid_taxa_windows = scan_invalid_taxa(alignment, windows_list)

    # Determine which taxa to keep (valid in ALL windows)
    dropped_taxa = sorted(invalid_taxa_windows.keys())
    kept_taxa_ids: set[str] = set(all_taxa_ids) - set(dropped_taxa)
    kept_taxa = len(kept_taxa_ids)

    if dropped_taxa:
        log(f"Dropping {len(dropped_taxa)} taxa (gaps/ambiguous in some windows)")
        logger.debug(
            f"Dropped taxa: {dropped_taxa[:10]}{'...' if len(dropped_taxa) > 10 else ''}"
        )

    if kept_taxa < 3:
        raise RuntimeError(
            f"Insufficient taxa for tree inference. Only {kept_taxa} taxa are valid "
            f"in all windows. Need at least 3 taxa. Consider using a smaller window size "
            f"or checking your alignment for excessive gaps."
        )

    # --- Step 3: Generate filtered windowed alignments ---
    log(f"Generating {num_windows} filtered window alignments...")
    report_stage(20, f"Generating {num_windows} filtered window alignments...")

    generate_filtered_window_alignments(
        alignment=alignment,
        windows_list=windows_list,
        kept_taxa_ids=kept_taxa_ids,
        windows_dir=windows_dir,
    )

    # --- Step 4: Infer Trees for each Window ---
    log(
        f"Running tree inference ({fasttree_config.description}) "
        f"on {num_windows} windows..."
    )
    report_stage(
        25,
        f"Running tree inference ({fasttree_config.description}) on {num_windows} windows...",
    )

    def report_tree_inference(completed: int, total: int, alignment_file: str) -> None:
        if total <= 0:
            return
        pct = 25 + (completed / total) * 70
        report_stage(
            pct,
            f"Inferred tree {completed}/{total} from {Path(alignment_file).name}",
        )

    try:
        master_tree_file = infer_trees_parallel(
            windows_dir=windows_dir,
            trees_dir=trees_dir,
            output_tree_filename=output_tree_filename,
            config=fasttree_config,
            progress_callback=report_tree_inference,
        )
    except RuntimeError as e:
        logger.error(f"Error during tree inference: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

    log(f"Complete: {num_windows} trees with {kept_taxa} taxa each")
    report_stage(100, f"Complete: {num_windows} trees with {kept_taxa} taxa each")

    return PipelineResult(
        tree_file_path=master_tree_file,
        total_taxa=total_taxa,
        kept_taxa=kept_taxa,
        dropped_taxa=dropped_taxa,
        dropped_taxa_reasons=invalid_taxa_windows,
        num_windows=num_windows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a sliding window tree inference analysis.",
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

    # Model options
    model_group = parser.add_argument_group("model options")
    model_group.add_argument(
        "--engine",
        choices=("fasttree", "iqtree"),
        default="iqtree",
        help="Tree inference engine to use.",
    )
    model_group.add_argument(
        "--gtr",
        dest="use_gtr",
        action="store_true",
        default=True,
        help="Use GTR (General Time Reversible) model. Most realistic for viral/nucleotide data.",
    )
    model_group.add_argument(
        "--no-gtr",
        dest="use_gtr",
        action="store_false",
        help="Disable GTR model (use JC model instead).",
    )
    model_group.add_argument(
        "--gamma",
        dest="use_gamma",
        action="store_true",
        default=True,
        help="Use gamma rate heterogeneity to account for rate variation across sites.",
    )
    model_group.add_argument(
        "--no-gamma",
        dest="use_gamma",
        action="store_false",
        help="Disable gamma rate heterogeneity.",
    )
    model_group.add_argument(
        "--fast",
        dest="fast_search",
        action="store_true",
        default=True,
        help="Use IQ-TREE fast search mode.",
    )
    model_group.add_argument(
        "--thorough",
        dest="fast_search",
        action="store_false",
        help="Disable IQ-TREE fast search mode.",
    )
    model_group.add_argument(
        "--noml",
        dest="no_ml",
        action="store_true",
        default=True,
        help="Disable ML NNI moves to produce fully bifurcating trees (prevents polytomies).",
    )
    model_group.add_argument(
        "--ml",
        dest="no_ml",
        action="store_false",
        help="Enable ML NNI moves (may create polytomies by collapsing low-support branches).",
    )
    model_group.add_argument(
        "--support-mode",
        choices=("none", "ufboot", "sh_alrt", "sh_alrt_ufboot"),
        default="none",
        help="IQ-TREE branch support mode for support-annotated window trees.",
    )
    model_group.add_argument(
        "--ufboot-replicates",
        type=int,
        default=1000,
        help="Number of IQ-TREE ultrafast bootstrap replicates when support mode includes UFBoot.",
    )
    model_group.add_argument(
        "--sh-alrt-replicates",
        type=int,
        default=1000,
        help="Number of IQ-TREE SH-aLRT replicates when support mode includes SH-aLRT.",
    )
    model_group.add_argument(
        "--bnni",
        action="store_true",
        help="Run IQ-TREE bootstrap NNI optimization when support mode is enabled.",
    )

    args = parser.parse_args()

    # Create tree inference configuration from CLI arguments
    fasttree_config: TreeInferenceConfig
    if args.engine == "iqtree":
        fasttree_config = IQTreeConfig(
            use_gtr=args.use_gtr,
            use_gamma=args.use_gamma,
            fast_search=args.fast_search,
            support_mode=args.support_mode,
            ufboot_replicates=args.ufboot_replicates,
            sh_alrt_replicates=args.sh_alrt_replicates,
            bnni=args.bnni,
        )
    else:
        fasttree_config = FastTreeConfig(
            use_gtr=args.use_gtr,
            use_gamma=args.use_gamma,
            no_ml=args.no_ml,
        )

    # For CLI, add a print callback
    def cli_progress(msg: str) -> None:
        print(f"[pipeline] {msg}")

    result = run_pipeline(
        input_file=args.input,
        output_directory=args.output_directory,
        window_size=args.window_size,
        step_size=args.step_size,
        fasttree_config=fasttree_config,
        progress_callback=cli_progress,
    )

    if result.has_dropped_taxa:
        print(f"\nWARNING: {len(result.dropped_taxa)} taxa were dropped:")
        for taxa_id in result.dropped_taxa:
            windows = result.dropped_taxa_reasons.get(taxa_id, [])
            print(f"  - {taxa_id}: invalid in {len(windows)} window(s)")
