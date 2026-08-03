"""Subprocess wrapper for running SatuTe against a fixed tree + alignment."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# SatuTe lives in the iq-tree-satute fork, a sibling checkout of this repo — the
# bundled iqtree3 in bin/ is a stock build and does not have the --satute flag.
# SATUTE_IQTREE_PATH overrides this for other machines/CI.
_DEFAULT_SATUTE_IQTREE_PATH = str(
    Path.home() / "Projects" / "iq-tree-satute" / "build" / "iqtree3"
)


def get_satute_iqtree_exe() -> str:
    """Resolve the path to a SatuTe-capable iqtree3 build.

    Priority: SATUTE_IQTREE_PATH env var > sibling iq-tree-satute checkout > PATH.
    """
    if "SATUTE_IQTREE_PATH" in os.environ:
        return os.environ["SATUTE_IQTREE_PATH"]

    if Path(_DEFAULT_SATUTE_IQTREE_PATH).exists():
        return _DEFAULT_SATUTE_IQTREE_PATH

    resolved = shutil.which("iqtree3")
    if resolved:
        return resolved

    return _DEFAULT_SATUTE_IQTREE_PATH


def _format_process_output(stdout: str | None, stderr: str | None) -> str:
    output_parts = []
    if stderr:
        output_parts.append(f"stderr: {stderr.strip()}")
    if stdout:
        output_parts.append(f"stdout: {stdout.strip()}")
    return "\n".join(output_parts) if output_parts else "no output captured"


@dataclass
class SatuteRunResult:
    prefix: Path
    stat_file: Path
    stdout: str
    stderr: str
    site_file: Path | None = None


def run_satute(
    alignment_file: str | Path,
    tree_file: str | Path,
    workdir: str | Path,
    alpha: float = 0.05,
    model: str | None = None,
    threads: int | None = None,
    write_sites: bool = False,
    sites_max: int = 0,
) -> SatuteRunResult:
    """Run SatuTe on a fixed tree topology against an alignment.

    Uses IQ-TREE's `-te` (fixed user tree) so the branch under test is the same
    topology phylo-movies already displays, plus `-blfix` so SatuTe's reported
    branch lengths are identical to the input tree's (no drift between what's
    shown and what's tested). Model is re-estimated by ModelFinder if omitted.

    write_sites additionally requests <prefix>.sat.site (per-alignment-site
    contributions), which sliding-window saturation is aggregated from.
    sites_max caps how many branches that file covers (highest satP first);
    without a cap the whole-tree export is ~507 MB on the 1871-taxon Tree of
    Life, versus ~12 MB at 200 branches.
    """
    # SatuTe evaluates every branch under both formulas and each rate
    # category, so it is the dominant cost on large trees and scales with
    # cores. Defaulting to 2 left most of the machine idle; leave a couple of
    # cores for the rest of the app instead.
    if threads is None:
        threads = max(1, min(8, (os.cpu_count() or 2) - 2))

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    prefix = workdir_path / "satute"

    satute_iqtree_exe = get_satute_iqtree_exe()
    cmd = [
        satute_iqtree_exe,
        "-s",
        str(alignment_file),
        "-te",
        str(tree_file),
        "--satute",
        "--satute-alpha",
        str(alpha),
        "-blfix",
        "-pre",
        str(prefix),
        "-nt",
        str(threads),
        "-redo",
    ]
    if write_sites:
        cmd.append("--satute-sites")
        if sites_max > 0:
            cmd.extend(["--satute-sites-max", str(sites_max)])
    if model:
        cmd.extend(["-m", model])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "SatuTe-capable iqtree3 executable was not found. "
            f"Tried: {satute_iqtree_exe}. "
            "Set SATUTE_IQTREE_PATH to a build of the iq-tree-satute fork "
            "(needs the --satute flag; the bundled phylo-movies iqtree3 does not have it)."
        ) from exc
    except PermissionError as exc:
        raise RuntimeError(
            f"SatuTe iqtree3 executable is not runnable: {satute_iqtree_exe}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not start SatuTe iqtree3 executable {satute_iqtree_exe}: {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = _format_process_output(exc.stdout, exc.stderr)
        raise RuntimeError(
            f"SatuTe failed on {alignment_file} / {tree_file} "
            f"using {satute_iqtree_exe} with exit code {exc.returncode}.\n{output}"
        ) from exc

    stat_file = Path(f"{prefix}.sat.stat")
    if not stat_file.exists():
        output = _format_process_output(result.stdout, result.stderr)
        raise RuntimeError(
            f"SatuTe finished but did not produce the expected stats file: "
            f"{stat_file}\n{output}"
        )

    site_file = Path(f"{prefix}.sat.site")
    return SatuteRunResult(
        prefix=prefix,
        stat_file=stat_file,
        stdout=result.stdout,
        stderr=result.stderr,
        site_file=site_file if site_file.exists() else None,
    )
