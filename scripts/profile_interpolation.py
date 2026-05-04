#!/usr/bin/env python3
"""Profile BranchArchitect pair interpolation on real tree files."""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)

DEFAULT_INPUTS = (
    "test/data/current_testfiles/focus.tree",
    "test/data/current_testfiles/small_example_cli.newick",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile interpolation over one or more newline-delimited Newick files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="Tree files to profile. Defaults to focus.tree and small_example_cli.newick.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to repeat all selected pairs.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum consecutive pairs to run per file.",
    )
    parser.add_argument(
        "--sort",
        default="cumtime",
        choices=(
            "calls",
            "cumulative",
            "cumtime",
            "file",
            "module",
            "name",
            "nfl",
            "pcalls",
            "stdname",
            "time",
            "tottime",
        ),
        help="pstats sort key.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of profiler rows to print.",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Only print wall-clock timing summary.",
    )
    return parser.parse_args()


def _load_trees(path: Path) -> list[Node]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [parse_newick(line) for line in lines]


def _iter_pairs(
    trees: list[Node], max_pairs: int | None
) -> Iterable[tuple[int, Node, Node]]:
    pair_count = len(trees) - 1
    if max_pairs is not None:
        pair_count = min(pair_count, max_pairs)
    for index in range(pair_count):
        yield index, trees[index], trees[index + 1]


def run_profile(
    paths: list[str], repeat: int, max_pairs: int | None
) -> tuple[int, int]:
    parsed_files = [(Path(path), _load_trees(Path(path))) for path in paths]
    total_pairs = 0
    total_frames = 0

    for _ in range(repeat):
        for path, trees in parsed_files:
            for pair_index, source_tree, destination_tree in _iter_pairs(
                trees, max_pairs
            ):
                result = process_tree_pair_interpolation(
                    source_tree.deep_copy(),
                    destination_tree.deep_copy(),
                    pair_index=pair_index,
                )
                total_pairs += 1
                total_frames += len(result.trees)

    return total_pairs, total_frames


def main() -> None:
    args = _parse_args()
    logging.disable(logging.CRITICAL)

    start = time.perf_counter()
    if args.no_profile:
        total_pairs, total_frames = run_profile(args.paths, args.repeat, args.max_pairs)
        stats_output = ""
    else:
        profiler = cProfile.Profile()
        total_pairs, total_frames = profiler.runcall(
            run_profile, args.paths, args.repeat, args.max_pairs
        )
        stats_stream = io.StringIO()
        pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats(
            args.sort
        ).print_stats(args.top)
        stats_output = stats_stream.getvalue()

    elapsed = time.perf_counter() - start
    print(
        f"pairs={total_pairs} frames={total_frames} repeat={args.repeat} "
        f"seconds={elapsed:.6f}"
    )
    if stats_output:
        print(stats_output)


if __name__ == "__main__":
    main()
