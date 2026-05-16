"""Pytest collection hygiene for manual and scenario scripts."""

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure() -> None:
    os.chdir(BACKEND_ROOT)


collect_ignore = [
    "data/test_heavy_alignment.py",
    "manual_sse_msa_test.py",
    "manual_sse_test.py",
    "test_norovirus_pipeline.py",
]
