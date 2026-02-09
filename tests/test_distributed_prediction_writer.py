"""Test for the distributed prediction writer and reader.

This test creates a simple lightning module that returns constant predictions,
runs distributed prediction, and verifies that the reader works correctly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_distributed_prediction_writer_and_reader(tmp_path):
    """Test the distributed prediction writer and reader with a simple module.

    Runs the actual test logic in a spawned subprocess to ensure a clean CUDA
    state, since the ddp_notebook strategy uses fork-based multiprocessing which
    is incompatible with a parent process that has already initialized CUDA
    (e.g., from earlier tests in the suite).
    """
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_distributed_predict_helper.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip("Test requires at least 2 GPUs for distributed testing")
    if result.returncode != 0:
        # Print stdout/stderr for debugging
        msg = f"Subprocess failed (exit code {result.returncode})\n"
        if result.stdout:
            msg += f"--- stdout ---\n{result.stdout}\n"
        if result.stderr:
            msg += f"--- stderr ---\n{result.stderr}\n"
        pytest.fail(msg)
