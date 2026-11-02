"""Shared pytest setup: MLX tests run only where MLX can initialise Metal."""

from __future__ import annotations

import functools
import subprocess
import sys

import pytest


@functools.lru_cache(maxsize=1)
def mlx_usable() -> bool:
    """Probe MLX in a child process: where Metal is unavailable (CI, sandboxes) importing it
    aborts the interpreter, which would otherwise take the whole test run down with it."""
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import mlx.core as mx; mx.array([1]).item()"],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "mlx: needs MLX with a working Metal device (Apple Silicon)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    marked = [item for item in items if "mlx" in item.keywords]
    if marked and not mlx_usable():
        skip = pytest.mark.skip(reason="MLX cannot initialise Metal here (no Apple GPU access)")
        for item in marked:
            item.add_marker(skip)
