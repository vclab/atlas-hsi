"""Pytest config for the atlas-hsi suite.

Registers the ``gpu`` marker and auto-skips GPU-marked tests when there is no CUDA
device or the atlas_gsplat kernel cannot be imported, so ``pytest tests/`` stays green
on CPU-only CI yet runs the full surface on a GPU box. Also puts ``tests/`` on sys.path
so sibling helper modules (``harness``, ``testcases``) import in every pytest mode.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a CUDA GPU and the atlas_gsplat kernel")


def pytest_collection_modifyitems(config, items):
    try:
        import harness
        ready = harness.gpu_ready()
    except Exception:
        ready = False
    if ready:
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU / atlas_gsplat kernel available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
