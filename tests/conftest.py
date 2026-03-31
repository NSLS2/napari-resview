"""Pytest configuration for napari-resview tests."""

from __future__ import annotations

import os
import sys

import pytest

# Speed up pytest on Windows by setting stricter collection patterns
collect_ignore = []


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "windows_skip: marks tests to skip on Windows",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip slow tests on Windows CI."""
    if sys.platform == "win32" and os.environ.get("CI"):
        skip_on_windows = pytest.mark.skip(
            reason="Skipped on Windows CI for speed"
        )
        for item in items:
            if "windows_skip" in item.keywords:
                item.add_marker(skip_on_windows)
