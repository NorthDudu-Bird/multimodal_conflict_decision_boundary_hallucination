"""Utilities for loading project-local Python dependencies without altering the base conda env."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / "vendor" / "cv_proj_sitepackages"


def ensure_local_deps() -> str | None:
    """Append the project-local dependency overlay to sys.path if it exists.

    Appending keeps the conda environment packages first, so existing torch/numpy
    remain authoritative while missing VLM dependencies are resolved from the
    project-local overlay.
    """
    if not LOCAL_DEPS.exists():
        return None

    deps_path = str(LOCAL_DEPS)
    if deps_path not in sys.path:
        sys.path.append(deps_path)
    return deps_path
