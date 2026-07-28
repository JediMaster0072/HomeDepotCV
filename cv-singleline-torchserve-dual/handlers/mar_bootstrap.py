"""Shared MAR unpack helpers for dual TorchServe handlers."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def prepare_model_dir(model_dir: Path) -> Path:
    """Ensure model_dir is on sys.path and extract any *.zip extras in place."""
    model_dir = model_dir.resolve()
    for zpath in sorted(model_dir.glob("*.zip")):
        print(f"[mar_bootstrap] extracting {zpath.name} -> {model_dir}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(model_dir)

    unpack = str(model_dir)
    while unpack in sys.path:
        sys.path.remove(unpack)
    sys.path.insert(0, unpack)
    return model_dir
