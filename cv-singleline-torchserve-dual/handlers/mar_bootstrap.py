"""Shared MAR unpack helpers for dual TorchServe handlers."""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

_PIPELINE_MARKERS = (
    "__init__.py",
    "stage1_detection.py",
    "stage2_segmentation.py",
    "pipeline_config.py",
    "label_record.py",
)


def _extract_zips(model_dir: Path) -> None:
    for zpath in sorted(model_dir.glob("*.zip")):
        print(f"[mar_bootstrap] extracting {zpath.name} -> {model_dir}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(model_dir)


def _repair_flattened_pipeline(model_dir: Path) -> None:
    """If archiver flattened service_pipeline_gpu/*, recreate the package dir."""
    pkg = model_dir / "service_pipeline_gpu"
    if pkg.is_dir() and (pkg / "__init__.py").is_file():
        return

    flat = [name for name in _PIPELINE_MARKERS if (model_dir / name).is_file()]
    if not flat:
        return

    print(f"[mar_bootstrap] repairing flattened pipeline files: {flat}")
    pkg.mkdir(parents=True, exist_ok=True)
    for name in _PIPELINE_MARKERS:
        src = model_dir / name
        if src.is_file():
            shutil.move(str(src), str(pkg / name))


def prepare_model_dir(model_dir: Path) -> Path:
    """Extract zip extras, repair layout, and put model_dir on sys.path."""
    model_dir = model_dir.resolve()
    _extract_zips(model_dir)
    _repair_flattened_pipeline(model_dir)

    unpack = str(model_dir)
    while unpack in sys.path:
        sys.path.remove(unpack)
    sys.path.insert(0, unpack)

    pkg = model_dir / "service_pipeline_gpu"
    cfg = model_dir / "common_config_gpu.py"
    print(
        f"[mar_bootstrap] ready model_dir={model_dir} "
        f"pipeline_pkg={pkg.is_dir()} common_config={cfg.is_file()} "
        f"entries={sorted(p.name for p in model_dir.iterdir())[:40]}"
    )
    if not pkg.is_dir():
        raise FileNotFoundError(
            f"service_pipeline_gpu/ missing under {model_dir}. "
            f"entries={sorted(p.name for p in model_dir.iterdir())}"
        )
    if not cfg.is_file():
        raise FileNotFoundError(
            f"common_config_gpu.py missing under {model_dir}. "
            f"entries={sorted(p.name for p in model_dir.iterdir())}"
        )
    return model_dir
