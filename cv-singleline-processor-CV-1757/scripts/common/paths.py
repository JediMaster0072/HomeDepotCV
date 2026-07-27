"""Path bootstrap helpers for scripts under scripts/."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_script_paths(from_file: str | Path) -> tuple[Path, Path, Path, Path]:
    """
    Configure import paths for a script living under scripts/<group>/.

    Returns:
        service_root, project_root, scripts_root, script_dir
    """
    file_path = Path(from_file).resolve()
    script_dir = file_path.parent
    scripts_root = file_path.parents[1]
    service_root = file_path.parents[2]
    project_root = service_root.parent

    for path in (service_root, scripts_root, script_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return service_root, project_root, scripts_root, script_dir


GOLDEN_TESTS_MARKER = "research_outputs/golden_dataset_local_tests/"


def path_for_csv(path: Path, project_root: Path) -> str:
    """Store paths relative to project root so CSV works on Mac and GPU host."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def resolve_project_data_path(raw_path: str | Path, project_root: Path) -> Path:
    """
    Resolve a crop/overlay path from CSV for the current machine.

    Handles project-relative paths and legacy absolute paths from another host.
    """
    if not raw_path:
        return Path(raw_path)

    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate

    relative = project_root / candidate
    if relative.is_file():
        return relative

    normalized = str(raw_path).replace("\\", "/")
    marker_idx = normalized.find(GOLDEN_TESTS_MARKER)
    if marker_idx >= 0:
        suffix = normalized[marker_idx + len(GOLDEN_TESTS_MARKER) :]
        remapped = project_root / "research_outputs" / "golden_dataset_local_tests" / suffix
        if remapped.is_file():
            return remapped

    legacy_idx = normalized.find("golden_dataset_local_tests/")
    if legacy_idx >= 0:
        suffix = normalized[legacy_idx + len("golden_dataset_local_tests/") :]
        remapped = project_root / "research_outputs" / "golden_dataset_local_tests" / suffix
        if remapped.is_file():
            return remapped

    return candidate
