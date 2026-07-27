"""Generate the human-review CSV for camera_cart EmptyItem regions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import setup_script_paths

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)

from empty_shelf_review_utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OVERLAY_DIR,
    DEFAULT_TEMPORAL_CSV,
    DEFAULT_TEMPORAL_DIR,
    DEFAULT_TRUTH_CSV,
    build_truth_rows,
    load_truth_rows,
    save_truth_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate empty-region review CSV template.")
    parser.add_argument("--temporal-csv", type=Path, default=DEFAULT_TEMPORAL_CSV)
    parser.add_argument("--temporal-dir", type=Path, default=DEFAULT_TEMPORAL_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing truth CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_csv.exists() and not args.force:
        rows = load_truth_rows(args.output_csv)
        print(f"Truth CSV already exists: {args.output_csv} ({len(rows)} rows)")
        print("Use --force to regenerate from temporal_data.")
        return

    rows = build_truth_rows(args.temporal_csv, args.temporal_dir, args.overlay_dir)
    save_truth_rows(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
