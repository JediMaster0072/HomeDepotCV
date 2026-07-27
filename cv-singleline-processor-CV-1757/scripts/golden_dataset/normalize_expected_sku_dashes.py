"""
Remove dashes and other non-digits from saved expected_sku values.

Updates golden_sku_truth.csv and syncs cleaned values into dataset JSONs.
Preserves N/A rows (notes required).

Usage:
  python scripts/golden_dataset/normalize_expected_sku_dashes.py
  python scripts/golden_dataset/normalize_expected_sku_dashes.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import setup_script_paths
from common.sku_review import is_na_expected_sku, normalize_sku_digits

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)

DEFAULT_TRUTH_CSV = PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "golden_sku_truth.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip dashes from saved expected_sku values.")
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH_CSV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-sync-json", action="store_true")
    return parser.parse_args()


def clean_expected_sku(raw: str) -> str:
    text = str(raw or "").strip()
    if not text or is_na_expected_sku(text):
        return text
    return normalize_sku_digits(text)


def main() -> None:
    args = parse_args()
    if not args.truth_csv.exists():
        raise FileNotFoundError(f"Truth CSV not found: {args.truth_csv}")

    with args.truth_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    changed = 0
    for row in rows:
        raw = str(row.get("expected_sku", "") or "").strip()
        if not raw or is_na_expected_sku(raw):
            continue
        cleaned = clean_expected_sku(raw)
        if cleaned and cleaned != raw:
            print(f"  {raw} -> {cleaned}  ({row.get('region_key', '')})")
            row["expected_sku"] = cleaned
            changed += 1

    print(f"Rows with dashes/non-digits cleaned: {changed}")

    if changed and not args.dry_run:
        with args.truth_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.truth_csv}")

    if not args.no_sync_json and changed and not args.dry_run:
        import subprocess

        sync_script = Path(__file__).resolve().parent / "sync_expected_sku_from_truth_csv.py"
        subprocess.run([sys.executable, str(sync_script), "--truth-csv", str(args.truth_csv)], check=True)
    elif changed and args.dry_run:
        print("Dry run — no files written.")
    elif not changed:
        print("All saved expected_sku values are already digits-only (or N/A).")


if __name__ == "__main__":
    main()
