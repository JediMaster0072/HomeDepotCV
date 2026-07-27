import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import setup_script_paths
from common.sku_review import is_na_expected_sku, is_reviewed_expected_sku, normalize_sku_digits

_, PROJECT_ROOT, _, _ = setup_script_paths(__file__)
DEFAULT_DATASET = PROJECT_ROOT / "Golden_Dataset_overhead_eval_expected_sku"
DEFAULT_TRUTH_CSV = PROJECT_ROOT / "research_outputs" / "golden_dataset_local_tests" / "golden_sku_truth.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync human-reviewed expected_sku values from golden_sku_truth.csv into copied golden JSONs.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Expected-SKU golden dataset folder.")
    parser.add_argument("--truth-csv", type=Path, default=DEFAULT_TRUTH_CSV, help="CSV containing expected_sku values.")
    parser.add_argument("--dry-run", action="store_true", help="Report updates without writing JSON files.")
    return parser.parse_args()


def normalize_sku(value: str) -> str:
    return normalize_sku_digits(value)


def load_truth_rows(truth_csv: Path) -> dict[tuple[str, int, str], dict]:
    rows = {}

    with truth_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw_expected = str(row.get("expected_sku", "")).strip()
            notes = row.get("notes", "").strip()

            if is_na_expected_sku(raw_expected):
                if not notes:
                    continue
                expected_sku = "N/A"
                review_status = row.get("review_status", "").strip() or "not_applicable"
            else:
                expected_sku = normalize_sku(raw_expected)
                if not expected_sku:
                    continue
                review_status = row.get("review_status", "").strip() or "reviewed"

            key = (row["json_file"], int(row["shape_idx"]), row["label"])
            rows[key] = {
                "expected_sku": expected_sku,
                "review_status": review_status,
                "reviewer": row.get("reviewer", "").strip(),
                "notes": notes,
            }

    return rows


def update_dataset(dataset: Path, truth_rows: dict[tuple[str, int, str], dict], dry_run: bool) -> tuple[int, int, int]:
    updated_shapes = 0
    updated_files = 0
    missing = 0

    by_file: dict[str, list[tuple[tuple[str, int, str], dict]]] = {}
    for key, value in truth_rows.items():
        by_file.setdefault(key[0], []).append((key, value))

    for json_file, file_rows in sorted(by_file.items()):
        json_path = dataset / json_file

        if not json_path.exists():
            missing += len(file_rows)
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        shapes = data.get("shapes", [])
        file_updated = False

        for (file_name, shape_idx, label), values in file_rows:
            if shape_idx >= len(shapes):
                missing += 1
                continue

            shape = shapes[shape_idx]

            if shape.get("label") != label:
                missing += 1
                continue

            shape["expected_sku"] = values["expected_sku"]
            shape["expected_sku_review_status"] = values["review_status"]
            shape["expected_sku_source"] = "golden_sku_truth.csv"

            if values["reviewer"]:
                shape["expected_sku_reviewer"] = values["reviewer"]

            if values["notes"]:
                shape["expected_sku_notes"] = values["notes"]

            updated_shapes += 1
            file_updated = True

        if file_updated:
            updated_files += 1

            if not dry_run:
                json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return updated_files, updated_shapes, missing


def main():
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    if not args.truth_csv.exists():
        raise FileNotFoundError(f"Truth CSV not found: {args.truth_csv}")

    truth_rows = load_truth_rows(args.truth_csv)
    updated_files, updated_shapes, missing = update_dataset(args.dataset, truth_rows, args.dry_run)
    mode = "Would update" if args.dry_run else "Updated"
    print(f"Reviewed expected_sku rows found: {len(truth_rows)}")
    print(f"{mode} {updated_shapes} shapes across {updated_files} JSON files")
    print(f"Rows missing matching JSON shape: {missing}")


if __name__ == "__main__":
    main()
