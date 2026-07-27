#!/usr/bin/env python3
"""Extract omitted model weight files (.pt) from the original HomeDepotCV zip.

Model checkpoints are large and are not committed to git. Use this when you have
the original archive locally:

  python scripts/extract_models_from_zip.py --zip "../HomeDepotCV 2.zip"
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "scripts" / "image_data" / "model_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, help="Path to HomeDepotCV 2.zip")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Missing {args.manifest}", file=sys.stderr)
        return 1
    if not args.zip.exists():
        print(f"Missing zip: {args.zip}", file=sys.stderr)
        return 1

    models = json.loads(args.manifest.read_text())
    print(f"{len(models)} model file(s) in manifest")
    if args.dry_run:
        for m in models:
            print(f"  {m['path']} ({m['size'] / 1e6:.1f} MB)")
        return 0

    ok = skip = fail = 0
    with zipfile.ZipFile(args.zip) as zf:
        for m in models:
            dest = args.root / m["path"]
            if dest.exists() and dest.stat().st_size == m["size"]:
                skip += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(m["zip_member"]) as src, open(dest, "wb") as out:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                ok += 1
                print(f"extracted {m['path']}")
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"FAIL {m['path']}: {exc}", file=sys.stderr)
    print(f"Done. restored={ok} skipped={skip} failed={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
