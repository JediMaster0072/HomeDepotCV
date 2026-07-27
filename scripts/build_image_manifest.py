#!/usr/bin/env python3
"""Rebuild scripts/image_data/image_manifest.jsonl from a local HomeDepotCV zip.

Useful if the zip layout changes or you need to refresh Azure URLs from
annotation JSON files.

  python scripts/build_image_manifest.py --zip "../HomeDepotCV 2.zip"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scripts" / "image_data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".ico"}
SKIP_PREFIXES = (
    "HomeDepotCV/.venv/",
    "HomeDepotCV/.git/",
    "HomeDepotCV/__pycache__/",
    "__MACOSX/",
)
url_re = re.compile(r"https?://[^\s'\"<>]+", re.I)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if not args.zip.exists():
        print(f"Missing zip: {args.zip}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    url_map: dict[str, str] = {}
    image_entries: list[dict] = []

    with zipfile.ZipFile(args.zip) as zf:
        for info in zf.infolist():
            name = info.filename
            if any(name.startswith(p) for p in SKIP_PREFIXES) or name.endswith("/"):
                continue
            if name.endswith(".json") and info.file_size <= 2_000_000 and any(
                k in name for k in ("Golden_Dataset", "drona_jsons", "predictions_json")
            ):
                try:
                    text = zf.read(name).decode("utf-8", "ignore")
                except Exception:
                    continue
                for u in url_re.findall(text):
                    u = u.rstrip(").,;]")
                    if any(e in u.lower() for e in (".jpg", ".jpeg", ".png", ".webp")):
                        base = os.path.basename(u.split("?")[0])
                        url_map.setdefault(base, u)

        for info in zf.infolist():
            name = info.filename
            if any(name.startswith(p) for p in SKIP_PREFIXES) or name.endswith("/"):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            base = os.path.basename(name)
            if base.startswith("._"):
                continue
            rel = name[len("HomeDepotCV/") :] if name.startswith("HomeDepotCV/") else name
            entry = {
                "path": rel,
                "size": info.file_size,
                "crc32": info.CRC,
                "zip_member": name,
            }
            if base in url_map:
                entry["url"] = url_map[base]
            image_entries.append(entry)

    manifest = args.out_dir / "image_manifest.jsonl"
    with manifest.open("w") as f:
        for e in image_entries:
            f.write(json.dumps(e) + "\n")

    summary = {
        "source_zip": str(args.zip),
        "image_count": len(image_entries),
        "images_with_url": sum(1 for e in image_entries if "url" in e),
        "total_image_bytes": sum(e["size"] for e in image_entries),
    }
    (args.out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
