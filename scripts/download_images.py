#!/usr/bin/env python3
"""Download or restore HomeDepotCV images without shipping them in git.

Images are intentionally omitted from the repo (~8.4 GB). This script restores
them on another machine using either:

1. Azure Blob URLs embedded in the image manifest (Golden Dataset / labelling), or
2. Extraction from the original local archive (HomeDepotCV 2.zip), when provided.

Usage examples:
  # Restore everything that has a remote URL
  python scripts/download_images.py --urls-only

  # Extract all images from the original zip on this machine
  python scripts/download_images.py --zip "../HomeDepotCV 2.zip"

  # Prefer URLs, fall back to zip for the rest
  python scripts/download_images.py --zip "../HomeDepotCV 2.zip"

  # Only a subdirectory
  python scripts/download_images.py --prefix Golden_Dataset_overhead_eval --urls-only
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "scripts" / "image_data" / "image_manifest.jsonl"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def download_one(url: str, dest: Path, timeout: float = 60.0) -> tuple[str, bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return (str(dest), True, "exists")
    req = Request(url, headers={"User-Agent": "HomeDepotCV-image-downloader/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
        return (str(dest), True, "downloaded")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return (str(dest), False, str(exc))


def extract_from_zip(
    zf: zipfile.ZipFile, member: str, dest: Path, expected_size: int | None = None
) -> tuple[str, bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if expected_size is None or dest.stat().st_size == expected_size:
            return (str(dest), True, "exists")
    try:
        with zf.open(member) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
        return (str(dest), True, "extracted")
    except KeyError:
        return (str(dest), False, f"missing zip member: {member}")
    except OSError as exc:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return (str(dest), False, str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to image_manifest.jsonl",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repo root where image relative paths are written",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Optional path to the original HomeDepotCV zip for offline restore",
    )
    parser.add_argument(
        "--urls-only",
        action="store_true",
        help="Only download entries that have a remote URL",
    )
    parser.add_argument(
        "--zip-only",
        action="store_true",
        help="Only extract from --zip (ignore URLs)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only restore images whose relative path starts with this prefix",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel URL download workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    entries = load_manifest(args.manifest)
    if args.prefix:
        entries = [e for e in entries if e["path"].startswith(args.prefix)]

    url_jobs: list[dict[str, Any]] = []
    zip_jobs: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for e in entries:
        has_url = bool(e.get("url"))
        if args.zip_only:
            if args.zip is None:
                unresolved.append(e)
            else:
                zip_jobs.append(e)
            continue
        if has_url and not args.zip_only:
            url_jobs.append(e)
            continue
        if args.zip is not None and not args.urls_only:
            zip_jobs.append(e)
            continue
        unresolved.append(e)
    print(f"Manifest entries selected: {len(entries)}")
    print(f"  URL downloads: {len(url_jobs)}")
    print(f"  Zip extracts:  {len(zip_jobs)}")
    print(f"  Unresolved:    {len(unresolved)}")

    if args.dry_run:
        for e in url_jobs[:10]:
            print(f"  [url] {e['path']}")
        if len(url_jobs) > 10:
            print(f"  ... +{len(url_jobs) - 10} more URL jobs")
        for e in zip_jobs[:10]:
            print(f"  [zip] {e['path']}")
        if len(zip_jobs) > 10:
            print(f"  ... +{len(zip_jobs) - 10} more zip jobs")
        return 0

    ok = fail = skip = 0

    if url_jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(download_one, e["url"], args.root / e["path"]): e
                for e in url_jobs
            }
            for fut in as_completed(futures):
                dest, success, msg = fut.result()
                if success and msg == "exists":
                    skip += 1
                elif success:
                    ok += 1
                else:
                    fail += 1
                    print(f"FAIL {dest}: {msg}", file=sys.stderr)

    if zip_jobs:
        if args.zip is None or not args.zip.exists():
            print(f"Zip not found: {args.zip}", file=sys.stderr)
            return 1
        with zipfile.ZipFile(args.zip) as zf:
            for e in zip_jobs:
                dest, success, msg = extract_from_zip(
                    zf, e["zip_member"], args.root / e["path"], e.get("size")
                )
                if success and msg == "exists":
                    skip += 1
                elif success:
                    ok += 1
                else:
                    fail += 1
                    print(f"FAIL {dest}: {msg}", file=sys.stderr)

    print(f"Done. restored={ok} skipped_existing={skip} failed={fail}")
    if unresolved:
        print(
            f"{len(unresolved)} images have no URL and no --zip was usable. "
            "Pass --zip path/to/HomeDepotCV 2.zip to restore research/stratified outputs.",
            file=sys.stderr,
        )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
