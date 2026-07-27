#!/usr/bin/env python3
"""Regenerate the line-number table of contents in LOCAL_TESTING_GUIDE.md."""

from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "LOCAL_TESTING_GUIDE.md"
TOC_START = "<!-- toc -->"
TOC_END = "<!-- /toc -->"


def slugify(title: str) -> str:
    slug = title.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug


def build_toc(lines: list[str]) -> str:
    rows: list[tuple[int, str, str]] = []

    for index, line in enumerate(lines, start=1):
        if line.startswith("# "):
            title = line[2:].strip()
            rows.append((index, title, slugify(title)))
        elif line.startswith("## "):
            title = line[3:].strip()
            if title == "Table of Contents":
                continue
            rows.append((index, title, slugify(title)))
        elif line.startswith("### "):
            title = line[4:].strip()
            rows.append((index, title, slugify(title)))

    body = "\n".join(
        f"| L{line_no} | [{title}](#{anchor}) |"
        for line_no, title, anchor in rows
    )
    return "\n".join(
        [
            TOC_START,
            "## Table of Contents",
            "",
            "| Line | Section |",
            "| --- | --- |",
            body,
            TOC_END,
        ]
    )


def main() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    start = text.find(TOC_START)
    end = text.find(TOC_END)
    if start == -1 or end == -1:
        raise SystemExit(f"TOC markers not found in {DOC_PATH}")

    end += len(TOC_END)
    new_toc = build_toc(lines)
    updated = text[:start] + new_toc + text[end:]
    DOC_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated TOC in {DOC_PATH}")


if __name__ == "__main__":
    main()
