#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def rel_link(from_file: Path, to_file: Path) -> str:
    return os.path.relpath(to_file.as_posix(), start=from_file.parent.as_posix())


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python merge_parts.py <manifest_path>")

    manifest_path = Path(sys.argv[1])
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    part_files = manifest["part_files"]
    target_full_path = Path(manifest["target_full_path"])
    index_path = Path(manifest["index_path"])
    root_name = manifest.get("root_name", target_full_path.parent.name)

    if not part_files:
        raise SystemExit(f"No part_files in manifest: {manifest_path}")

    full_bytes = b""
    for part in part_files:
        part_path = Path(part)
        if not part_path.exists():
            raise SystemExit(f"Missing part file: {part_path}")
        full_bytes += part_path.read_bytes()

    target_full_path.parent.mkdir(parents=True, exist_ok=True)
    old_bytes = target_full_path.read_bytes() if target_full_path.exists() else None

    if old_bytes != full_bytes:
        target_full_path.write_bytes(full_bytes)
        print(f"Updated merged file: {target_full_path}")
    else:
        print(f"Merged file unchanged: {target_full_path}")

    index_lines = [
        f"# {root_name}",
        "",
        "## Files",
        "",
        f"- [{target_full_path.name}]({rel_link(index_path, target_full_path)})",
        "",
        "## Parts",
        "",
    ]

    for p in part_files:
        p_path = Path(p)
        index_lines.append(f"- [{p_path.name}]({rel_link(index_path, p_path)})")

    index_lines.extend([
        "",
        "## Meta",
        "",
        f"- Manifest: `{manifest_path.as_posix()}`",
        f"- Part count: `{len(part_files)}`",
        f"- Encoding: `{manifest.get('encoding', 'utf-8')}`",
        "",
    ])

    new_index_text = "\n".join(index_lines) + "\n"
    old_index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else None

    if old_index_text != new_index_text:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(new_index_text, encoding="utf-8")
        print(f"Updated INDEX: {index_path}")
    else:
        print(f"INDEX unchanged: {index_path}")


if __name__ == "__main__":
    main()
