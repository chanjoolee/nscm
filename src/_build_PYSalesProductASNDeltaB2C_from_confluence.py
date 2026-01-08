#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build PYSalesProductASNDeltaB2C.py from 5 Confluence pages under "Source1 V2 PYSalesProductASNDeltaB2C".

Usage:
  export CONFLUENCE_BASE_URL="https://gregrechan-1736411991864.atlassian.net"
  export CONFLUENCE_EMAIL="your_email@example.com"
  export CONFLUENCE_API_TOKEN="<your_api_token>"
  python nscm/src/_build_PYSalesProductASNDeltaB2C_from_confluence.py

This script fetches storage XHTML for each page and extracts code blocks
contained in <ac:structured-macro ac:name="code"> (plain-text-body CDATA).
It concatenates them in numeric order into:
  nscm/src/PYSalesProductASNDeltaB2C.py
"""
from __future__ import annotations

import os
import sys
import time
import json
import html
import re
from typing import List, Tuple

try:
    import requests  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: python-requests is required. Try: pip install requests", file=sys.stderr)
    raise

BASE_URL = os.getenv("CONFLUENCE_BASE_URL", "https://gregrechan-1736411991864.atlassian.net").rstrip("/")
WIKI_BASE = BASE_URL + "/wiki"
EMAIL = os.getenv("CONFLUENCE_EMAIL")
TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

# Order: 01 → 05
PAGE_IDS = [
    124977153,  # 01. Source1 V2 PYSalesProductASNDeltaB2C
    124977160,  # 02. Source1 V2 PYSalesProductASNDeltaB2C
    125075467,  # 03. Source1 V2 PYSalesProductASNDeltaB2C
    124977181,  # 04. Source1 VYSalesProductASNDeltaB2C
    125075489,  # 05. Source1 VYSalesProductASNDeltaB2C
]

OUT_PATH = os.path.join(os.getcwd(), "nscm", "src", "PYSalesProductASNDeltaB2C.py")

CODE_MACRO_RE = re.compile(
    r"<ac:structured-macro[^>]*ac:name=\"code\"[^>]*>.*?<ac:plain-text-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-body>.*?</ac:structured-macro>",
    re.DOTALL | re.IGNORECASE,
)
PRE_TAG_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


def _auth_ok() -> bool:
    return bool(EMAIL and TOKEN)


def fetch_storage_html(page_id: int) -> Tuple[str, str, int]:
    url = f"{WIKI_BASE}/rest/api/content/{page_id}?expand=body.storage,version,title"
    resp = requests.get(url, auth=(EMAIL, TOKEN), headers={"Accept": "application/json"})
    try:
        resp.raise_for_status()
    except Exception:
        print(f"ERROR: GET {url} -> {resp.status_code} {resp.text[:2000]}", file=sys.stderr)
        raise
    data = resp.json()
    storage_html = data.get("body", {}).get("storage", {}).get("value", "")
    title = data.get("title", str(page_id))
    ver = int(data.get("version", {}).get("number", 0) or 0)
    return storage_html, title, ver


def extract_code_blocks(storage_html: str) -> List[str]:
    blocks: List[str] = []
    # 1) Confluence code macro
    for m in CODE_MACRO_RE.finditer(storage_html):
        blocks.append(m.group(1))
    # 2) Fallback: <pre>...</pre>
    if not blocks:
        for m in PRE_TAG_RE.finditer(storage_html):
            text = TAG_RE.sub("", m.group(1))
            blocks.append(html.unescape(text))
    return blocks


def build_file() -> None:
    if not _auth_ok():
        print("ERROR: Please set CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN environment variables.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    pieces: List[str] = []
    banner = (
        "# -*- coding: utf-8 -*-\n"
        "# Auto-generated from Confluence pages by _build_PYSalesProductASNDeltaB2C_from_confluence.py\n"
        f"# Source pages: {', '.join(map(str, PAGE_IDS))}\n"
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    pieces.append(banner)

    for idx, pid in enumerate(PAGE_IDS, start=1):
        storage_html, title, ver = fetch_storage_html(pid)
        blocks = extract_code_blocks(storage_html)
        if not blocks:
            print(f"WARN: No code block found in page {pid} ({title}). Skipping.")
            continue
        header = (
            "\n\n" + "#" * 120 + "\n" +
            f"# Begin Confluence Page {idx}/{len(PAGE_IDS)} — ID {pid} — {title} (v{ver})\n" +
            "#" * 120 + "\n\n"
        )
        pieces.append(header)
        # Some pages may have multiple code blocks — concatenate in order
        for bi, code in enumerate(blocks, start=1):
            # Normalize line endings
            code = code.replace('\r\n', '\n').replace('\r', '\n')
            pieces.append(code)
            if not code.endswith("\n"):
                pieces.append("\n")

    content = "".join(pieces)

    # Light sanity check: very small output probably means auth/parse failure
    if len(content) < 2000:
        print("ERROR: Output content looks too small (<2KB). Aborting write.", file=sys.stderr)
        sys.exit(3)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Wrote {len(content):,} bytes to {OUT_PATH}")


if __name__ == "__main__":
    try:
        build_file()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
