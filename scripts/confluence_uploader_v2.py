#!/usr/bin/env python3
"""
Confluence Q/A uploader (multi pages, links visible)

- Create QUESTION ROOT page under --parent-id
- Create ANSWER page under QUESTION ROOT
- Split txt into chunks and create PART pages under QUESTION ROOT
- Update QUESTION ROOT once with:
  - ANSWER link
  - PART links list

Fix: Use HTML <a href="...">text</a> links (more reliable on Confluence Cloud ADF conversion)


launch.json
        {
            "name": "Confluence Upload (API Token)",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/confluence_uploader_v2.py",
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": false,
            // 토큰을 launch.json에 직접 쓰기 싫으면 아래 envFile 추천
            "envFile": "${workspaceFolder}/.env.confluence",
            "args": [
                // "--file", "${workspaceFolder}/Questions/질문_o9_NA.txt",
                "--file", "C:\\workspace\\scripts\\confluence_uploader_v2.py",
                "--language", "none",
                "--max-bytes", "40000",
                "--sleep-secs", "11"
            ],
            "env": {
                "REQUESTS_CA_BUNDLE"  : "C:\\workspace\\scripts\\corp_root.cer",
                "CONFLUENCE_AUTH_MODE": "basic",
                "CONFLUENCE_BASE_URL": "https://api.atlassian.com/ex/confluence/c2edc623-2385-453a-a1b9-1148844c3872",
                "CONFLUENCE_SITE_URL": "https://gregrechan-1736411991864.atlassian.net",
                "CONFLUENCE_SPACE_KEY": "~5b552f42f220fc2d9cb5c68e",
                "CONFLUENCE_PARENT_PAGE_ID": "174883033"

            }
        }

"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple, List, Dict

import requests
from requests.auth import HTTPBasicAuth
import json

DEFAULT_MAX_BYTES = 45_000  # headroom
DEFAULT_TIMEOUT_SECS = 60

# ---------------------------
# Storage helpers
# ---------------------------

def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )

def safe_cdata(text: str) -> str:
    return text.replace("]]>", "]]]]><![CDATA[>")

def make_code_macro(text: str, language: str = "none", title: Optional[str] = None) -> str:
    parts = ['<ac:structured-macro ac:name="code">']
    if title:
        parts.append(f'<ac:parameter ac:name="title">{escape_xml(title)}</ac:parameter>')
    if language:
        parts.append(f'<ac:parameter ac:name="language">{escape_xml(language)}</ac:parameter>')
    parts.append(f"<ac:plain-text-body><![CDATA[{safe_cdata(text)}]]></ac:plain-text-body>")
    parts.append("</ac:structured-macro>")
    return "\n".join(parts)

def make_a(url: Optional[str], text: str) -> str:
    # url이 없으면 그냥 텍스트만 표시(최악의 fallback)
    if not url:
        return escape_xml(text)
    return f'<a href="{escape_xml(url)}">{escape_xml(text)}</a>'

def make_hr() -> str:
    return "<hr/>"

def make_h2(text: str) -> str:
    return f"<h2>{escape_xml(text)}</h2>"

def make_p_raw(inner_html: str) -> str:
    return f"<p>{inner_html}</p>"


# ---------------------------
# Chunking (UTF-8 safe)
# ---------------------------
def iter_utf8_chunks_(file_path: str, max_bytes: int) -> Iterator[str]:
    """
    - max_bytes 이내에서 '가능하면' 마지막 개행(b'\\n') 위치로 잘라서 chunk가 줄 단위로 끝나게 함
    - UTF-8 경계도 안전하게 맞춤
    - 각 chunk 텍스트가 최소 1개의 개행으로 끝나도록 보장(붙여넣기 편의)
    """
    data = Path(file_path).read_bytes()
    i = 0
    n = len(data)

    # 너무 작은 조각 방지용(마지막 개행이 너무 앞이면 무시)
    min_chunk = max(1024, max_bytes // 10)  # 1KB 또는 10%

    while i < n:
        end = min(i + max_bytes, n)
        window = data[i:end]

        # 1) 가능하면 마지막 개행 위치로 자르기
        nl = window.rfind(b"\n")
        if nl != -1:
            candidate_end = i + nl + 1  # 개행 포함
            if (candidate_end - i) >= min_chunk:
                end = candidate_end

        # 2) UTF-8 경계 맞추기(필요시 end를 뒤로 이동)
        chunk = data[i:end]
        while True:
            try:
                text = chunk.decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
                if end <= i:
                    # 최후의 수단
                    text = data[i:min(i + max_bytes, n)].decode("utf-8", errors="replace")
                    end = min(i + max_bytes, n)
                    break
                chunk = data[i:end]

        # 3) 붙여넣기 편의: chunk가 개행으로 끝나지 않으면 개행 1개 추가
        # (대부분은 위에서 개행 기준으로 잘려서 이미 \n 으로 끝남)
        if not text.endswith("\n"):
            text += "\n"

        yield text
        i = end

def iter_utf8_chunks(file_path: str, max_bytes: int) -> Iterator[str]:
    data = Path(file_path).read_bytes()
    i, n = 0, len(data)
    while i < n:
        end = min(i + max_bytes, n)
        chunk = data[i:end]
        # ensure valid UTF-8 boundary
        while True:
            try:
                text = chunk.decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
                if end <= i:
                    text = data[i:min(i + max_bytes, n)].decode("utf-8", errors="replace")
                    end = min(i + max_bytes, n)
                    break
                chunk = data[i:end]
        yield text
        i = end


# ---------------------------
# Confluence REST API v1 client
# ---------------------------

class ConfluenceClientV1:
    def __init__(
        self,
        base_url: str,
        auth: Optional[HTTPBasicAuth] = None,
        bearer_token: Optional[str] = None,
        verify_tls: bool = True,
        ca_bundle: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify_tls
        # requests는 기본적으로 env(HTTP_PROXY/HTTPS_PROXY/REQUESTS_CA_BUNDLE 등)를 읽습니다.
        # 회사 프록시 환경이면 이게 도움이 됩니다.
        self.session.trust_env = True

        # TLS 검증 설정
        if ca_bundle:
            # 특정 CA 번들 경로를 지정하면 그 파일로 검증
            self.session.verify = ca_bundle
        else:
            # True/False
            self.session.verify = verify_tls
        
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        if bearer_token:
            self.session.headers.update({"Authorization": f"Bearer {bearer_token}"})
        if auth:
            self.session.auth = auth

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def create_child_page(self, space_key: str, parent_id: str, title: str, body_storage_value: str) -> dict:
        payload = {
            "type": "page",
            "title": title,
            "ancestors": [{"id": str(parent_id)}],
            "space": {"key": space_key},
            "body": {"storage": {"value": body_storage_value, "representation": "storage"}},
        }
        # r = self.session.post(self._url("/rest/api/content"), json=payload, timeout=self.timeout)
        r = self._post_json("/rest/api/content", payload)
        if not r.ok:
            _debug_http_error(r, "create_child_page")
        r.raise_for_status()
        return r.json()

    def get_page_storage_and_version(self, page_id: str) -> Tuple[str, int, str, str]:
        r = self.session.get(
            self._url(f"/rest/api/content/{page_id}?expand=body.storage,version,title,space"),
            timeout=self.timeout,
        )
        if not r.ok:
            _debug_http_error(r, "get_page_storage_and_version")
        r.raise_for_status()
        j = r.json()
        return (
            j["body"]["storage"]["value"],
            int(j["version"]["number"]),
            j["title"],
            j["space"]["key"],
        )

    def update_page_storage(self, page_id: str, new_body_storage_value: str, new_version_number: int, title: str, space_key: str) -> dict:
        payload = {
            "id": str(page_id),
            "type": "page",
            "title": title,
            "space": {"key": space_key},
            "body": {"storage": {"value": new_body_storage_value, "representation": "storage"}},
            "version": {"number": int(new_version_number)},
        }
        # r = self.session.put(self._url(f"/rest/api/content/{page_id}"), json=payload, timeout=self.timeout)
        r = self._put_json(f"/rest/api/content/{page_id}", payload)
        if not r.ok:
            _debug_http_error(r, "update_page_storage")
        r.raise_for_status()
        return r.json()
    
    def _post_json(self, path: str, payload: dict) -> requests.Response:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = dict(self.session.headers)
        headers["Content-Type"] = "application/json; charset=utf-8"
        # 디버그용(원하면)
        print(f"[DEBUG] POST {path} bytes={len(body)}")
        return self.session.post(self._url(path), data=body, headers=headers, timeout=self.timeout)

    def _put_json(self, path: str, payload: dict) -> requests.Response:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = dict(self.session.headers)
        headers["Content-Type"] = "application/json; charset=utf-8"
        print(f"[DEBUG] PUT  {path} bytes={len(body)}")
        return self.session.put(self._url(path), data=body, headers=headers, timeout=self.timeout)



def _debug_http_error(r: requests.Response, where: str) -> None:
    print(f"[HTTP ERROR] {where}: {r.status_code} {r.reason}")
    x_login = r.headers.get("x-seraph-loginreason")
    if x_login:
        print(f"[HTTP ERROR] x-seraph-loginreason: {x_login}")
    try:
        print(r.text[:800])
    except Exception:
        pass


# ---------------------------
# URL extraction (important)
# ---------------------------

def normalize_site_url(site_url: str) -> str:
    """
    Convert:
      https://xxx.atlassian.net       -> https://xxx.atlassian.net/wiki
      https://xxx.atlassian.net/wiki  -> (same)
    """
    site_url = (site_url or "").strip().rstrip("/")
    if not site_url:
        return ""
    if site_url.endswith("/wiki"):
        return site_url
    return site_url + "/wiki"

def extract_web_url(created: dict, site_url: str = "") -> str:
    """
    Prefer Confluence response links:
      created["_links"]["base"] + created["_links"]["webui"]
    If base is relative (e.g. /wiki), we need site_url.
    """
    links = created.get("_links") or {}
    base = links.get("base") or ""
    webui = links.get("webui") or ""

    if base and webui:
        if str(base).startswith("http"):
            return str(base).rstrip("/") + str(webui)
        # base like "/wiki"
        su = normalize_site_url(site_url)
        if su:
            return su + str(webui)  # su already endswith /wiki, webui usually startswith /spaces...
        return ""  # can't build absolute

    # fallback: if only webui exists
    if webui:
        su = normalize_site_url(site_url)
        if su:
            return su + str(webui)
    return ""


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base-url", default=os.getenv("CONFLUENCE_BASE_URL", "").strip(),
                    help="e.g. https://api.atlassian.com/ex/confluence/<cloudId>/wiki OR https://<site>.atlassian.net/wiki")
    ap.add_argument("--site-url", default=os.getenv("CONFLUENCE_SITE_URL", "").strip(),
                    help="(Recommended) Your human site URL, e.g. https://<site>.atlassian.net or https://<site>.atlassian.net/wiki")
    ap.add_argument("--space-key", default=os.getenv("CONFLUENCE_SPACE_KEY", "").strip(),
                    help="Space key (e.g. ~xxxxx)")
    ap.add_argument("--parent-id", default=os.getenv("CONFLUENCE_PARENT_PAGE_ID", "").strip(),
                    help="Parent page id (QUESTION ROOT will be created under this)")

    ap.add_argument("--title", default=None, help="QUESTION ROOT title (default: filename + timestamp)")
    ap.add_argument("--answer-title-suffix", default=os.getenv("CONFLUENCE_ANSWER_SUFFIX", " - ANSWER"),
                    help="Suffix for ANSWER page title")
    ap.add_argument("--part-title-prefix", default=os.getenv("CONFLUENCE_PART_PREFIX", " - PART "),
                    help="Prefix for PART pages title")

    ap.add_argument("--file", required=True, help="Local txt file path")
    ap.add_argument("--language", default="none", help="Code block language (none/python/sql/...)")
    ap.add_argument("--max-bytes", type=int, default=int(os.getenv("CONFLUENCE_MAX_BYTES", DEFAULT_MAX_BYTES)))
    ap.add_argument("--sleep-secs", type=int, default=int(os.getenv("CONFLUENCE_SLEEP_SECS", 0)),
                    help="Sleep between creating PART pages")

    # TLS
    ap.add_argument("--verify-tls", dest="verify_tls", action="store_true", default=True,
                    help="Verify TLS certificates (default: on)")
    ap.add_argument("--no-verify-tls", dest="verify_tls", action="store_false",
                    help="Disable TLS verification (NOT recommended, but useful to test quickly)")
    ap.add_argument("--ca-bundle", default=os.getenv("REQUESTS_CA_BUNDLE", "").strip(),
                    help="Path to CA bundle file (.pem/.cer). If set, used for TLS verification.")
    # Auth
    ap.add_argument("--auth-mode", choices=["basic", "bearer"], default=os.getenv("CONFLUENCE_AUTH_MODE", "basic"))
    ap.add_argument("--email", default=os.getenv("CONFLUENCE_EMAIL", ""))
    ap.add_argument("--api-token", default=os.getenv("CONFLUENCE_API_TOKEN", ""))
    ap.add_argument("--bearer-token", default=os.getenv("CONFLUENCE_BEARER_TOKEN", ""))

    ap.add_argument("--timeout", type=int, default=int(os.getenv("CONFLUENCE_TIMEOUT", DEFAULT_TIMEOUT_SECS)))

    args = ap.parse_args()

    if not args.base_url or not args.space_key or not args.parent_id:
        raise SystemExit("ERROR: --base-url, --space-key, --parent-id are required (or set env vars).")

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"ERROR: file not found: {file_path}")

    ca_bundle = args.ca_bundle if args.ca_bundle else None

    # Build client
    if args.auth_mode == "basic":
        if not args.email or not args.api_token:
            raise SystemExit("ERROR: basic auth requires --email and --api-token (or env vars).")
        client = ConfluenceClientV1(
            args.base_url,
            auth=HTTPBasicAuth(args.email, args.api_token),
            verify_tls=args.verify_tls,
            ca_bundle=ca_bundle,
            timeout=args.timeout,
        )
    else:
        if not args.bearer_token:
            raise SystemExit("ERROR: bearer auth requires --bearer-token (or env vars).")
        client = ConfluenceClientV1(
            args.base_url,
            bearer_token=args.bearer_token,
            verify_tls=args.verify_tls,
            ca_bundle=ca_bundle,
            timeout=args.timeout,
        )

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    q_title = args.title or f"{file_path.name} ({ts})"
    a_title = f"{q_title}{args.answer_title_suffix}"

    # 1) Create QUESTION ROOT (placeholder; will update once at end)
    q_root_body_placeholder = (
        make_h2("NAVIGATION")
        + make_p_raw("<strong>ANSWER:</strong> (creating...)")
        + make_hr()
        + make_h2("QUESTION PARTS")
        + make_p_raw("(creating...)")
    )
    created_q_root = client.create_child_page(args.space_key, args.parent_id, q_title, q_root_body_placeholder)
    q_root_id = created_q_root["id"]
    q_root_url = extract_web_url(created_q_root, args.site_url)  # may be empty if site_url not provided
    print(f"Created QUESTION ROOT: id={q_root_id} title={q_title}")

    # 2) Create ANSWER page under QUESTION ROOT
    answer_body = (
        make_h2("ANSWER")
        + make_p_raw("휴대폰에서 질문 복사 → ChatGPT 질문 → 아래에 답변 붙여넣기")
        + make_p_raw(f"<strong>Back:</strong> {make_a(q_root_url, 'QUESTION ROOT')}")
        + make_hr()
        + make_code_macro("", language="none", title="Paste answer here")
    )
    created_a = client.create_child_page(args.space_key, q_root_id, a_title, answer_body)
    a_id = created_a["id"]
    a_url = extract_web_url(created_a, args.site_url)
    print(f"Created ANSWER page: id={a_id} title={a_title}")

    # 3) Create PART pages (children of QUESTION ROOT)
    parts: List[Dict[str, str]] = []
    idx = 0
    for idx, chunk in enumerate(iter_utf8_chunks(str(file_path), args.max_bytes), start=1):
        part_no = f"{idx:03d}"
        part_title = f"{q_title}{args.part_title_prefix}{part_no}"

        part_body = (
            make_h2(f"QUESTION PART {part_no}")
            + make_p_raw(f"<strong>QUESTION ROOT:</strong> {make_a(q_root_url, 'Go to ROOT')}")
            + make_p_raw(f"<strong>ANSWER:</strong> {make_a(a_url, 'Go to ANSWER')}")
            + make_hr()
            + make_code_macro(chunk, language=args.language, title=f"Part {part_no}")
        )

        created_part = client.create_child_page(args.space_key, q_root_id, part_title, part_body)
        part_id = str(created_part["id"])
        part_url = extract_web_url(created_part, args.site_url)

        parts.append({"id": part_id, "title": part_title, "url": part_url})
        print(f"Created PART {idx}: id={part_id} title={part_title}")

        if args.sleep_secs:
            time.sleep(args.sleep_secs)

    total_parts = idx

    # 4) Update QUESTION ROOT once with visible <a href> links
    nav_html: List[str] = []
    nav_html.append(make_h2("NAVIGATION"))
    nav_html.append(make_p_raw(f"<strong>ANSWER:</strong> {make_a(a_url, 'Open ANSWER page')}"))
    nav_html.append(make_hr())

    nav_html.append(make_h2("QUESTION PARTS"))
    nav_html.append("<ul>")
    for i, p in enumerate(parts, start=1):
        part_no = f"{i:03d}"
        # 링크가 비어도 텍스트는 보이게: "PART 001" 텍스트 + (가능하면 링크)
        link = make_a(p.get("url"), f"PART {part_no}")
        nav_html.append(f"<li><p>{link}</p></li>")
    nav_html.append("</ul>")

    nav_html.append(make_hr())
    nav_html.append(make_h2("META"))
    nav_html.append(f"<p>Source file: {escape_xml(str(file_path))}</p>")
    nav_html.append(f"<p>Total parts: {total_parts} (max {args.max_bytes} bytes per part)</p>")
    nav_html.append("<p>Tip: On mobile, open PART pages to copy question text.</p>")

    new_q_body = "\n".join(nav_html)

    cur_body, cur_ver, cur_title_now, cur_space = client.get_page_storage_and_version(q_root_id)
    client.update_page_storage(q_root_id, new_q_body, cur_ver + 1, cur_title_now, cur_space)
    print("Updated QUESTION ROOT with link navigation.")
    print("Done.")


if __name__ == "__main__":
    main()
