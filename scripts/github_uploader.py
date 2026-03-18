#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional
from urllib.parse import quote

import requests


DEFAULT_MAX_BYTES = 45_000
DEFAULT_TIMEOUT_SECS = 60
GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

TEXT_EXTENSIONS = {
    ".py", ".txt", ".md", ".sql", ".json", ".yaml", ".yml",
    ".xml", ".csv", ".log", ".ini", ".cfg", ".conf", ".js",
    ".ts", ".tsx", ".jsx", ".java", ".kt", ".scala", ".c",
    ".cpp", ".h", ".hpp", ".cs", ".sh", ".bat", ".ps1",
    ".html", ".css"
}


def join_repo_path(*parts: str) -> str:
    return "/".join([p.strip("/") for p in parts if p and p.strip("/")])


def normalize_repo_path(path: str) -> str:
    return str(PurePosixPath(path.strip().replace("\\", "/"))).strip("/")


def default_staging_path_from_publish_path(publish_path: str) -> str:
    publish = PurePosixPath(normalize_repo_path(publish_path))
    parent = "" if str(publish.parent) == "." else str(publish.parent)
    stem = Path(publish.name).stem
    return join_repo_path(parent, ".ai_staging", stem)


def estimate_base64_len(raw_bytes_len: int) -> int:
    """
    Base64 인코딩 후 길이(문자 수) 대략 계산
    """
    return ((raw_bytes_len + 2) // 3) * 4


def effective_raw_bytes_for_github(
    max_payload_bytes: int,
    json_overhead_bytes: int = 2048,
    min_raw_bytes: int = 1024,
) -> int:
    """
    GitHub Contents API 요청 시:
    - content는 base64 문자열로 들어가고
    - branch/message/sha/json wrapper 오버헤드가 붙는다.

    따라서 사용자가 넣은 max_payload_bytes를
    '실제 안전한 원본 바이트 한도'로 환산한다.
    """
    usable = max_payload_bytes - json_overhead_bytes
    if usable <= 0:
        return min_raw_bytes

    # base64는 대략 4/3배
    raw_bytes = (usable // 4) * 3
    return max(min_raw_bytes, raw_bytes)


def is_probably_text(file_path: Path, mode: str) -> bool:
    if mode == "text":
        return True
    if mode == "binary":
        return False

    if file_path.suffix.lower() in TEXT_EXTENSIONS:
        return True

    try:
        data = file_path.read_bytes()[:4096]
        data.decode("utf-8")
        return True
    except Exception:
        return False


def iter_utf8_chunks(file_path: str, max_bytes: int) -> Iterator[str]:
    """
    - max_bytes 이내에서 가능하면 마지막 개행 기준으로 자름
    - UTF-8 경계 안전하게 맞춤
    - 각 chunk 끝에 개행 1개 보장
    """
    data = Path(file_path).read_bytes()
    i = 0
    n = len(data)
    min_chunk = max(1024, max_bytes // 10)

    while i < n:
        end = min(i + max_bytes, n)
        window = data[i:end]

        nl = window.rfind(b"\n")
        if nl != -1:
            candidate_end = i + nl + 1
            if (candidate_end - i) >= min_chunk:
                end = candidate_end

        chunk = data[i:end]

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

        if not text.endswith("\n"):
            text += "\n"

        yield text
        i = end


class GitHubContentsError(Exception):
    pass


class GitHubContentsClient:
    def __init__(
        self,
        token: str,
        base_url: str = GITHUB_API,
        verify_tls: bool = True,
        ca_bundle: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_SECS,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.trust_env = True

        if ca_bundle:
            self.session.verify = ca_bundle
        else:
            self.session.verify = verify_tls

        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "github-uploader",
        })

    def _url(self, owner: str, repo: str, repo_path: str) -> str:
        encoded_path = quote(repo_path.strip("/"), safe="/")
        return f"{self.base_url}/repos/{owner}/{repo}/contents/{encoded_path}"

    def get_file_metadata(
        self,
        owner: str,
        repo: str,
        repo_path: str,
        branch: str,
    ) -> Optional[dict]:
        r = self.session.get(
            self._url(owner, repo, repo_path),
            params={"ref": branch},
            timeout=self.timeout,
        )
        if r.status_code == 404:
            return None
        if not r.ok:
            self._debug_http_error(r, f"get_file_metadata({repo_path})")
            raise GitHubContentsError(f"GET failed: {r.status_code} / {r.text}")
        return r.json()

    def get_file_text(
        self,
        owner: str,
        repo: str,
        repo_path: str,
        branch: str,
    ) -> Optional[str]:
        meta = self.get_file_metadata(owner, repo, repo_path, branch)
        if not meta:
            return None

        content = meta.get("content")
        encoding = meta.get("encoding")
        if content and encoding == "base64":
            return base64.b64decode(content).decode("utf-8")

        download_url = meta.get("download_url")
        if download_url:
            r = self.session.get(download_url, timeout=self.timeout)
            if not r.ok:
                self._debug_http_error(r, f"download({repo_path})")
                raise GitHubContentsError(f"Download failed: {r.status_code} / {r.text}")
            return r.text

        return None

    def upsert_bytes(
        self,
        owner: str,
        repo: str,
        repo_path: str,
        content_bytes: bytes,
        branch: str,
        commit_message: str,
    ) -> dict:
        url = self._url(owner, repo, repo_path)
        existing = self.get_file_metadata(owner, repo, repo_path, branch)
        sha = existing.get("sha") if existing else None

        payload = {
            "message": commit_message,
            "content": base64.b64encode(content_bytes).decode("ascii"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = dict(self.session.headers)
        headers["Content-Type"] = "application/json; charset=utf-8"

        print(f"[DEBUG] PUT {repo_path} bytes={len(content_bytes)}")
        r = self.session.put(url, data=body, headers=headers, timeout=self.timeout)

        if not r.ok:
            self._debug_http_error(r, f"upsert_bytes({repo_path})")
            raise GitHubContentsError(f"PUT failed: {r.status_code} / {r.text}")

        return r.json()

    def delete_file(
        self,
        owner: str,
        repo: str,
        repo_path: str,
        branch: str,
        commit_message: str,
    ) -> Optional[dict]:
        existing = self.get_file_metadata(owner, repo, repo_path, branch)
        if not existing:
            return None

        payload = {
            "message": commit_message,
            "sha": existing["sha"],
            "branch": branch,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = dict(self.session.headers)
        headers["Content-Type"] = "application/json; charset=utf-8"

        print(f"[DEBUG] DELETE {repo_path}")
        r = self.session.delete(
            self._url(owner, repo, repo_path),
            data=body,
            headers=headers,
            timeout=self.timeout,
        )

        if not r.ok:
            self._debug_http_error(r, f"delete_file({repo_path})")
            raise GitHubContentsError(f"DELETE failed: {r.status_code} / {r.text}")

        return r.json()

    def trigger_workflow_dispatch(
        self,
        owner: str,
        repo: str,
        workflow_id: str,
        ref: str,
        inputs: dict,
    ) -> None:
        url = (
            f"{self.base_url}/repos/{owner}/{repo}/actions/workflows/"
            f"{quote(workflow_id, safe='')}/dispatches"
        )
        payload = {
            "ref": ref,
            "inputs": inputs,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = dict(self.session.headers)
        headers["Content-Type"] = "application/json; charset=utf-8"

        print(f"[DEBUG] POST workflow dispatch: {workflow_id}")
        r = self.session.post(url, data=body, headers=headers, timeout=self.timeout)

        if not r.ok:
            self._debug_http_error(r, f"trigger_workflow_dispatch({workflow_id})")
            raise GitHubContentsError(f"Dispatch failed: {r.status_code} / {r.text}")

    @staticmethod
    def _debug_http_error(r: requests.Response, where: str) -> None:
        print(f"[HTTP ERROR] {where}: {r.status_code} {r.reason}")
        try:
            print(r.text[:1000])
        except Exception:
            pass


def load_old_manifest(
    client: GitHubContentsClient,
    owner: str,
    repo: str,
    manifest_path: str,
    branch: str,
) -> Optional[dict]:
    text = client.get_file_text(owner, repo, manifest_path, branch)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def upload_direct_file(
    client: GitHubContentsClient,
    owner: str,
    repo: str,
    branch: str,
    publish_path: str,
    local_file: Path,
) -> None:
    client.upsert_bytes(
        owner=owner,
        repo=repo,
        repo_path=publish_path,
        content_bytes=local_file.read_bytes(),
        branch=branch,
        commit_message=f"upload {Path(publish_path).name}",
    )
    print(f"Uploaded direct file: {publish_path}")


def upload_chunked_text_file(
    client: GitHubContentsClient,
    owner: str,
    repo: str,
    branch: str,
    publish_path: str,
    staging_path: str,
    local_file: Path,
    max_bytes: int,
    sleep_secs: float,
    dispatch_after_upload: bool,
    dispatch_workflow: str,
) -> None:
    parts_dir = join_repo_path(staging_path, "parts")
    manifest_path = join_repo_path(staging_path, "manifest.json")
    index_path = join_repo_path(staging_path, "INDEX.md")

    old_manifest = load_old_manifest(client, owner, repo, manifest_path, branch)
    old_part_files = set((old_manifest or {}).get("part_files", []))

    new_part_files = []

    for idx, chunk in enumerate(iter_utf8_chunks(str(local_file), max_bytes), start=1):
        part_no = f"{idx:03d}"
        repo_part_path = join_repo_path(parts_dir, f"PART_{part_no}.txt")

        client.upsert_bytes(
            owner=owner,
            repo=repo,
            repo_path=repo_part_path,
            content_bytes=chunk.encode("utf-8"),
            branch=branch,
            commit_message=f"upload {Path(publish_path).name} part {part_no}",
        )
        new_part_files.append(repo_part_path)
        print(f"Uploaded part: {repo_part_path}")

        if sleep_secs > 0:
            time.sleep(sleep_secs)

    stale_files = sorted(old_part_files - set(new_part_files))
    for stale_path in stale_files:
        client.delete_file(
            owner=owner,
            repo=repo,
            repo_path=stale_path,
            branch=branch,
            commit_message=f"delete stale part {Path(stale_path).name}",
        )
        print(f"Deleted stale part: {stale_path}")

    manifest = {
        "root_name": Path(publish_path).stem,
        "source_file_name": local_file.name,
        "encoding": "utf-8",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_full_path": publish_path,
        "index_path": index_path,
        "parts_dir": parts_dir,
        "part_files": new_part_files,
        "part_count": len(new_part_files),
        "max_bytes": max_bytes,
    }

    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"

    client.upsert_bytes(
        owner=owner,
        repo=repo,
        repo_path=manifest_path,
        content_bytes=manifest_text.encode("utf-8"),
        branch=branch,
        commit_message=f"upload manifest for {Path(publish_path).name}",
    )
    print(f"Uploaded manifest: {manifest_path}")

    if dispatch_after_upload:
        client.trigger_workflow_dispatch(
            owner=owner,
            repo=repo,
            workflow_id=dispatch_workflow,
            ref=branch,
            inputs={"manifest_path": manifest_path},
        )
        print(f"Triggered workflow: {dispatch_workflow}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", default="main")

    ap.add_argument("--file", required=True, help="Local file path")
    ap.add_argument(
        "--publish-path",
        required=True,
        help="Final repo path, e.g. src/daily_netting/mx_post_process/processor.py",
    )
    ap.add_argument(
        "--staging-path",
        default=None,
        help="Optional staging repo path. Default: sibling .ai_staging/<file_stem>",
    )
    ap.add_argument(
        "--target-name",
        default=None,
        help="Optional override for final repo filename. Usually omit this.",
    )

    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--sleep-secs", type=float, default=0.0)

    ap.add_argument("--mode", choices=["auto", "text", "binary"], default="auto")
    ap.add_argument("--always-chunk-text", action="store_true")

    ap.add_argument("--dispatch-after-upload", action="store_true")
    ap.add_argument("--dispatch-workflow", default="merge_parts.yml")

    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    ap.add_argument("--base-url", default=os.getenv("GITHUB_BASE_URL", GITHUB_API))

    ap.add_argument("--verify-tls", dest="verify_tls", action="store_true", default=True)
    ap.add_argument("--no-verify-tls", dest="verify_tls", action="store_false")
    ap.add_argument("--ca-bundle", default=os.getenv("REQUESTS_CA_BUNDLE", "").strip())
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECS)

    args = ap.parse_args()

    if not args.token:
        raise SystemExit("ERROR: --token or GITHUB_TOKEN is required.")

    local_file = Path(args.file)
    if not local_file.exists():
        raise SystemExit(f"ERROR: file not found: {local_file}")

    publish_path = normalize_repo_path(args.publish_path)
    if args.target_name:
        publish_path_obj = PurePosixPath(publish_path)
        parent = "" if str(publish_path_obj.parent) == "." else str(publish_path_obj.parent)
        publish_path = join_repo_path(parent, args.target_name)

    staging_path = (
        normalize_repo_path(args.staging_path)
        if args.staging_path
        else default_staging_path_from_publish_path(publish_path)
    )

    client = GitHubContentsClient(
        token=args.token,
        base_url=args.base_url,
        verify_tls=args.verify_tls,
        ca_bundle=args.ca_bundle if args.ca_bundle else None,
        timeout=args.timeout,
    )

    is_text = is_probably_text(local_file, args.mode)
    file_size = local_file.stat().st_size
    effective_max_bytes = effective_raw_bytes_for_github(args.max_bytes)

    print(f"Local file     : {local_file}")
    print(f"Publish path   : {publish_path}")
    print(f"Staging path   : {staging_path}")
    print(f"CA bundle      : {args.ca_bundle if args.ca_bundle else '(not set)'}")
    print(f"Verify TLS     : {args.verify_tls}")
    print(f"Detected text  : {is_text}")
    print(f"File size      : {file_size}")
    print(f"Max payload    : {args.max_bytes}")
    print(f"Safe raw bytes : {effective_max_bytes}")

    should_chunk = is_text and (args.always_chunk_text or file_size > effective_max_bytes)

    if should_chunk:
        upload_chunked_text_file(
            client=client,
            owner=args.owner,
            repo=args.repo,
            branch=args.branch,
            publish_path=publish_path,
            staging_path=staging_path,
            local_file=local_file,
            max_bytes=effective_max_bytes,
            sleep_secs=args.sleep_secs,
            dispatch_after_upload=args.dispatch_after_upload,
            dispatch_workflow=args.dispatch_workflow,
        )
    else:
        if (not is_text) and file_size > effective_max_bytes:
            print("WARNING: binary file is larger than safe raw limit, but direct upload will still be attempted.")
        upload_direct_file(
            client=client,
            owner=args.owner,
            repo=args.repo,
            branch=args.branch,
            publish_path=publish_path,
            local_file=local_file,
        )

    print("Done.")


if __name__ == "__main__":
    main()