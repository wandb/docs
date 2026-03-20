#!/usr/bin/env python3
"""
Download AI assistant conversation history from Mintlify Analytics.

Calls GET https://api.mintlify.com/v1/analytics/{projectId}/assistant and
paginates until all conversations are collected. See:
https://www.mintlify.com/docs/api/analytics/assistant-conversations

Credentials are read only from dotenv files at the repository root (next to
this repo's top-level directories such as scripts/ and models/):

  .env
  .env.local   (optional; values override .env when the same key appears)

Required keys in those files:

  MINTLIFY_PROJECT_ID  Project ID from the organization API keys page (same page as the key)
  MINTLIFY_API_KEY     Organization admin API key (must start with mint_)

Usage:
  python scripts/download_mintlify_assistant_conversations.py -o conversations.json
  python scripts/download_mintlify_assistant_conversations.py --verbose

Cloudflare in front of api.mintlify.com may reject Python's default User-Agent or TLS
fingerprint. This script sends a normal browser User-Agent and, if it still gets
error 1010, retries the same request with curl when curl is installed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://api.mintlify.com"

_REPO_ROOT = Path(__file__).resolve().parent.parent

_DASHBOARD_API_KEYS = "https://dashboard.mintlify.com/settings/organization/api-keys"

# Default urllib User-Agent is often blocked by Cloudflare (error 1010).
_BROWSER_LIKE_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _normalize_api_key(value: str) -> str:
    key = value.strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def _scrub_credential(value: str) -> str:
    """Strip whitespace and invisible characters often pasted from chat or PDF."""
    for ch in (
        "\u200b",
        "\u200c",
        "\u200d",
        "\ufeff",
    ):
        value = value.replace(ch, "")
    return value.strip()


def _request_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": _BROWSER_LIKE_UA,
        "Accept-Language": "en-US,en;q=0.9",
    }


def _fetch_via_curl(url: str, token: str, timeout: int) -> dict[str, Any]:
    header_args: list[str] = []
    for key, val in _request_headers(token).items():
        header_args.extend(["-H", f"{key}: {val}"])
    with tempfile.TemporaryDirectory() as tmp:
        body_path = Path(tmp) / "body"
        cmd = [
            "curl",
            "-sS",
            *header_args,
            "-o",
            str(body_path),
            "-w",
            "%{http_code}",
            "--max-time",
            str(timeout),
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.stderr.strip():
            raise RuntimeError(proc.stderr.strip())
        try:
            status = int(proc.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(f"curl: unexpected status line: {proc.stdout!r}") from exc
        raw = body_path.read_text(encoding="utf-8")
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {raw}")
        return json.loads(raw)


def _unquote_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1].replace("\\n", "\n").replace("\\r", "\r")
    return raw


def _parse_dotenv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not match:
            continue
        key, value = match.group(1), _unquote_value(match.group(2))
        out[key] = value
    return out


def load_root_dotenv(repo_root: Path) -> dict[str, str]:
    merged: dict[str, str] = {}
    for name in (".env", ".env.local"):
        path = repo_root / name
        if path.is_file():
            merged.update(_parse_dotenv_file(path))
    return merged


def fetch_page(
    project_id: str,
    token: str,
    *,
    date_from: str | None,
    date_to: str | None,
    limit: int,
    cursor: str | None,
) -> dict[str, Any]:
    path_seg = quote(project_id, safe="-_.~")
    path = f"/v1/analytics/{path_seg}/assistant"
    params: dict[str, str | int] = {"limit": limit}
    if date_from:
        params["dateFrom"] = date_from
    if date_to:
        params["dateTo"] = date_to
    if cursor:
        params["cursor"] = cursor
    query = urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    req = Request(url, headers=_request_headers(token), method="GET")
    try:
        with urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        if e.code == 403 and shutil.which("curl") is not None:
            try:
                return _fetch_via_curl(url, token, 120)
            except RuntimeError as curl_err:
                raise RuntimeError(
                    f"HTTP {e.code}: {e.reason}\n{detail}\n\nRetry with curl failed:\n{curl_err}"
                ) from curl_err
        raise RuntimeError(f"HTTP {e.code}: {e.reason}\n{detail}") from e


def download_all(
    project_id: str,
    token: str,
    *,
    date_from: str | None,
    date_to: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    all_conversations: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        data = fetch_page(
            project_id,
            token,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            cursor=cursor,
        )
        batch = data.get("conversations") or []
        all_conversations.extend(batch)
        has_more = data.get("hasMore", False)
        next_cursor = data.get("nextCursor")
        if not has_more or not next_cursor:
            break
        cursor = next_cursor
    return all_conversations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Mintlify assistant conversations via the Analytics API."
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="ISO 8601 or YYYY-MM-DD start date (optional)",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="ISO 8601 or YYYY-MM-DD end date, exclusive (optional)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Page size (1-1000, default 500)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mintlify_assistant_conversations.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print request URL and non-secret credential shape (for debugging 401s)",
    )
    args = parser.parse_args()

    env = load_root_dotenv(_REPO_ROOT)
    project_id = _scrub_credential(env.get("MINTLIFY_PROJECT_ID") or "")
    api_key = _normalize_api_key(_scrub_credential(env.get("MINTLIFY_API_KEY") or ""))

    if not project_id or not api_key:
        print(
            "Error: set MINTLIFY_PROJECT_ID and MINTLIFY_API_KEY in "
            f"{_REPO_ROOT / '.env'} (optional overrides in .env.local).",
            file=sys.stderr,
        )
        return 1

    if not api_key.startswith("mint_"):
        print(
            "Warning: Mintlify admin API keys normally start with mint_. "
            f"If requests fail, create a key on {_DASHBOARD_API_KEYS}.",
            file=sys.stderr,
        )

    if args.limit < 1 or args.limit > 1000:
        print("Error: --limit must be between 1 and 1000.", file=sys.stderr)
        return 1

    if args.verbose:
        probe = urlencode({"limit": min(args.limit, 1000)})
        path_seg = quote(project_id, safe="-_.~")
        print(
            f"GET {BASE_URL}/v1/analytics/{path_seg}/assistant?{probe}",
            file=sys.stderr,
        )
        print(
            f"Project ID length={len(project_id)} repr={project_id!r}",
            file=sys.stderr,
        )
        print(
            f"API key length={len(api_key)} starts_with_mint_={api_key.startswith('mint_')}",
            file=sys.stderr,
        )

    try:
        conversations = download_all(
            project_id,
            api_key,
            date_from=args.date_from,
            date_to=args.date_to,
            limit=args.limit,
        )
    except RuntimeError as e:
        msg = str(e)
        print(msg, file=sys.stderr)
        if "1010" in msg and shutil.which("curl") is None:
            print(
                "Tip: install curl so this script can retry blocked requests.",
                file=sys.stderr,
            )
        if "401" in msg or "Unauthorized" in msg:
            print(
                "\n401 means Mintlify rejected the Bearer token for this project (not a transport bug).\n"
                f"  - Open {_DASHBOARD_API_KEYS}\n"
                "  - Create or rotate an organization admin API key (usually mint_...).\n"
                "  - Set MINTLIFY_PROJECT_ID to the Project ID shown on that page, not the docs "
                "subdomain unless the dashboard explicitly says they are the same.\n"
                "  - Put only the raw secret in MINTLIFY_API_KEY (no 'Bearer ' prefix).\n"
                "  - Run again with --verbose to confirm URL, ID length, and key prefix.\n"
                "  - If it still fails, the key may lack Analytics API access for your plan; "
                "contact Mintlify support with the time of the request.\n",
                file=sys.stderr,
            )
        return 1
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        print(f"HTTP {e.code}: {e.reason}", file=sys.stderr)
        if detail:
            print(detail, file=sys.stderr)
        return 1
    except URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        return 1

    out_path = args.output
    payload = {"count": len(conversations), "conversations": conversations}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(conversations)} conversations to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
