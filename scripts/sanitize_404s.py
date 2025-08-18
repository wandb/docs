#!/usr/bin/env python3
"""
Redirect sanitization tool for GSC 404 exports.

Conservative pipeline:
1) Normalize and filter to owned hosts; drop excluded and duplicates
2) Verify current 404 (polite HEAD/GET with safety & caching)
3) Establish historical existence (Wayback; optional git slug signal)
4) Try safe heuristic mappings; validate target 200 on owned host
5) Emit Cloudflare _redirects candidates and needs_review.csv

Safety:
- Allowlist hosts only, no crawling, dedupe inputs
- Polite user-agent, timeouts, per-host pacing + jitter, robots.txt check
- Backoff on 429/503 with Retry-After; capped retries
- Wayback API throttled; persistent SQLite cache w/ TTLs

Usage example:
  python scripts/sanitize_404s.py \
    --input gsc_404s.csv \
    --out-redirects _redirects.candidates \
    --out-review needs_review.csv \
    --owned-hosts www.example.com,docs.example.com \
    --exclude-hosts medium.com,reddit.com,news.ycombinator.com,stackoverflow.com
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import errno
import json
import os
import random
import re
import sqlite3
import string
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from urllib.parse import urlsplit, urlunsplit, unquote, urlencode, parse_qsl
from urllib import robotparser


# ----------------------------- Config defaults ----------------------------- #

DEFAULT_USER_AGENT = "Redirect Validator (+contact@example.com)"
CONNECT_TIMEOUT_S = 3
READ_TIMEOUT_S = 5
TOTAL_TIMEOUT_TUPLE = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)

# Per-host politeness. 0.5 rps = 1 request every 2s.
PER_HOST_MIN_INTERVAL_S = 2.0
JITTER_RANGE_S = (0.1, 0.4)

# Retry policy
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_S = 0.5
RETRY_BACKOFF_CAP_S = 30.0

# Wayback rate cap
WAYBACK_MIN_INTERVAL_S = 1.0

# Cache TTLs
TTL_SUCCESS_S = int(timedelta(days=14).total_seconds())
TTL_404_S = int(timedelta(days=7).total_seconds())
TTL_TRANSIENT_S = int(timedelta(hours=1).total_seconds())

# Hard budgets
GLOBAL_MAX_REQUESTS = 2000
PER_HOST_MAX_REQUESTS = 500


# ------------------------------ Data classes ------------------------------ #

@dataclass
class UrlRecord:
    raw_url: str
    host: str
    path: str
    scheme: str
    suggested_rule: str = ""


@dataclass
class FetchResult:
    status: Optional[int]
    headers: Dict[str, str]
    error: Optional[str]
    method: str
    ts: float


# --------------------------- Utility and helpers -------------------------- #

_SAFE_QUERY_PREFIXES = ("utm_", "gclid", "fbclid")


def ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def now_s() -> float:
    return time.time()


def normalize_url(raw: str) -> Tuple[str, str, str]:
    parts = urlsplit(raw.strip())
    scheme = parts.scheme or "https"
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    # Strip common tracking params
    if parts.query:
        q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        q = [(k, v) for (k, v) in q if not k.lower().startswith(_SAFE_QUERY_PREFIXES)]
        query = urlencode(q)
    else:
        query = ""

    path = unquote(parts.path or "/")
    # Normalize index.html and trailing slash
    if path.endswith("/index.html"):
        path = path[: -len("/index.html")] or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # Reassemble but we return host, path, scheme only
    return host, path or "/", scheme or "https"


def extract_slug(path: str) -> str:
    # Last path segment without extension
    segment = path.strip("/").split("/")[-1] if path and path != "/" else ""
    segment = re.sub(r"\.[a-zA-Z0-9]+$", "", segment)
    return segment


def heuristic_targets(path: str) -> List[str]:
    candidates: List[str] = []
    # 1) drop .html
    if path.endswith(".html"):
        candidates.append(path[: -len(".html")])
    # 2) normalize blog structure /YYYY/MM/slug -> /blog/slug (if present)
    m = re.search(r"/(?:\d{4}/\d{2}/)?([a-z0-9-]+)$", path)
    if m:
        slug = m.group(1)
        candidates.append(f"/blog/{slug}")
    # 3) docs version strip: /docs/vX/... -> /docs/...
    candidates.append(re.sub(r"^/docs/v\d+(/|$)", "/docs/", path))
    # 4) ensure with trailing slash
    if not path.endswith("/"):
        candidates.append(path + "/")
    # Dedup preserve order
    seen = set()
    deduped = []
    for c in candidates:
        if c and c != path and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


# ------------------------------- Robots logic ------------------------------ #

class RobotsCache:
    def __init__(self, session: requests.Session, cache_db: sqlite3.Connection, user_agent: str):
        self.session = session
        self.db = cache_db
        self.user_agent = user_agent
        self._init_table()
        self._per_host_crawl_delay: Dict[str, float] = {}

    def _init_table(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS robots_cache (
              host TEXT PRIMARY KEY,
              robots_txt TEXT,
              fetched_at INTEGER
            )
            """
        )
        self.db.commit()

    def _get_cached(self, host: str) -> Optional[Tuple[str, int]]:
        cur = self.db.execute(
            "SELECT robots_txt, fetched_at FROM robots_cache WHERE host = ?",
            (host,),
        )
        row = cur.fetchone()
        return (row[0], int(row[1])) if row else None

    def _set_cached(self, host: str, robots_txt: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO robots_cache (host, robots_txt, fetched_at) VALUES (?, ?, ?)",
            (host, robots_txt, int(now_s())),
        )
        self.db.commit()

    def _parse_crawl_delay(self, robots_txt: str, host: str) -> float:
        # Very simple crawl-delay parse honoring specific UA then *
        ua = self.user_agent
        delay = None
        block = None
        current_ua_block = None
        lines = robots_txt.splitlines()
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("user-agent:"):
                current_ua_block = s.split(":", 1)[1].strip()
                continue
            if s.lower().startswith("crawl-delay:"):
                val = s.split(":", 1)[1].strip()
                try:
                    d = float(val)
                except ValueError:
                    continue
                # Prefer exact UA match, then *
                if current_ua_block and current_ua_block in (ua, "*"):
                    delay = d
        return float(delay) if delay is not None else 0.0

    def can_fetch(self, scheme: str, host: str, path: str) -> Tuple[bool, float]:
        cached = self._get_cached(host)
        txt: Optional[str]
        if cached:
            txt = cached[0]
        else:
            url = f"{scheme}://{host}/robots.txt"
            headers = {"User-Agent": self.user_agent}
            try:
                r = self.session.get(url, headers=headers, timeout=TOTAL_TIMEOUT_TUPLE)
                txt = r.text if r.status_code == 200 else ""
            except requests.RequestException:
                txt = ""
            self._set_cached(host, txt or "")

        rp = robotparser.RobotFileParser()
        rp.parse((txt or "").splitlines())
        allowed = rp.can_fetch(self.user_agent, f"{scheme}://{host}{path}")
        crawl_delay = self._parse_crawl_delay(txt or "", host)
        return allowed, crawl_delay


# ---------------------------- Requests and cache --------------------------- #

class RequestCache:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._init_tables()

    def _init_tables(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS url_cache (
              key TEXT PRIMARY KEY,
              status INTEGER,
              headers TEXT,
              method TEXT,
              ts INTEGER,
              error TEXT
            )
            """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS wayback_cache (
              key TEXT PRIMARY KEY,
              has_snapshot INTEGER,
              snapshot_url TEXT,
              ts INTEGER
            )
            """
        )
        self.db.commit()

    def get_url(self, key: str) -> Optional[FetchResult]:
        cur = self.db.execute(
            "SELECT status, headers, method, ts, error FROM url_cache WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        status = row[0]
        headers = json.loads(row[1]) if row[1] else {}
        method = row[2]
        ts = float(row[3])
        error = row[4]
        return FetchResult(status=status, headers=headers, error=error, method=method, ts=ts)

    def set_url(self, key: str, result: FetchResult) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO url_cache (key, status, headers, method, ts, error) VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                result.status if result.status is not None else None,
                json.dumps(result.headers or {}),
                result.method,
                int(result.ts),
                result.error or None,
            ),
        )
        self.db.commit()

    def get_wayback(self, key: str) -> Optional[Tuple[bool, Optional[str], float]]:
        cur = self.db.execute(
            "SELECT has_snapshot, snapshot_url, ts FROM wayback_cache WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return bool(row[0]), row[1], float(row[2])

    def set_wayback(self, key: str, has_snapshot: bool, snapshot_url: Optional[str]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO wayback_cache (key, has_snapshot, snapshot_url, ts) VALUES (?, ?, ?, ?)",
            (key, 1 if has_snapshot else 0, snapshot_url or None, int(now_s())),
        )
        self.db.commit()


class PoliteFetcher:
    def __init__(self, session: requests.Session, cache: RequestCache, robots: RobotsCache, user_agent: str):
        self.session = session
        self.cache = cache
        self.robots = robots
        self.user_agent = user_agent
        self.per_host_next_time: Dict[str, float] = defaultdict(lambda: 0.0)
        self.per_host_counts: Dict[str, int] = defaultdict(int)
        self.global_count = 0
        self.wayback_next_time = 0.0

    def _sleep_until(self, ts: float) -> None:
        delay = max(0.0, ts - time.monotonic())
        if delay > 0:
            time.sleep(delay)

    def _polite_delay(self, host: str, extra_delay: float = 0.0) -> None:
        next_ok = max(self.per_host_next_time[host], time.monotonic())
        jitter = random.uniform(*JITTER_RANGE_S)
        next_ok = max(next_ok, time.monotonic())
        next_ok += jitter
        if extra_delay > 0:
            next_ok = max(next_ok, time.monotonic() + extra_delay)
        self._sleep_until(next_ok)
        self.per_host_next_time[host] = time.monotonic() + PER_HOST_MIN_INTERVAL_S

    def _respect_budgets(self, host: str) -> bool:
        if self.global_count >= GLOBAL_MAX_REQUESTS:
            return False
        if self.per_host_counts[host] >= PER_HOST_MAX_REQUESTS:
            return False
        return True

    def _should_use_cache(self, cached: Optional[FetchResult], intended_404_check: bool) -> bool:
        if not cached:
            return False
        age = now_s() - cached.ts
        if cached.status in (200, 301, 302, 410):
            return age < TTL_SUCCESS_S
        if cached.status == 404 and intended_404_check:
            return age < TTL_404_S
        # Transient or errors
        return age < TTL_TRANSIENT_S

    def head_then_get(self, scheme: str, host: str, path: str) -> FetchResult:
        key = f"{scheme}://{host}{path}"
        cached = self.cache.get_url(key)
        if self._should_use_cache(cached, intended_404_check=True):
            return cached  # type: ignore

        if not self._respect_budgets(host):
            return FetchResult(status=None, headers={}, error="budget_exceeded", method="NONE", ts=now_s())

        allowed, crawl_delay = self.robots.can_fetch(scheme, host, path)
        if not allowed:
            res = FetchResult(status=None, headers={}, error="robots_disallow", method="NONE", ts=now_s())
            self.cache.set_url(key, res)
            return res

        headers = {"User-Agent": self.user_agent}

        def attempt_head() -> Optional[requests.Response]:
            try:
                return self.session.head(
                    key,
                    allow_redirects=False,
                    headers=headers,
                    timeout=TOTAL_TIMEOUT_TUPLE,
                )
            except requests.RequestException:
                return None

        def attempt_get_range() -> Optional[requests.Response]:
            try:
                return self.session.get(
                    key,
                    allow_redirects=False,
                    headers={**headers, "Range": "bytes=0-0"},
                    stream=True,
                    timeout=TOTAL_TIMEOUT_TUPLE,
                )
            except requests.RequestException:
                return None

        attempt = 0
        result: Optional[FetchResult] = None
        while attempt <= MAX_RETRIES:
            self._polite_delay(host, extra_delay=crawl_delay)
            self.global_count += 1
            self.per_host_counts[host] += 1
            r = attempt_head()
            method = "HEAD"
            if r is None or r.status_code in (405, 501):
                self._polite_delay(host, extra_delay=crawl_delay)
                self.global_count += 1
                self.per_host_counts[host] += 1
                r = attempt_get_range()
                method = "GET"

            if r is None:
                # timeout/network error
                if attempt >= MAX_RETRIES:
                    result = FetchResult(status=None, headers={}, error="network_error", method=method, ts=now_s())
                    break
                backoff_sleep(attempt)
                attempt += 1
                continue

            status = r.status_code
            headers_out = {k: v for k, v in r.headers.items()}

            if status in (429, 503):
                retry_after = headers_out.get("Retry-After")
                if attempt >= MAX_RETRIES:
                    result = FetchResult(status=status, headers=headers_out, error="rate_limited", method=method, ts=now_s())
                    break
                backoff_sleep(attempt, retry_after)
                attempt += 1
                continue

            # No retry on other statuses
            result = FetchResult(status=status, headers=headers_out, error=None, method=method, ts=now_s())
            break

        assert result is not None
        self.cache.set_url(key, result)
        return result

    def wayback_available(self, scheme: str, host: str, path: str) -> Tuple[bool, Optional[str]]:
        key = f"{scheme}://{host}{path}"
        cached = self.cache.get_wayback(key)
        if cached:
            has, url, ts = cached
            age = now_s() - ts
            if age < TTL_404_S:  # reuse for up to a week
                return has, url

        # Enforce 1 rps
        self._sleep_until(self.wayback_next_time)
        self.wayback_next_time = time.monotonic() + WAYBACK_MIN_INTERVAL_S

        try:
            r = self.session.get(
                "https://archive.org/wayback/available",
                params={"url": key},
                headers={"User-Agent": self.user_agent},
                timeout=TOTAL_TIMEOUT_TUPLE,
            )
            data = r.json() if r.status_code == 200 else {}
        except Exception:
            data = {}
        snapshot = data.get("archived_snapshots", {}).get("closest") or None
        has = bool(snapshot)
        url = snapshot.get("url") if snapshot else None
        self.cache.set_wayback(key, has, url)
        return has, url


def backoff_sleep(attempt: int, retry_after_header: Optional[str] = None) -> None:
    if retry_after_header:
        try:
            # Could be seconds or HTTP-date; handle seconds only for simplicity
            seconds = float(retry_after_header)
            time.sleep(seconds)
            return
        except Exception:
            pass
    sleep_cap = min(RETRY_BACKOFF_CAP_S, RETRY_BACKOFF_BASE_S * (2 ** attempt))
    time.sleep(random.uniform(0.0, sleep_cap))


# ------------------------------- Git heuristics ---------------------------- #

def git_slug_existed(repo_root: Path, slug: str) -> bool:
    if not slug:
        return False
    try:
        # List historical objects with paths; search slug in path string
        proc = subprocess.run(
            ["git", "rev-list", "--all", "--objects"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            # format: <sha> <path>
            parts = line.split(" ", 1)
            if len(parts) == 2:
                path = parts[1]
                if slug in path:
                    return True
    except Exception:
        return False
    return False


# ------------------------------ Core pipeline ------------------------------ #

def read_input_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def build_url_records(rows: List[Dict[str, str]], owned: Sequence[str]) -> List[UrlRecord]:
    records: List[UrlRecord] = []
    seen: set = set()
    for row in rows:
        raw = row.get("url") or row.get("URL") or row.get("page") or row.get("Page") or ""
        if not raw:
            continue
        host, path, scheme = normalize_url(raw)
        if not host:
            continue
        if not (host in owned or any(host.endswith("." + h) for h in owned)):
            continue
        key = (host, path, scheme)
        if key in seen:
            continue
        seen.add(key)
        suggested = row.get("suggested_rule") or row.get("suggestion") or row.get("Suggested") or ""
        records.append(UrlRecord(raw_url=raw, host=host, path=path, scheme=scheme, suggested_rule=suggested))
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sanitize GSC 404 report into safe Cloudflare redirects")
    parser.add_argument("--input", required=True, help="Path to GSC CSV export")
    parser.add_argument("--out-redirects", required=True, help="Output path for _redirects candidates")
    parser.add_argument("--out-review", required=True, help="Output CSV path for needs review")
    parser.add_argument("--owned-hosts", required=True, help="Comma-separated allowlist of owned hosts (no scheme; e.g., www.example.com,docs.example.com)")
    parser.add_argument("--exclude-hosts", default="", help="Comma-separated excluded hosts (and subdomains)")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent for requests")
    parser.add_argument("--cache-db", default=".cache/redirect_sanitizer.sqlite", help="Path to SQLite cache DB")
    parser.add_argument("--dry-run-limit", type=int, default=0, help="If >0, only process first N URLs for a dry run")
    parser.add_argument("--enable-git", action="store_true", help="Enable git slug existence heuristic (may be slow on first run)")
    parser.add_argument("--include-410", action="store_true", help="If set, emit 410 for owned pages with evidence of historical existence but no replacement")

    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    input_csv = Path(args.input)
    out_redirects = Path(args.out_redirects)
    out_review = Path(args.out_review)
    cache_db_path = Path(args.cache_db)
    ensure_dir(cache_db_path.parent)

    def _norm_host_token(h: str) -> str:
        h = h.strip().lower().lstrip(".")
        return h[4:] if h.startswith("www.") else h

    owned_hosts = [_norm_host_token(h) for h in args.owned_hosts.split(",") if h.strip()]
    excluded_hosts = [_norm_host_token(h) for h in args.exclude_hosts.split(",") if h.strip()]

    rows = read_input_csv(input_csv)
    records = build_url_records(rows, owned_hosts)

    # Exclude explicitly excluded hosts and their subdomains
    def is_excluded(host: str) -> bool:
        return host in excluded_hosts or any(host.endswith("." + h) for h in excluded_hosts)

    records = [r for r in records if not is_excluded(r.host)]

    if args.dry_run_limit and len(records) > args.dry_run_limit:
        records = records[: args.dry_run_limit]

    # Setup HTTP session and caches
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    db = sqlite3.connect(str(cache_db_path))
    cache = RequestCache(db)
    robots = RobotsCache(session, db, args.user_agent)
    fetcher = PoliteFetcher(session, cache, robots, args.user_agent)

    redirects: List[Tuple[str, str, int]] = []
    review_rows: List[Dict[str, str]] = []

    for rec in records:
        # 2) Verify 404
        fr = fetcher.head_then_get(rec.scheme, rec.host, rec.path)
        if fr.error == "budget_exceeded":
            review_rows.append({
                "url": rec.raw_url,
                "reason": "budget_exceeded",
                "suggested": "",
            })
            break
        if fr.error == "robots_disallow":
            # Skip network check but keep for review
            review_rows.append({
                "url": rec.raw_url,
                "reason": "robots_disallow",
                "suggested": "",
            })
            continue
        if fr.status is None:
            review_rows.append({
                "url": rec.raw_url,
                "reason": fr.error or "unknown_error",
                "suggested": "",
            })
            continue
        if fr.status != 404:
            # Not a current 404 → ignore
            continue

        # 3) Historical existence check
        wayback_has, _snapshot = fetcher.wayback_available(rec.scheme, rec.host, rec.path)
        slug = extract_slug(rec.path)
        git_has = git_slug_existed(repo_root, slug) if args.enable_git else False

        # 4) Heuristic mapping
        mapped_target: Optional[str] = None
        for candidate in heuristic_targets(rec.path):
            # Only same host target for safety; if cross-host needed, handle manually later
            t_res = fetcher.head_then_get(rec.scheme, rec.host, candidate)
            if t_res.status == 200:
                mapped_target = candidate
                break

        # 5) Decision
        if mapped_target:
            redirects.append((rec.path or "/", mapped_target, 301))
            continue

        if (wayback_has or git_has):
            if args.include_410:
                redirects.append((rec.path or "/", "410", 410))  # placeholder; emitted as 410 format later
            else:
                review_rows.append({
                    "url": rec.raw_url,
                    "reason": "historical_exists_no_target",
                    "suggested": "",
                })
        else:
            review_rows.append({
                "url": rec.raw_url,
                "reason": "no_evidence",
                "suggested": "",
            })

    # Emit outputs
    with out_redirects.open("w", encoding="utf-8") as outf:
        for src, dst, code in redirects:
            if code == 410:
                outf.write(f"{src}  410\n")
            else:
                outf.write(f"{src}  {dst}  {code}\n")

    with out_review.open("w", encoding="utf-8", newline="") as outf:
        w = csv.DictWriter(outf, fieldnames=["url", "reason", "suggested"])
        w.writeheader()
        for row in review_rows:
            w.writerow(row)

    print(f"Processed {len(records)} URLs. Redirects: {len(redirects)}. Review: {len(review_rows)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

