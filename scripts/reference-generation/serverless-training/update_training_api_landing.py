#!/usr/bin/env python3
"""
Update the Serverless Training API landing page with the current endpoints from the OpenAPI spec.

Mirrors Mintlify's OpenAPI page path logic (@mintlify/scraping processOpenApiPath):
- Slug folder from the first tag (prepareStringToBeValidFilename).
- Slug file from summary, else "{method}-{slugified-path}".
- Duplicate (tag, slug) pairs get "-1", "-2", ... suffixes (same order as Mint: spec path
  order, then OpenAPI HTTP method order get, put, post, delete, ...).

See docs.json for the openapi `directory` (default: serverless-training/api-reference).

Endpoint links in the landing page use absolute https://docs.wandb.ai/... URLs on purpose:
`mint broken-links` registers OpenAPI targets using a single default directory
(`api-reference`, from @mintlify/link-rot getOpenApiPagePaths) and does not match
per-nav paths like `/serverless-training/api-reference/...`. Absolute links are checked
as external and still resolve to the real on-site paths under docs.wandb.ai.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.mintlify_openapi_paths import (  # noqa: E402
    PUBLIC_DOCS_ORIGIN,
    build_method_path_to_href,
    display_title,
    iter_operations,
)

DEFAULT_OPENAPI_DIRECTORY = "serverless-training/api-reference"
LANDING_PAGE = Path("serverless-training/api-reference.mdx")

# Display order for ### headings; unknown tags append after these.
TAG_HEADING_ORDER = ["chat-completions", "models", "training-jobs", "health"]


def fetch_openapi_spec() -> dict:
    """Load OpenAPI spec from local file, or fetch remote if missing."""
    local_spec = Path("serverless-training/api-reference/openapi.json")
    if local_spec.exists():
        print(f"  Using local OpenAPI spec: {local_spec}")
        with open(local_spec, "r", encoding="utf-8") as f:
            return json.load(f)

    import requests

    print("  Fetching remote OpenAPI spec from https://api.training.wandb.ai/openapi.json")
    response = requests.get("https://api.training.wandb.ai/openapi.json", timeout=30)
    response.raise_for_status()
    return response.json()


def list_display_rows(spec: dict) -> List[Tuple[str, str, str, str]]:
    """Rows of (method, path, summary, tag0) sorted for the landing page."""
    rows: List[Tuple[str, str, str, str]] = []
    for method, path, operation in iter_operations(spec):
        tags = operation.get("tags") or ["Uncategorized"]
        tag0 = tags[0]
        summary = display_title(operation.get("summary"))
        rows.append((method.upper(), path, summary, tag0))
    rows.sort(key=lambda r: (r[3], r[1], r[0]))
    return rows


def generate_endpoints_section(
    spec: dict, href_by_method_path: Dict[Tuple[str, str], str], out_dir: str
) -> str:
    """Markdown for ## Available endpoints from spec + precomputed Mintlify hrefs."""
    rows = list_display_rows(spec)
    by_tag: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for method, path, summary, tag0 in rows:
        by_tag.setdefault(tag0, []).append((method, path, summary, tag0))

    tag_sequence = [t for t in TAG_HEADING_ORDER if t in by_tag]
    for t in sorted(by_tag.keys()):
        if t not in tag_sequence:
            tag_sequence.append(t)

    lines: List[str] = ["## Available endpoints\n\n"]
    for tag in tag_sequence:
        lines.append(f"\n### {tag}\n\n")
        for method, path, summary, _ in by_tag[tag]:
            key = (method, path)
            rel = href_by_method_path.get(key)
            if not rel:
                raise KeyError(f"No computed href for {method} {path}")
            url = f"{PUBLIC_DOCS_ORIGIN}{rel}"
            label = summary if summary else method
            # Backtick the path so MDX renders {param} placeholders literally
            # instead of evaluating them as empty JSX expressions.
            lines.append(f"- **[{method} `{path}`]({url})** - {label}\n")

    _ = out_dir  # reserved if section text ever needs the base path
    return "".join(lines)


def update_landing_page(endpoints_section: str) -> bool:
    """Replace ## Available endpoints through the next ## section in the landing MDX."""
    if not LANDING_PAGE.exists():
        print(f"  ✗ Landing page not found at {LANDING_PAGE}")
        return False

    content = LANDING_PAGE.read_text(encoding="utf-8")
    if "## Available endpoints" not in content and "## Available Endpoints" not in content:
        print("  ✗ No ## Available endpoints section found in landing page")
        return False

    pattern = r"## Available endpoints\n.*?(?=\n## [^#]|\Z)"
    new_content, n = re.subn(
        pattern, endpoints_section.rstrip() + "\n", content, flags=re.DOTALL | re.IGNORECASE
    )
    if n == 0:
        # Legacy title casing
        pattern2 = r"## Available Endpoints\n.*?(?=\n## [^#]|\Z)"
        new_content, n = re.subn(
            pattern2, endpoints_section.rstrip() + "\n", content, flags=re.DOTALL
        )
    if n == 0:
        print("  ✗ Could not replace Available endpoints section")
        return False

    if new_content != content:
        LANDING_PAGE.write_text(new_content, encoding="utf-8")
        print(f"  ✓ Updated {LANDING_PAGE}")
        return True
    print(f"  ✓ No changes needed for {LANDING_PAGE}")
    return False


def main() -> int:
    print("Updating Serverless Training API landing page (Mintlify-aligned links)...")
    try:
        spec = fetch_openapi_spec()
        href_map = build_method_path_to_href(spec, DEFAULT_OPENAPI_DIRECTORY)
        section = generate_endpoints_section(spec, href_map, DEFAULT_OPENAPI_DIRECTORY)
        if update_landing_page(section):
            print("✓ Serverless Training API landing page updated successfully!")
        else:
            print("✓ Serverless Training API landing page is already up to date")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
