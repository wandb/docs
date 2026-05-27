#!/usr/bin/env python3
"""
Update the Service API landing page with the current endpoints from the OpenAPI spec.

This script fetches the OpenAPI spec (either from local file or remote URL) and
updates the Available Endpoints section in the service-api.mdx landing page.

Endpoint links mirror Mintlify's OpenAPI page path logic (@mintlify/scraping
processOpenApiPath):
- Slug folder from the first tag (prepareStringToBeValidFilename).
- Slug file from summary, else "{method}-{slugified-path}".
- Duplicate (tag, slug) pairs get "-1", "-2", ... suffixes.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# Same order as OpenAPIV3.HttpMethods / Mintlify's Object.values(HttpMethods)
HTTP_METHODS_ORDER = ("get", "put", "post", "delete", "options", "head", "patch", "trace")

DEFAULT_OPENAPI_DIRECTORY = "weave/reference/service-api"
PUBLIC_DOCS_ORIGIN = "https://docs.wandb.ai"
LANDING_PAGE = Path("weave/reference/service-api.mdx")

# Display order for ### headings; unknown tags append after these.
TAG_HEADING_ORDER = [
    "Calls",
    "Costs",
    "Feedback",
    "Files",
    "Objects",
    "OpenTelemetry",
    "Refs",
    "Service",
    "Tables",
    "Threads",
    "Inference",
]


def fetch_openapi_spec() -> dict:
    """Fetch OpenAPI spec from local file or remote URL."""
    local_spec = Path("weave/reference/service-api/openapi.json")
    if local_spec.exists():
        print(f"  Using local OpenAPI spec: {local_spec}")
        with open(local_spec, "r", encoding="utf-8") as f:
            return json.load(f)

    print("  Fetching remote OpenAPI spec from https://trace.wandb.ai/openapi.json")
    response = requests.get("https://trace.wandb.ai/openapi.json", timeout=30)
    response.raise_for_status()
    return response.json()


def prepare_string_to_be_valid_filename(value: str | None) -> str | None:
    """Match Mintlify prepareStringToBeValidFilename (apiPages/common.js)."""
    if not value:
        return None
    s = value.replace(" ", "-")
    s = re.sub(r"\{.*?\}", "-", s)
    s = re.sub(r"^-", "", s)
    s = re.sub(r"-$", "", s)
    s = re.sub(r"[{}(),.'\n/]", "", s)
    while "--" in s:
        s = s.replace("--", "-")
    return s.lower()


def generate_unique_filename_without_extension(pages: List[str], base: str) -> str:
    """Match Mintlify generateUniqueFilenameWithoutExtension (apiPages/common.js)."""
    filename = base
    if filename in pages:
        ext = 1
        filename = f"{base}-{ext}"
        while filename in pages:
            ext += 1
            filename = f"{base}-{ext}"
    return filename.lower()


def _is_hidden(operation: dict) -> bool:
    return operation.get("x-hidden") is True


def _is_excluded(operation: dict) -> bool:
    return operation.get("x-excluded") is True


def build_method_path_to_href(spec: dict, out_dir: str) -> Dict[Tuple[str, str], str]:
    """
    Map (HTTP_METHOD, openapi_path) -> site path beginning with /, in Mintlify order.
    """
    out_dir = out_dir.strip("/")
    nav_pages_by_tag: Dict[str, List[str]] = {}
    href_by_key: Dict[Tuple[str, str], str] = {}

    paths = spec.get("paths") or {}
    for path, path_item in paths.items():
        if not path_item or not isinstance(path_item, dict):
            continue
        for method in HTTP_METHODS_ORDER:
            if method not in path_item:
                continue
            operation = path_item[method]
            if not isinstance(operation, dict):
                continue
            if _is_excluded(operation) or _is_hidden(operation):
                continue

            tags = operation.get("tags") or []
            group_name = tags[0] if tags else "API Reference"

            summary = operation.get("summary")
            title = prepare_string_to_be_valid_filename(summary)
            if not title:
                path_part = prepare_string_to_be_valid_filename(path)
                title = f"{method}-{path_part}"

            folder = prepare_string_to_be_valid_filename(group_name) or ""
            base = "/".join(p for p in [out_dir, folder, title] if p)

            pages = nav_pages_by_tag.setdefault(group_name, [])
            filename = generate_unique_filename_without_extension(pages, base)
            pages.append(filename)
            href_by_key[(method.upper(), path)] = f"/{filename}"

    return href_by_key


def list_display_rows(spec: dict) -> List[Tuple[str, str, str, str]]:
    """Rows of (method, path, summary, tag0) sorted for the landing page."""
    rows: List[Tuple[str, str, str, str]] = []
    paths = spec.get("paths") or {}
    for path, path_item in paths.items():
        if not path_item or not isinstance(path_item, dict):
            continue
        for method in HTTP_METHODS_ORDER:
            if method not in path_item:
                continue
            operation = path_item[method]
            if not isinstance(operation, dict):
                continue
            if _is_excluded(operation) or _is_hidden(operation):
                continue
            tags = operation.get("tags") or ["Uncategorized"]
            tag0 = tags[0]
            summary = operation.get("summary") or ""
            rows.append((method.upper(), path, summary, tag0))
    rows.sort(key=lambda r: (r[3], r[1], r[0]))
    return rows


def generate_endpoints_section(
    spec: dict, href_by_method_path: Dict[Tuple[str, str], str]
) -> str:
    """Generate the markdown for the Available Endpoints section."""
    rows = list_display_rows(spec)
    by_tag: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for method, path, summary, tag0 in rows:
        by_tag.setdefault(tag0, []).append((method, path, summary, tag0))

    tag_sequence = [t for t in TAG_HEADING_ORDER if t in by_tag]
    for tag in sorted(by_tag.keys()):
        if tag not in tag_sequence:
            tag_sequence.append(tag)

    lines = ["## Available Endpoints\n\n"]
    for tag in tag_sequence:
        lines.append(f"\n### {tag}\n\n")
        seen_endpoints = set()
        for method, path, summary, _ in by_tag[tag]:
            endpoint_key = (method, path)
            if endpoint_key in seen_endpoints:
                continue
            seen_endpoints.add(endpoint_key)

            rel = href_by_method_path.get(endpoint_key)
            if not rel:
                raise KeyError(f"No computed href for {method} {path}")
            url = f"{PUBLIC_DOCS_ORIGIN}{rel}"
            lines.append(f"- **[{method} `{path}`]({url})** - {summary}\n")

    return "".join(lines)


def update_landing_page(endpoints_section: str) -> bool:
    """Update the service-api.mdx landing page with new endpoints section."""
    if not LANDING_PAGE.exists():
        print(f"  ✗ Landing page not found at {LANDING_PAGE}")
        return False

    content = LANDING_PAGE.read_text(encoding="utf-8")

    # Match from "## Available Endpoints" to the end of file or next H2 section.
    # Use (?!\#) so H3 headings like "### Calls" are not mistaken for H2.
    pattern = r"## Available Endpoints\n.*?(?=\n##(?!\#)|\Z)"

    if not re.search(pattern, content, re.DOTALL):
        print("  ✗ Could not find 'Available Endpoints' section in landing page")
        return False

    new_content = re.sub(pattern, endpoints_section.rstrip(), content, flags=re.DOTALL)

    if new_content != content:
        LANDING_PAGE.write_text(new_content, encoding="utf-8")
        print(f"  ✓ Updated {LANDING_PAGE}")
        return True

    print(f"  ✓ No changes needed for {LANDING_PAGE}")
    return False


def main() -> int:
    """Main function."""
    print("Updating Service API landing page...")

    try:
        spec = fetch_openapi_spec()
        href_map = build_method_path_to_href(spec, DEFAULT_OPENAPI_DIRECTORY)
        total = len(href_map)
        categories = len({tag for _, _, _, tag in list_display_rows(spec)})
        print(f"  Found {total} endpoints in {categories} categories")

        endpoints_section = generate_endpoints_section(spec, href_map)

        if update_landing_page(endpoints_section):
            print("✓ Service API landing page updated successfully!")
        else:
            print("✓ Service API landing page is already up to date")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
