#!/usr/bin/env python3
"""
Mintlify OpenAPI page path derivation, ported from @mintlify/scraping processOpenApiPath.

Mintlify builds one page per operation and derives its URL as
`{directory}/{slugified first tag}/{slugified summary}`:

- Folder from the first tag (prepareStringToBeValidFilename).
- File from summary, else "{method}-{slugified-path}".
- Duplicate (tag, slug) pairs get "-1", "-2", ... suffixes, in spec path order then
  OpenAPI HTTP method order (get, put, post, delete, ...).

This module is the single source of truth for that algorithm. It is shared by the
landing-page generators and by generate_openapi_stubs.py so a page's URL is derived
identically everywhere.

Verified against https://docs.wandb.ai/sitemap.xml: build_method_path_to_href reproduces
all 115 published weave/reference/service-api pages and all 12
serverless-training/api-reference pages exactly.
"""
from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional, Tuple

# Same order as OpenAPIV3.HttpMethods / Mintlify's Object.values(HttpMethods)
HTTP_METHODS_ORDER = ("get", "put", "post", "delete", "options", "head", "patch", "trace")

PUBLIC_DOCS_ORIGIN = "https://docs.wandb.ai"

# Operations with no tag land in this group, matching Mintlify's fallback.
UNTAGGED_GROUP = "API Reference"

# Acronyms that upstream summary generation title-cases into unreadable forms. The service
# defines e.g. Create_SFT_Training_Job, but FastAPI derives summary "Create Sft Training
# Job", so the rendered page reads "Sft". Restoring these is display-only and cannot move a
# URL: slug derivation lowercases everything, so "Create SFT Training Job" and "Create Sft
# Training Job" both slugify to "create-sft-training-job". Extend as new ones appear.
DISPLAY_ACRONYMS = {
    "Sft": "SFT",
    "Rl": "RL",
    "Genai": "GenAI",
    "Api": "API",
    "Id": "ID",
    "Ids": "IDs",
    "Json": "JSON",
    "Llm": "LLM",
    "Otel": "OTel",
    "Sdk": "SDK",
    "Ui": "UI",
    "Uri": "URI",
    "Url": "URL",
}

_ACRONYM_RE = re.compile(r"\b(" + "|".join(sorted(DISPLAY_ACRONYMS)) + r")\b")


def display_title(summary: Optional[str]) -> str:
    """
    A summary rendered for humans, with title-cased acronyms restored.

    Used for page titles and landing-page labels only — never for slugs, so that fixing
    how a title reads never changes where the page lives.
    """
    if not summary:
        return ""
    return _ACRONYM_RE.sub(lambda m: DISPLAY_ACRONYMS[m.group(0)], summary)


def prepare_string_to_be_valid_filename(value: Optional[str]) -> Optional[str]:
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


def is_hidden(operation: dict) -> bool:
    """True when Mintlify should not build a page for this operation at all."""
    return operation.get("x-hidden") is True


def is_nav_excluded(operation: dict) -> bool:
    """
    True when the operation's page is built but omitted from navigation.

    Mintlify honors `x-exclude` by hiding the page from the sidebar while still
    publishing it — the five `x-exclude` operations in the Weave spec are all present
    in the production sitemap. `x-excluded` is accepted as a legacy spelling.

    IMPORTANT: this must not gate URL derivation. An excluded page still has a URL, and
    links to it (from landing pages or prose) must resolve. Use it only when deciding
    what to list in navigation.
    """
    return operation.get("x-exclude") is True or operation.get("x-excluded") is True


def group_name_for(operation: dict) -> str:
    """The nav group an operation belongs to: its first tag, else the untagged fallback."""
    tags = operation.get("tags") or []
    return tags[0] if tags else UNTAGGED_GROUP


def iter_operations(spec: dict) -> Iterator[Tuple[str, str, dict]]:
    """
    Yield (method_lower, path, operation) in Mintlify's order: spec path order, then
    HTTP_METHODS_ORDER within each path. Skips hidden operations, which are never built.
    """
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
            if is_hidden(operation):
                continue
            yield method, path, operation


def slug_for(operation: dict, path: str, method: str) -> str:
    """The page's own slug segment: from summary, else '{method}-{slugified-path}'."""
    title = prepare_string_to_be_valid_filename(operation.get("summary"))
    if not title:
        title = f"{method}-{prepare_string_to_be_valid_filename(path)}"
    return title


def build_method_path_to_href(spec: dict, out_dir: str) -> Dict[Tuple[str, str], str]:
    """
    Map (HTTP_METHOD, openapi_path) -> site path beginning with "/", in Mintlify order.

    Includes nav-excluded operations, because their pages are published and linkable.
    """
    out_dir = out_dir.strip("/")
    nav_pages_by_tag: Dict[str, List[str]] = {}
    href_by_key: Dict[Tuple[str, str], str] = {}

    for method, path, operation in iter_operations(spec):
        group_name = group_name_for(operation)
        folder = prepare_string_to_be_valid_filename(group_name) or ""
        base = "/".join(p for p in [out_dir, folder, slug_for(operation, path, method)] if p)

        pages = nav_pages_by_tag.setdefault(group_name, [])
        filename = generate_unique_filename_without_extension(pages, base)
        pages.append(filename)
        href_by_key[(method.upper(), path)] = f"/{filename}"

    return href_by_key


def ordered_groups(spec: dict, tag_heading_order: List[str]) -> List[str]:
    """
    Group names in display order: those in tag_heading_order first, then any remaining
    groups in the order they appear in the spec.
    """
    present: List[str] = []
    for _, _, operation in iter_operations(spec):
        name = group_name_for(operation)
        if name not in present:
            present.append(name)

    ordered = [t for t in tag_heading_order if t in present]
    ordered += [t for t in present if t not in ordered]
    return ordered
