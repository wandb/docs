#!/usr/bin/env python3
"""
Generate one stub MDX page per OpenAPI operation, and the matching English nav.

Mintlify can build API reference pages two ways. With `openapi.directory` in docs.json it
autogenerates them, deriving each URL from `tag` + slugified `summary` — so a translated
`summary` moves the page, which breaks English links carried into translated MDX and
produces non-ASCII URLs. With a stub page per operation, the URL comes from the
**filename** and the operation binding from **method + path**, neither of which is ever
translated. Locadex then translates each stub's `title` and body like any other MDX,
mirroring it to `{locale}/...` under the same filename.

This is the approach Mintlify uses in its own docs (mintlify/docs: `api/<group>/<op>.mdx`
mirrored to `fr/api/<group>/<op>.mdx`, no `openapi` block in nav).

Design rules:

- **Create if absent; never overwrite a body.** Hand-written prose per operation is the
  main reason to use stubs, so an existing file is only ever touched to correct its
  machine-owned `openapi:` line.
- **Never rename.** A stub's filename is frozen once created, even if `summary` changes.
  That is what makes URLs stable. A genuine API path change appears as a prune plus a
  create in the sync PR, where a human reviews it.
- **`title` is set at creation only**, so a curated title is not clobbered on later syncs.
- **Nav omits `x-exclude` operations** but their stubs are still generated, because
  Mintlify publishes those pages — they are simply hidden from the sidebar.

Usage:
    python scripts/reference-generation/common/generate_openapi_stubs.py --spec weave
    python scripts/reference-generation/common/generate_openapi_stubs.py --spec all --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.mintlify_openapi_paths import (  # noqa: E402
    build_method_path_to_href,
    group_name_for,
    is_nav_excluded,
    iter_operations,
    ordered_groups,
)

DOCS_JSON = Path("docs.json")

Key = Tuple[str, str]  # (HTTP_METHOD, openapi path)


@dataclass
class SpecTarget:
    """One OpenAPI spec and where its generated pages live in the repo and the nav."""

    key: str
    spec_path: str
    directory: str
    #: A page already listed in the nav group, used to locate that group unambiguously.
    #: Chosen because it survives removing the `openapi` block, unlike the block itself.
    anchor_page: str
    group_name: str
    tag_order: List[str] = field(default_factory=list)


TARGETS: Dict[str, SpecTarget] = {
    "weave": SpecTarget(
        key="weave",
        spec_path="weave/reference/service-api/openapi.json",
        directory="weave/reference/service-api",
        anchor_page="weave/reference/service-api",
        group_name="Service API",
        tag_order=[
            "Calls", "Costs", "Feedback", "Files", "Objects", "OpenTelemetry",
            "Refs", "Service", "Tables", "Threads", "Inference",
        ],
    ),
    "serverless-training": SpecTarget(
        key="serverless-training",
        spec_path="serverless-training/api-reference/openapi.json",
        directory="serverless-training/api-reference",
        anchor_page="serverless-training/api-reference",
        group_name="Serverless Training",
        tag_order=["chat-completions", "models", "training-jobs", "health"],
    ),
}

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---(\n|\Z)", re.DOTALL)
# "openapi: /path/to/openapi.json POST /some/{path}"
OPENAPI_LINE_RE = re.compile(
    r"^openapi:\s*(?P<spec>\S+)\s+(?P<method>[A-Za-z]+)\s+(?P<path>\S+)\s*$"
)


# --------------------------------------------------------------------------- stubs


def openapi_directive(target: SpecTarget, method: str, api_path: str) -> str:
    """The machine-owned frontmatter line binding this page to one operation."""
    return f"openapi: /{target.spec_path} {method} {api_path}"


def render_stub(target: SpecTarget, title: str, method: str, api_path: str) -> str:
    """A new stub: frontmatter only. Mintlify renders the operation from the spec."""
    return (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        f"{openapi_directive(target, method, api_path)}\n"
        "---\n"
    )


def parse_stub(path: Path) -> Optional[Tuple[str, str, str]]:
    """Return (spec_path, METHOD, api_path) from a stub's `openapi:` line, or None."""
    match = FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    if not match:
        return None
    for line in match.group(1).splitlines():
        found = OPENAPI_LINE_RE.match(line.strip())
        if found:
            return (
                found.group("spec").lstrip("/"),
                found.group("method").upper(),
                found.group("path"),
            )
    return None


def scan_existing_stubs(target: SpecTarget) -> Tuple[Dict[Key, Path], List[Path]]:
    """
    Map (METHOD, path) -> stub file, for every stub under the target directory.

    Returns the map plus any .mdx files that look like stubs but could not be parsed,
    which are reported and then left strictly alone.
    """
    root = Path(target.directory)
    found: Dict[Key, Path] = {}
    unparsed: List[Path] = []
    if not root.exists():
        return found, unparsed

    for mdx in sorted(root.rglob("*.mdx")):
        # Only files one level below the directory are generated stubs
        # (<directory>/<tag>/<slug>.mdx); the landing page sits alongside.
        if mdx.parent == root:
            continue
        parsed = parse_stub(mdx)
        if parsed is None:
            unparsed.append(mdx)
            continue
        spec_path, method, api_path = parsed
        if spec_path != target.spec_path:
            unparsed.append(mdx)
            continue
        found[(method, api_path)] = mdx
    return found, unparsed


def sync_stubs(
    target: SpecTarget, spec: dict, dry_run: bool
) -> Tuple[Dict[Key, str], List[str], List[str], List[str]]:
    """
    Create, correct, and prune stubs so they match the spec.

    Returns (page_by_key, created, rebound, pruned) where page_by_key maps each operation
    to the repo-relative page path (no .mdx) actually on disk — which for a pre-existing
    stub is its frozen filename, not necessarily the freshly derived slug.
    """
    href_by_key = build_method_path_to_href(spec, target.directory)
    existing, unparsed = scan_existing_stubs(target)

    for path in unparsed:
        print(f"    ! not a recognized stub, leaving untouched: {path}")

    titles: Dict[Key, str] = {}
    for method, api_path, operation in iter_operations(spec):
        titles[(method.upper(), api_path)] = operation.get("summary") or api_path

    page_by_key: Dict[Key, str] = {}
    created: List[str] = []
    rebound: List[str] = []

    for key, href in href_by_key.items():
        method, api_path = key
        if key in existing:
            # Frozen filename. Only the machine-owned openapi: line may be corrected.
            path = existing[key]
            page_by_key[key] = path.with_suffix("").as_posix()
            text = path.read_text(encoding="utf-8")
            wanted = openapi_directive(target, method, api_path)
            lines = text.splitlines(keepends=True)
            for i, line in enumerate(lines):
                if OPENAPI_LINE_RE.match(line.strip()):
                    if line.rstrip("\n") != wanted:
                        lines[i] = wanted + "\n"
                        rebound.append(path.as_posix())
                        if not dry_run:
                            path.write_text("".join(lines), encoding="utf-8")
                    break
            continue

        page_path = href.lstrip("/")
        target_file = Path(page_path + ".mdx")
        page_by_key[key] = page_path
        created.append(target_file.as_posix())
        if not dry_run:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(
                render_stub(target, titles[key], method, api_path), encoding="utf-8"
            )

    pruned: List[str] = []
    for key, path in sorted(existing.items()):
        if key not in href_by_key:
            pruned.append(path.as_posix())
            if not dry_run:
                path.unlink()

    return page_by_key, created, rebound, pruned


# ----------------------------------------------------------------------------- nav


def build_nav_groups(
    target: SpecTarget, spec: dict, page_by_key: Dict[Key, str]
) -> List[dict]:
    """One nav group per tag, in display order, listing pages in Mintlify order."""
    pages_by_group: Dict[str, List[str]] = {}
    for method, api_path, operation in iter_operations(spec):
        if is_nav_excluded(operation):
            continue
        page = page_by_key.get((method.upper(), api_path))
        if page is None:
            continue
        pages_by_group.setdefault(group_name_for(operation), []).append(page)

    groups = []
    for name in ordered_groups(spec, target.tag_order):
        pages = pages_by_group.get(name)
        if pages:
            groups.append({"group": name, "pages": pages})
    return groups


def find_nav_group(docs: dict, target: SpecTarget) -> dict:
    """Locate the English nav group for this spec via its anchor page."""
    english = docs["navigation"]["languages"][0]
    if english.get("language") != "en":
        raise SystemExit("  ✗ navigation.languages[0] is not the English block")

    def walk(node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    hits = [
        node for node in walk(english)
        if isinstance(node.get("pages"), list) and target.anchor_page in node["pages"]
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"  ✗ expected exactly 1 nav group containing {target.anchor_page!r}, "
            f"found {len(hits)}"
        )
    group = hits[0]
    if group.get("group") != target.group_name:
        print(
            f"    ! nav group is named {group.get('group')!r}, "
            f"expected {target.group_name!r} — continuing"
        )
    return group


def is_generated_group(entry, target: SpecTarget) -> bool:
    """True for a group this script previously wrote (all pages under our directory)."""
    if not isinstance(entry, dict) or "pages" not in entry:
        return False
    pages = entry["pages"]
    return bool(pages) and all(
        isinstance(p, str) and p.startswith(target.directory + "/") for p in pages
    )


def update_docs_json(target: SpecTarget, nav_groups: List[dict], dry_run: bool) -> bool:
    """
    Replace the target group's generated children and drop its `openapi` block.

    docs.json is rewritten with indent=2, ensure_ascii=False and no trailing newline,
    which round-trips the current file byte-for-byte, so the diff shows only real changes.
    """
    raw = DOCS_JSON.read_text(encoding="utf-8")
    docs = json.loads(raw)
    group = find_nav_group(docs, target)

    preserved = [
        entry for entry in group.get("pages", [])
        if not is_generated_group(entry, target)
    ]
    new_pages = preserved + nav_groups

    had_openapi = "openapi" in group
    changed = had_openapi or group.get("pages") != new_pages

    group["pages"] = new_pages
    group.pop("openapi", None)

    if not changed:
        print("    ✓ docs.json already up to date")
        return False

    out = json.dumps(docs, indent=2, ensure_ascii=False)
    if dry_run:
        print(
            f"    would update docs.json: {len(nav_groups)} group(s), "
            f"{sum(len(g['pages']) for g in nav_groups)} page(s)"
            f"{', removing openapi block' if had_openapi else ''}"
        )
        return True

    DOCS_JSON.write_text(out, encoding="utf-8")
    print(
        f"    ✓ docs.json updated: {len(nav_groups)} group(s), "
        f"{sum(len(g['pages']) for g in nav_groups)} page(s)"
        f"{', removed openapi block' if had_openapi else ''}"
    )
    return True


# ---------------------------------------------------------------------------- main


def run(target: SpecTarget, dry_run: bool) -> int:
    print(f"  {target.key}: {target.spec_path}")
    spec_file = Path(target.spec_path)
    if not spec_file.exists():
        print(f"    ✗ spec not found at {spec_file}")
        return 1

    with open(spec_file, encoding="utf-8") as handle:
        spec = json.load(handle)

    page_by_key, created, rebound, pruned = sync_stubs(target, spec, dry_run)
    verb = "would " if dry_run else ""
    print(
        f"    {verb}create {len(created)}, {verb}rebind {len(rebound)}, "
        f"{verb}prune {len(pruned)}  (total operations: {len(page_by_key)})"
    )
    for path in created[:10]:
        print(f"      + {path}")
    if len(created) > 10:
        print(f"      + ... and {len(created) - 10} more")
    for path in pruned:
        print(f"      - {path}")

    nav_groups = build_nav_groups(target, spec, page_by_key)
    listed = sum(len(g["pages"]) for g in nav_groups)
    hidden = len(page_by_key) - listed
    if hidden:
        print(f"    {hidden} operation(s) generated but omitted from nav (x-exclude)")

    update_docs_json(target, nav_groups, dry_run)

    # Machine-readable line for the sync workflows to surface in the PR body. A rename
    # upstream shows up here as created>0 together with pruned>0, which is the signal a
    # reviewer needs — otherwise it is two files lost in a large spec diff.
    print(
        f"STUBS_SUMMARY spec={target.key} created={len(created)} "
        f"pruned={len(pruned)} rebound={len(rebound)} "
        f"operations={len(page_by_key)} nav_pages={listed} nav_groups={len(nav_groups)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", default="all", choices=[*TARGETS, "all"],
        help="which spec to generate stubs for (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would change without writing anything",
    )
    args = parser.parse_args()

    print("Generating OpenAPI stub pages..." + (" (dry run)" if args.dry_run else ""))
    keys = list(TARGETS) if args.spec == "all" else [args.spec]
    status = 0
    for key in keys:
        status |= run(TARGETS[key], args.dry_run)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
