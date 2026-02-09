#!/usr/bin/env python3
"""
Sync localized navigation with English navigation in docs.json.

What this script does:
- Copies the English navigation structure and uses it as the base for ja and ko.
- Prefixes localized page paths with the language code.
- Removes localized navigation entries when the corresponding file does not exist.
- Reports localized files that exist on disk but are not referenced in navigation.

File existence checks:
- Accepts explicit file paths with extensions.
- For extensionless page paths, checks for:
  - <page>.mdx, <page>.md, <page>.ipynb
  - <page>/index.mdx, <page>/index.md, <page>/index.ipynb

Usage:
  python scripts/sync_localized_navigation.py
  python scripts/sync_localized_navigation.py --docs-json path/to/docs.json
  python scripts/sync_localized_navigation.py --languages ja,ko

Remote English navigation source:
  Use --remote-docs-json-url to supply an older docs.json as the English source
  while still writing updates into the local docs.json.

  Example with your URL:
    python scripts/sync_localized_navigation.py --remote-docs-json-url "https://raw.githubusercontent.com/wandb/docs/ec481ab84221f543df0622546f03c8328fe7fa00/docs.json"
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen


def prefix_path(path: str, language: str) -> str:
    if path.startswith(("http://", "https://", "/")):
        return path
    if path.startswith(f"{language}/"):
        return path
    return f"{language}/{path}"


def is_external_path(path: str) -> bool:
    return path.startswith(("http://", "https://", "/"))


def page_exists(page: str, repo_root: Path) -> bool:
    if is_external_path(page):
        return True

    page_path = Path(page)
    if page_path.suffix:
        return (repo_root / page_path).is_file()

    for ext in (".mdx", ".md", ".ipynb"):
        if (repo_root / f"{page}{ext}").is_file():
            return True
        if (repo_root / page / f"index{ext}").is_file():
            return True

    return False


def page_from_file(path: Path) -> str | None:
    if path.suffix not in {".mdx", ".md", ".ipynb"}:
        return None
    if path.stem == "index":
        return str(path.parent.as_posix())
    return str(path.with_suffix("").as_posix())


def collect_nav_pages(pages: list[Any]) -> set[str]:
    collected: set[str] = set()
    for item in pages:
        if isinstance(item, str):
            if not is_external_path(item):
                collected.add(item)
            continue
        if isinstance(item, dict) and isinstance(item.get("pages"), list):
            collected.update(collect_nav_pages(item["pages"]))
    return collected


def list_language_files(repo_root: Path, language: str) -> set[str]:
    language_root = repo_root / language
    if not language_root.is_dir():
        return set()
    pages: set[str] = set()
    for path in language_root.rglob("*"):
        if not path.is_file():
            continue
        page = page_from_file(path)
        if page:
            pages.add(page)
    return pages


def sync_pages(pages: list[Any], language: str, repo_root: Path) -> list[Any]:
    updated: list[Any] = []
    for item in pages:
        if isinstance(item, str):
            localized = prefix_path(item, language)
            if page_exists(localized, repo_root):
                updated.append(localized)
            else:
                print(f"Removed missing page: {localized}")
            continue
        if isinstance(item, dict):
            new_item: dict[str, Any] = {}
            for key, value in item.items():
                if key == "pages" and isinstance(value, list):
                    new_pages = sync_pages(value, language, repo_root)
                    if not new_pages:
                        new_item = {}
                        break
                    new_item[key] = new_pages
                else:
                    new_item[key] = value
            if new_item:
                updated.append(new_item)
            continue
        updated.append(item)
    return updated


def sync_entry(
    en_entry: dict[str, Any],
    language: str,
    repo_root: Path,
) -> dict[str, Any]:
    entry = copy.deepcopy(en_entry)
    entry["language"] = language
    if "tabs" in entry and isinstance(entry["tabs"], list):
        updated_tabs: list[Any] = []
        for tab in entry["tabs"]:
            if isinstance(tab, dict) and isinstance(tab.get("pages"), list):
                new_tab = copy.deepcopy(tab)
                new_tab["pages"] = sync_pages(new_tab["pages"], language, repo_root)
                updated_tabs.append(new_tab)
            else:
                updated_tabs.append(tab)
        entry["tabs"] = updated_tabs
    return entry


def sync_navigation(
    data: dict[str, Any],
    languages: Iterable[str],
    repo_root: Path,
    en_source: dict[str, Any] | None,
) -> None:
    # Use the local docs.json for output, but optionally load English navigation
    # from a remote docs.json to mirror a historical snapshot.
    navigation = data.get("navigation")
    if not isinstance(navigation, dict):
        raise ValueError("docs.json is missing a navigation object.")

    language_entries = navigation.get("languages")
    if not isinstance(language_entries, list):
        raise ValueError("docs.json navigation.languages is missing or invalid.")

    en_entry_source = en_source or data
    en_navigation = en_entry_source.get("navigation")
    if not isinstance(en_navigation, dict):
        raise ValueError("English source docs.json is missing a navigation object.")
    en_language_entries = en_navigation.get("languages")
    if not isinstance(en_language_entries, list):
        raise ValueError("English source navigation.languages is missing or invalid.")

    en_entry = next(
        (entry for entry in en_language_entries if entry.get("language") == "en"),
        None,
    )
    if not isinstance(en_entry, dict):
        raise ValueError("English source navigation.languages does not include English.")

    languages = list(languages)
    kept_entries = [
        entry
        for entry in language_entries
        if entry.get("language") not in languages
    ]

    for language in languages:
        kept_entries.append(sync_entry(en_entry, language, repo_root))

    navigation["languages"] = kept_entries

    for language in languages:
        language_entry = next(
            (
                entry
                for entry in navigation["languages"]
                if entry.get("language") == language
            ),
            None,
        )
        if not language_entry:
            continue
        tabs = language_entry.get("tabs", [])
        nav_pages: set[str] = set()
        if isinstance(tabs, list):
            for tab in tabs:
                if isinstance(tab, dict) and isinstance(tab.get("pages"), list):
                    nav_pages.update(collect_nav_pages(tab["pages"]))

        # Compare navigation pages to localized files on disk and report orphans.
        file_pages = list_language_files(repo_root, language)
        missing_from_nav = sorted(file_pages - nav_pages)
        for page in missing_from_nav:
            print(f"Unreferenced page: {page}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync localized navigation with English navigation.",
        epilog=__doc__.strip() if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--docs-json",
        default="docs.json",
        help="Path to docs.json. Optional. Default: docs.json.",
    )
    parser.add_argument(
        "--languages",
        default="ja,ko",
        help="Comma-separated languages to sync. Optional. Default: ja,ko.",
    )
    parser.add_argument(
        "--remote-docs-json-url",
        default="",
        help=(
            "URL to a docs.json to use for English navigation. Optional. "
            "Default: empty."
        ),
    )
    return parser.parse_args()


def load_remote_docs_json(url: str) -> dict[str, Any]:
    # Load a remote docs.json to use as the English navigation source.
    with urlopen(url) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def main() -> None:
    args = parse_args()
    docs_json_path = Path(args.docs_json)
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    en_source: dict[str, Any] | None = None

    if not docs_json_path.is_file():
        raise FileNotFoundError(f"docs.json not found: {docs_json_path}")

    if args.remote_docs_json_url:
        # Pull English navigation from the remote docs.json.
        en_source = load_remote_docs_json(args.remote_docs_json_url)

    data = json.loads(docs_json_path.read_text(encoding="utf-8"))
    # Update navigation, then write back to the local docs.json.
    sync_navigation(data, languages, docs_json_path.parent, en_source)

    docs_json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    joined_languages = ", ".join(languages)
    print(f"Synchronized navigation for: {joined_languages}")


if __name__ == "__main__":
    main()
