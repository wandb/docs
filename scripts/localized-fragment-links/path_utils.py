"""
Path resolution for localized docs (ja/, ko/) and their EN counterparts.

- Page key: stable identifier for a page (e.g. "ja/launch/launch-terminology").
- EN counterpart: localized path -> path to the English MDX file.
- Link resolution: href + current file -> (page_key, fragment) for localized targets.
"""

from pathlib import Path


def get_en_counterpart(localized_path: Path, repo_root: Path) -> Path | None:
    """
    Return the path to the English MDX counterpart, or None if not found.

    Rules (for this repo):
    - ja/launch/... or ko/launch/... -> platform/launch/... (EN launch is under platform)
    - ja/platform/... or ko/platform/... -> platform/...
    - ja/<other>/..., ko/<other>/... -> <other>/...
    """
    try:
        rel = localized_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    locale = parts[0]
    if locale not in ("ja", "ko"):
        return None
    first_seg = parts[1]
    rest_list = list(parts[2:])
    rest_str = "/".join([first_seg] + rest_list)
    # EN launch content lives under platform/launch (ja/launch/... -> platform/launch/...)
    if first_seg == "launch":
        en_relative = "platform/" + rest_str
    else:
        en_relative = rest_str
    en_path = repo_root / en_relative
    return en_path if en_path.is_file() else None


def path_to_page_key(path: Path, repo_root: Path) -> str | None:
    """
    Convert a path to a page key: forward slashes, no .mdx, relative to repo.

    Example: repo_root/ja/launch/foo.mdx -> "ja/launch/foo"
    """
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    s = str(rel).replace("\\", "/")
    if s.endswith(".mdx"):
        s = s[:-4]
    return s


def resolve_link_target(
    href: str,
    from_path: Path,
    repo_root: Path,
    locale: str,
) -> tuple[str, str] | None:
    """
    Resolve an href (from a localized file) to (page_key, fragment).

    Returns None if the link does not point to a localized page with a fragment.
    - href: value inside (...) e.g. "/ja/launch/terminology#launch-job" or "#walkthrough"
    - from_path: path to the MDX file containing the link
    - locale: "ja" or "ko" (the locale of from_path)
    """
    if "#" not in href:
        return None
    path_part, fragment = href.split("#", 1)
    fragment = fragment.strip()
    if not fragment:
        return None

    try:
        from_rel = from_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    from_parts = from_rel.parts
    if from_parts[0] not in ("ja", "ko"):
        return None

    # Same-page link
    if not path_part or path_part == "#":
        page_key = path_to_page_key(from_path, repo_root)
        return (page_key, fragment) if page_key else None

    # Absolute-like: /ja/launch/... or /ko/...
    path_part = path_part.lstrip("/")
    if path_part.startswith("ja/") or path_part.startswith("ko/"):
        if path_part.endswith(".mdx"):
            path_part = path_part[:-4]
        return (path_part, fragment)

    # Relative: resolve from from_path's directory
    from_dir = from_path.parent
    resolved = (from_dir / path_part).resolve()
    try:
        resolved = resolved.relative_to(repo_root.resolve())
    except ValueError:
        return None
    page_key = path_to_page_key(resolved, repo_root)
    if page_key is None:
        return None
    if not (page_key.startswith("ja/") or page_key.startswith("ko/")):
        return None
    return (page_key, fragment)
