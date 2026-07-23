#!/usr/bin/env python3
"""Update the "New articles" log page (release-notes/new-articles.mdx).

Finds pull requests merged into main during the lookback window, collects the
English .mdx articles those PRs added, and inserts them into the page under a
"Month YYYY" heading. Months from the current year are top-level headings;
months from prior years are collapsed inside an <Accordion title="YYYY"> block.

The script is idempotent: articles already listed on the page are skipped, so
overlapping lookback windows never produce duplicate entries.

Environment variables:
    GITHUB_TOKEN or GH_TOKEN  Token for the GitHub API (optional, but avoids
                              the low unauthenticated rate limit).
    GITHUB_REPOSITORY         Repository to query (default: wandb/docs).
    LOOKBACK_DAYS             How many days of merged PRs to scan (default: 8,
                              one day more than the weekly cadence so runs
                              overlap rather than leave gaps).

Run from the repository root, with the target branch checked out:
    python scripts/new-articles/update_new_articles.py
"""

import os
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

REPO = os.environ.get("GITHUB_REPOSITORY", "wandb/docs")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "8"))
PAGE_PATH = "release-notes/new-articles.mdx"
START_MARKER = "{/* new-articles:start */}"
END_MARKER = "{/* new-articles:end */}"

# Paths that never count as new articles: localized content, reusable
# snippets, and machine-generated reference docs.
EXCLUDED_PREFIXES = (
    "fr/",
    "ja/",
    "ko/",
    "snippets/",
    "models/ref/",
    "weave/reference/",
    "docengine",
    ".github/",
    PAGE_PATH,
)

# Support tag index pages (support/<product>/tags/...) are listings, not
# articles.
EXCLUDED_PATH_RE = re.compile(r"^support/[^/]+/tags/")

# Display labels for each article's service, keyed by top-level directory.
# Directories not listed here fall back to the title-cased directory name.
CATEGORY_LABELS = {
    "models": "Models",
    "weave": "Weave",
    "platform": "Platform",
    "inference": "Inference",
    "serverless-training": "Serverless Training",
    "sandboxes": "Sandboxes",
    "support": "Support",
    "release-notes": "Release Notes",
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTH_HEADING_RE = re.compile(
    r"^#{2,3} (" + "|".join(MONTH_NAMES) + r") (\d{4})\s*$"
)
ACCORDION_OPEN_RE = re.compile(r'^<Accordion title="(\d{4})">\s*$')
ACCORDION_CLOSE_RE = re.compile(r"^</Accordion>\s*$")
BULLET_RE = re.compile(r"^- (\[|\*\*)")
FRONTMATTER_FIELD_RE = re.compile(
    r'^(title|description):\s*(?:"(.*)"|\'(.*)\'|(.*?))\s*$'
)


def api_get(session, url, params=None):
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response


def merged_prs_since(session, since):
    """Return merged PRs (number, merged_at) targeting main, newest first."""
    prs = []
    query = (
        f"repo:{REPO} is:pr is:merged base:main "
        f"merged:>={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    page = 1
    while True:
        response = api_get(
            session,
            "https://api.github.com/search/issues",
            params={"q": query, "per_page": 100, "page": page,
                    "sort": "created", "order": "desc"},
        )
        items = response.json().get("items", [])
        prs.extend(items)
        if len(items) < 100 or page >= 10:
            break
        page += 1
    return prs


def added_article_paths(session, pr_number):
    """Return .mdx files added by a PR, excluding non-article paths."""
    paths = []
    page = 1
    while True:
        response = api_get(
            session,
            f"https://api.github.com/repos/{REPO}/pulls/{pr_number}/files",
            params={"per_page": 100, "page": page},
        )
        files = response.json()
        for f in files:
            path = f["filename"]
            if (
                f["status"] == "added"
                and path.endswith(".mdx")
                and not path.startswith(EXCLUDED_PREFIXES)
                and not EXCLUDED_PATH_RE.match(path)
            ):
                paths.append(path)
        if len(files) < 100:
            break
        page += 1
    return paths


def read_frontmatter(path):
    """Return (title, description) from a page's frontmatter, or None."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    title = description = ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = FRONTMATTER_FIELD_RE.match(line)
        if match:
            value = next(v for v in match.groups()[1:] if v is not None)
            if match.group(1) == "title":
                title = value
            else:
                description = value
    return (title, description) if title else None


def category_label(path):
    """Return the service label for an article, from its top-level directory."""
    if "/" not in path:
        return "General"
    top = path.split("/", 1)[0]
    return CATEGORY_LABELS.get(top, top.replace("-", " ").title())


# Lines that can't start a body paragraph: imports, JSX components, headings,
# comments, code fences, blockquotes, lists, tables, and images.
NON_PARAGRAPH_RE = re.compile(r"^(import |export |<|#|\{/\*|```|>|[-*] |\d+\. |\||!\[)")
MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")


def first_body_sentence(path):
    """Return the first sentence of an article's body, as a plain string."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    # Skip the frontmatter block.
    body_start = 0
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                body_start = i + 1
                break
    paragraph = []
    for line in lines[body_start:]:
        if not line.strip() or NON_PARAGRAPH_RE.match(line.strip()):
            if paragraph:
                break
            continue
        paragraph.append(line.strip())
    text = " ".join(paragraph)
    text = MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"[*_`]", "", text)
    sentence = SENTENCE_END_RE.split(text, 1)[0].strip()
    return sentence if len(sentence) <= 300 else ""


def page_url(path):
    url = "/" + path.removesuffix(".mdx")
    url = url.removesuffix("/index") or "/"
    return url + "/" if not url.endswith("/") else url


def parse_generated_region(region):
    """Parse the generated region into {(year, month_index): [bullet lines]}."""
    sections = {}
    current = None
    for line in region.splitlines():
        heading = MONTH_HEADING_RE.match(line)
        if heading:
            key = (int(heading.group(2)), MONTH_NAMES.index(heading.group(1)) + 1)
            current = sections.setdefault(key, [])
        elif BULLET_RE.match(line) and current is not None:
            current.append(line.rstrip())
        elif ACCORDION_OPEN_RE.match(line) or ACCORDION_CLOSE_RE.match(line):
            current = None
        elif line.strip() and current is not None and not line.startswith("#"):
            # Continuation of a wrapped bullet; keep it with its entry.
            if current:
                current[-1] += "\n" + line.rstrip()
    return sections


def render_generated_region(sections, current_year):
    """Render sections newest-first, collapsing prior years into accordions."""
    output = []
    keys = sorted(sections, reverse=True)
    open_accordion_year = None
    for year, month in keys:
        if not sections[(year, month)]:
            continue
        if year >= current_year:
            heading_level = "##"
        else:
            if open_accordion_year != year:
                if open_accordion_year is not None:
                    output.append("</Accordion>")
                    output.append("")
                output.append(f'<Accordion title="{year}">')
                output.append("")
                open_accordion_year = year
            heading_level = "###"
        output.append(f"{heading_level} {MONTH_NAMES[month - 1]} {year}")
        output.append("")
        output.extend(sections[(year, month)])
        output.append("")
    if open_accordion_year is not None:
        output.append("</Accordion>")
        output.append("")
    return "\n".join(output).strip("\n")


def main():
    if not os.path.exists(PAGE_PATH):
        sys.exit(f"error: {PAGE_PATH} not found; run from the repository root")

    session = requests.Session()
    session.headers["Accept"] = "application/vnd.github+json"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    prs = merged_prs_since(session, since)
    print(f"Found {len(prs)} PRs merged into main since {since.date()}")

    with open(PAGE_PATH, encoding="utf-8") as f:
        page = f.read()
    if START_MARKER not in page or END_MARKER not in page:
        sys.exit(f"error: markers not found in {PAGE_PATH}")
    head, rest = page.split(START_MARKER, 1)
    region, tail = rest.split(END_MARKER, 1)
    sections = parse_generated_region(region)

    added = 0
    # Search results are newest-first; walk oldest-first so that prepending
    # each new entry leaves the newest article at the top of its month.
    for pr in reversed(prs):
        merged_at = datetime.fromisoformat(
            pr["pull_request"]["merged_at"].replace("Z", "+00:00")
        )
        for path in added_article_paths(session, pr["number"]):
            url = page_url(path)
            if f"]({url})" in region or any(
                f"]({url})" in line for lines in sections.values() for line in lines
            ):
                continue
            frontmatter = read_frontmatter(path)
            if frontmatter is None:
                print(f"  skipping {path}: deleted or missing a title")
                continue
            title, description = frontmatter
            entry = f"- **{category_label(path)}**: [{title}]({url})"
            description = description or first_body_sentence(path)
            if description:
                entry += f": {description}"
            key = (merged_at.year, merged_at.month)
            sections.setdefault(key, []).insert(0, entry)
            added += 1
            print(f"  added {url} (PR #{pr['number']}, {merged_at.date()})")

    if added == 0:
        print("No new articles found; page left unchanged")
        return

    current_year = datetime.now(timezone.utc).year
    body = render_generated_region(sections, current_year)
    with open(PAGE_PATH, "w", encoding="utf-8") as f:
        f.write(f"{head}{START_MARKER}\n{body}\n{END_MARKER}{tail}")
    print(f"Added {added} article(s) to {PAGE_PATH}")


if __name__ == "__main__":
    main()
