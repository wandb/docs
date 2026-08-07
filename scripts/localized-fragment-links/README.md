# Localized fragment links (ja/ko anchor slugs)

Mintlify generates heading anchors from the **heading text**: spaces become hyphens, ASCII is lowercased, other characters (e.g. CJK) are kept, duplicates get `-2`, `-3`. It does not support custom heading IDs.

So in translated content, links that use English fragments (e.g. `#how-to-get-started`) point at slugs that do not exist on the translated page, where the same section has a translated heading and thus a different slug (e.g. `#開始方法`).

## Who runs this and when

The main user is the **translator** (or whoever owns the translation refresh). Run the rewriter **toward the end of a translation refresh**, after the localized MDX (ja/, ko/) is in place. That way heading text and structure are final, and the script can reliably map EN fragments to localized slugs by position. Run from the docs repo root and use `--dry-run --verbose` first to review changes before applying.

## Slug algorithm

Implemented in `mintlify_slug.py`:

1. Collapse and strip whitespace; replace spaces with hyphens.
2. Lowercase ASCII letters only; keep letters (any script), digits, hyphens.
3. Strip any other characters (punctuation, etc.).
4. Collapse repeated hyphens and strip leading/trailing hyphens.
5. For duplicate slugs on the same page, Mintlify appends `-2`, `-3`, ...

This matches observed behavior on local and live docs (ja/ko pages).

## Fixing links by position

1. **Slug from heading text** – We compute the slug for any heading (see above).
2. **Match by position** – The Nth heading on the EN page is assumed to correspond to the Nth heading on the localized page. A link to `#en-slug` is rewritten to the localized slug at that position.
3. **Caveat** – If a translated page has different heading order or count than the EN page, the mapping can be wrong. Always run with `--dry-run --verbose` first and spot-check pages where EN and localized structure may differ.

## Module layout

| Module | Purpose |
|--------|--------|
| `mintlify_slug.py` | Slug generation and duplicate handling; run as script to verify against observed ids. |
| `headings.py` | Extract `##` / `###` and `<h2>` / `<h3>` from MDX in document order. |
| `path_utils.py` | Resolve EN counterpart path, page keys, and link targets for ja/ko. |
| `extract_heading_slugs.py` | CLI: list headings/slugs for one file, or compare EN vs JA by position. |
| `rewrite_fragment_links.py` | CLI: build slug maps and rewrite fragment links in localized MDX. |

## Running

Run from the **docs repo root** (parent of `ja/` and `ko/`).

```bash
# Verify slug function against observed ids
python3 scripts/localized-fragment-links/mintlify_slug.py

# Extract headings and show computed slugs for an MDX file
python3 scripts/localized-fragment-links/extract_heading_slugs.py ja/launch/launch-terminology.mdx

# Compare EN vs JA headings/slugs by position (for link rewriting)
python3 scripts/localized-fragment-links/extract_heading_slugs.py --en platform/launch/launch-terminology.mdx --ja ja/launch/launch-terminology.mdx

# Rewrite fragment links (dry run; no files changed)
python3 scripts/localized-fragment-links/rewrite_fragment_links.py --dry-run --verbose

# Rewrite fragment links (apply changes)
python3 scripts/localized-fragment-links/rewrite_fragment_links.py

# Only Japanese
python3 scripts/localized-fragment-links/rewrite_fragment_links.py --locale ja --dry-run
```
