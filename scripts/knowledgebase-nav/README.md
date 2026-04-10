# Knowledgebase Nav Generator

A standalone script that regenerates knowledgebase nav pages and updates the `docs.json` navigation for the W&B documentation repository.

The generator reads MDX article files from `support/<product>/articles/`, aggregates them by keyword tags, and:

- **Updates tab-page Badges on articles.** Only `<Badge>` components whose link goes to `/support/<product>/tags/<tag-slug>` are rewritten from `keywords` (order preserved). Managed Badges are wrapped in MDX comment markers (`{/* AUTO-GENERATED: tab badges */}` and `{/* END AUTO-GENERATED: tab badges */}`) so the generator can locate them without regex matching on subsequent runs. Other Badges, prose, and anything outside the markers stay as you wrote them. If a new article has no tab Badges yet, the generator will insert them for you when `keywords` is non-empty.
- **Produces tag pages** at `support/<product>/tags/<tag-slug>.mdx`. Each lists the articles tagged with that keyword as Mintlify Card components.
- **Product index pages** at `support/<product>.mdx`. Each shows a "Featured articles" section (if any) and a "Browse by category" listing of all tags with article counts.
- **Updated `docs.json` navigation.** Hidden support tabs are updated to reflect the current set of tag pages.
- **Updated root `support.mdx`.** The generator replaces the article and tag count lines inside each product `<Card>` (matched by `href="/support/<slug>"`) so the landing page stays in sync with the crawl. Count lines are wrapped in `{/* auto-generated counts */}` and `{/* end auto-generated counts */}` markers so writers can add other content in the Card body. The featured-articles section is also managed between its own markers.

The generator runs automatically through GitHub Actions (workflow file `.github/workflows/knowledgebase-nav.yml`) when a pull request is opened, updated with new commits, or reopened, and at least one changed file matches `support/**` or `scripts/knowledgebase-nav/**`. You can also run that workflow manually from the Actions tab for previews.

---

## For tech writers

This section walks through the common tasks you will perform as an article author. For normal changes, you can rely on the CI workflow: it regenerates navigation when you open a pull request, so you do not need Python on your machine.

If you want to run the generator locally (for example to preview footers and tag pages before you push), see [Running the generator locally](#running-the-generator-locally).

### Adding a new article to an existing tag

1. Create a new MDX file in the appropriate product's articles directory:

   ```
   support/<product>/articles/your-article-slug.mdx
   ```

   For example: `support/models/articles/how-do-i-reset-my-api-key.mdx`

2. Add YAML front matter at the top of the file with the article's title and keywords (the tags it belongs to):

   ```yaml
   ---
   title: "How do I reset my API key?"
   keywords: ["Security", "Administrator"]
   ---
   ```

   The `keywords` list controls which tag pages the article appears on. Each keyword must exactly match an entry in `config.yaml` (case-sensitive). For example, use `"Environment Variables"`, not `"environment variables"` or `"Env Vars"`.

3. Write the article body content after the front matter. You can stop after the last paragraph. When the workflow runs, it updates only the tab-page `<Badge>` links (targets under `/support/<product>/tags/`) to match `keywords`, wrapped in MDX comment markers. You may add a `---` line or other text yourself; anything outside the markers is left alone. The first time tab Badges are needed, the generator appends a blank line, markers, and the Badges at the end of the body (no `---` is added automatically).

4. Open a pull request. The workflow checks out your branch, runs the generator, and commits any updates to article footers, tag pages, product index pages, `docs.json`, and `support.mdx` when those files change. You do not need to edit generated files by hand. **Pull requests from forks** still run the generator (so logs show problems), but GitHub cannot push commits back to your fork. Run the generator locally and push the regenerated files, or ask a maintainer to regenerate after merge.

   If you remove every keyword from front matter (`keywords: []` or omit the field), the generator removes tab-page Badges only. Other Badges are unchanged.

### Adding a new keyword (tag) that does not exist yet

If you want to use a keyword that is not yet recognized for a product, you need to add it to the configuration file:

1. Open `scripts/knowledgebase-nav/config.yaml`.

2. Find the product entry under `products:` and add your new keyword to its `allowed_keywords` list, in alphabetical order.

   **Before:**
   ```yaml
   - slug: models
     display_name: "W&B Models"
     allowed_keywords:
       - Academic
       - Administrator
       - Alerts
   ```

   **After (adding "API Keys"):**
   ```yaml
   - slug: models
     display_name: "W&B Models"
     allowed_keywords:
       - Academic
       - Administrator
       - Alerts
       - API Keys
   ```

3. Use the keyword in your article's `keywords` front matter. On the next PR, the generator creates a new tag page at `support/<product>/tags/api-keys.mdx` and adds it to the `docs.json` navigation.

**What if I forget to update config.yaml?** The generator still creates the tag page, but prints a warning in the CI logs. This is intentional. We never silently drop content. Resolve the warning by adding the keyword to config.yaml.

### Making an article featured (or removing featured status)

Featured articles appear in the "Featured articles" section at the top of the product index page (for example, `support/models.mdx`), rendered as large horizontal Cards with a body preview and tag Badges.

1. To feature an article, add `featured: true` to its front matter:

   ```yaml
   ---
   title: "How do I reset my API key?"
   keywords: ["Security", "Administrator"]
   featured: true
   ---
   ```

2. To remove featured status, either set `featured: false` or remove the `featured` line entirely. Articles without the field are not featured.

3. There is no hard limit on how many articles can be featured per product, but we recommend keeping it to 3-5 for a clean layout.

### Customizing Card preview text

By default, the generator creates Card preview text by stripping Markdown and MDX from the article body and truncating to 120 characters. You can override this with front matter fields so the Card shows exactly the text you want.

The generator resolves preview text using a three-level hierarchy:

1. **`docengineDescription`** (highest priority). Use this when you want to control the Card preview independently of SEO. This field is not used by Mintlify for anything else.
2. **`description`**. If `docengineDescription` is not set, the generator uses this field. Note that Mintlify also renders `description` as the page's `<meta name="description">` tag for search engines. Setting it affects both the Card preview and the SEO metadata. Use `docengineDescription` instead when you want the Card text to differ from the SEO description.
3. **Auto-generated body snippet**. If neither field is set, the generator falls back to the existing behavior: convert the article body to plain text and truncate to 120 characters.

The Card preview appears in three places:

- Tag page Cards (for example, `support/models/tags/experiments.mdx`).
- Featured article Cards on the product index page (for example, `support/models.mdx`).
- Featured article Cards on the root support landing page (`support.mdx`).

**Processing rules.** When using `docengineDescription` or `description`, the generator applies only minimal processing:

- Outer wrapping quotes (`"` or `'`) are stripped. YAML sometimes preserves them depending on how you quote the value.
- Internal newlines are collapsed to a single space. YAML block scalars (`|`, `>`) can produce multiline strings, but Card bodies must be single-line. If you need precise control over the text, use a single-line quoted string in front matter.
- No other processing is applied. The value is not passed through Markdown stripping, HTML entity decoding, or truncation.

**MDX safety.** Override text is emitted directly inside `<Card>` components without sanitization. Avoid characters or strings that break MDX parsing, such as unmatched `<`, raw `</Card>`, or unescaped `{`.

**Example:**

```yaml
---
title: "How do I reset my API key?"
keywords: ["Security", "Administrator"]
docengineDescription: "Step-by-step instructions for resetting your W&B API key from the user settings page."
description: "Reset your W&B API key."
---
```

In this example, the Card preview shows the `docengineDescription` value. The `description` value is used only by Mintlify for SEO. If `docengineDescription` were removed, the Card preview would show the `description` value instead.

### Front matter quick reference

| Field                  | Required | Default         | Description                                                                                                                                                                                                                                                                                                                                                                                             |
|------------------------|----------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `title`                | Expected | `""` if omitted | Article title. Used in Cards and listings. The generator does not fail if it is missing.                                                                                                                                                                                                                                                                                                                |
| `keywords`             | Expected | `[]` if omitted | YAML list of tag names. Each should match an entry in `config.yaml` (case-sensitive). Controls which tag pages the article appears on. If the list is empty or missing, the article is not listed under any tag. If you accidentally use a single string (for example `keywords: "Security"`), the generator treats it as one tag and emits a warning. Other non-list types are ignored with a warning. |
| `featured`             | No       | `false`         | Set to `true` to feature the article on the product index page.                                                                                                                                                                                                                                                                                                                                         |
| `docengineDescription` | No       | `""` if omitted | Card preview text for tag pages, featured sections, and the support landing page. Takes priority over `description` and the auto-generated body snippet. Use this when you want to set Card text independently of the SEO description. Outer wrapping quotes are stripped and newlines are collapsed to a single space.                                                                                  |
| `description`          | No       | `""` if omitted | Card preview text if `docengineDescription` is not set. Mintlify also uses this field for the page's `<meta name="description">` SEO tag, so setting it affects both the Card preview and search engine metadata. Use `docengineDescription` to decouple the two. Same processing rules: outer quotes stripped, newlines collapsed.                                                                     |

### Running the generator locally

**You do not need to run the generator locally to open a pull request.** The CI workflow runs it automatically when you open a PR.

If you want to run the generator locally (for example to preview footers and tag pages before you push), follow the instructions in this section.

These steps are written for **macOS**. They use a **virtual environment** so packages install only for this project and do not affect your system Python.

**Prerequisites:** [Python](https://www.python.org/downloads/) 3.11 or another current Python 3 release (3.11 is what CI uses).

1. Open Terminal and go to the root of the `wandb-docs` clone (the directory that contains `docs.json` and `support/`).

2. Create a virtual environment in a folder named `.venv` (you can pick another name, but keep that folder out of git commits), then activate it:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   After activation, your prompt usually starts with `(.venv)`.

3. Upgrade `pip` (recommended once per new venv), then install the generator dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r scripts/knowledgebase-nav/requirements.txt
   ```

4. Run the script from the repo root:

   ```bash
   python scripts/knowledgebase-nav/generate_tags.py --repo-root .
   ```

   The command updates article footers, tag pages, product indexes, `docs.json`, and `support.mdx` in your working tree. Review the changes with `git diff` before you commit.

5. When you are done, leave the virtual environment:

   ```bash
   deactivate
   ```

Next time you want to run the generator, activate the same `.venv` again (repeat only the activation command from step 2), then run step 4.

---

## For developers

### File layout

```
scripts/knowledgebase-nav/
  generate_tags.py          Main generator script (single file, all phases below)
  config.yaml               Products and allowed keywords configuration
  requirements.txt          Python dependencies (pyyaml, jinja2)
  README.md                 This file
  Architecture.md           Architecture overview and Mermaid diagrams for developers
  templates/
    support_tag.mdx.j2      Jinja2 template for tag pages
    support_product_index.mdx.j2  Jinja2 template for product index pages
  tests/
    __init__.py             Package marker for tests
    conftest.py             Registers the `integration` pytest marker
    test_generate_tags.py   Unit tests (pytest, mocked filesystem and docs.json)
    test_golden_output.py   Golden-file integration tests (real repo layout)
```

### How the generator works

The script runs one pipeline after loading `config.yaml` and Jinja2 templates. The template environment registers a `tojson_unicode` filter for YAML front matter in the MDX templates (Jinja's default `tojson` uses HTML-oriented escapes that this project avoids).

1. **Crawl and parse** (`crawl_articles`, `parse_frontmatter`, `build_tag_index`, `get_featured_articles`): For each product, reads every `.mdx` file in `support/<product>/articles/`, parses YAML front matter (`title`, `keywords`, `featured`), and extracts the article body (everything before the `_BADGE_START` marker). The `keywords` field is normalized with `_normalize_keywords` (YAML list of strings; a single string is coerced to a one-item list with a warning; other shapes warn and become an empty list). Body text is turned into a Card preview with `plain_text` and `extract_body_preview`. The `plain_text` step removes fenced code, horizontal rules, links and image syntax (keeping link labels), autolinks, bare `http(s)` URLs, HTML and MDX or JSX tags and simple `{...}` expressions, emphasis markers, common list or heading prefixes, decodes HTML entities, replaces non-breaking spaces (U+00A0) with a normal space, maps typographic quotes and apostrophes to ASCII, then applies an allowlist of safe characters (including `_` and `=` for identifiers) and collapses whitespace. `extract_body_preview` truncates to 120 characters and appends ` ...` when longer. Unknown `keywords` values warn once per keyword but still get tag pages.

2. **Generate tag pages** (`render_tag_pages`, `cleanup_stale_tag_pages`): For each tag that appears in at least one article, renders `support/<product>/tags/<tag-slug>.mdx` from `support_tag.mdx.j2`. Tags present only in `config.yaml` and not used by any article do not get a file. After writing current pages, `cleanup_stale_tag_pages` deletes any `.mdx` files in the tags directory that no longer correspond to a keyword used by any article, keeping the tags directory and `docs.json` free of stale entries.

3. **Generate product index pages** (`render_product_index`): Renders `support/<product>.mdx` with optional "Featured articles" and a "Browse by category" section from `support_product_index.mdx.j2`.

4. **Sync tab Badges** (`sync_all_support_article_footers`, `sync_support_article_footer`, `build_tab_badges_mdx`, `build_keyword_footer_mdx`): For each `support/<product>/articles/*.mdx` file, replaces managed `<Badge>` links with one Badge per `keywords` entry (in list order). Managed Badges are enclosed in MDX comment markers (`_BADGE_START` / `_BADGE_END`); the generator matches markers when present and falls back to regex matching for articles that predate markers. Other Badges and the rest of the body are not edited. If there are no such Badges yet and `keywords` is non-empty, appends a blank line, markers, and the tab Badges (no `---`). If `keywords` is empty, removes the marker block (or bare tab-page Badges). Runs after tag pages are generated so articles are not modified if earlier phases fail.

5. **Update docs.json** (`update_docs_json`): Reads `docs.json`, finds the English entry (`navigation.languages[]` where `language == "en"`), then finds or creates hidden tabs named `Support: <display_name>`. Each tab's `pages` list is `support/<slug>` followed by sorted tag page paths. Other language entries and non-support tabs are left unchanged.

6. **Update support landing page** (`update_support_index`, `update_support_featured`): Edits the repository root `support.mdx` in place. Count lines inside each product `<Card>` are matched by `{/* auto-generated counts */}` markers (falling back to regex for migration), and replaced with current article and tag counts (including singular or plural labels). The featured-articles section between its own markers is regenerated from articles with `featured: true`.

### Running locally

On macOS, set up a virtual environment as in [Running the generator locally](#running-the-generator-locally) (same steps as in the tech writers section), then from the repo root:

```bash
python scripts/knowledgebase-nav/generate_tags.py --repo-root .
```

Use `--config /path/to/config.yaml` to point at a different config file. If you omit `--config`, the script uses `scripts/knowledgebase-nav/config.yaml` next to `generate_tags.py`.

The script prints a progress summary to stdout. The `warnings` module emits unknown keywords, skipped articles, and missing `support.mdx` product cards to stderr.

### Running tests

With the same virtual environment activated, install `pytest` and run:

```bash
python -m pip install pytest
pytest scripts/knowledgebase-nav/tests/test_generate_tags.py -v
pytest scripts/knowledgebase-nav/tests/test_golden_output.py -v -m integration
```

The unit tests use mocked file systems and run fast.

The golden-file module (`test_golden_output.py`) is marked `@pytest.mark.integration`. The marker is registered in `tests/conftest.py` so pytest does not warn about an unknown mark. It copies `support/`, `docs.json`, and `support.mdx` from the real repo into a temporary directory, runs the full pipeline there, and asserts:

- Every tag page under `support/<product>/tags/*.mdx` matches the real repo byte-for-byte (including that generated and existing tag file sets match).
- Each product index `support/<product>.mdx` matches the real repo byte-for-byte.
- Every article under `support/<product>/articles/*.mdx` matches the real repo byte-for-byte (catches regressions in footer sync and marker formatting).
- Support tabs in `docs.json` (names, `pages` order, `hidden: true`) match the real file, and tabs that do not start with `Support:` are unchanged.
- Root `support.mdx` matches the real file byte-for-byte.

If `docs.json` is not found at the repository root the tests resolve to (the parent directory of `scripts/`, when this package lives at `scripts/knowledgebase-nav/`), the golden tests skip.

### Adding a new product

1. Add a new entry to `config.yaml` under `products:` with `slug`, `display_name`, and `allowed_keywords`.
2. Create the directory `support/<slug>/articles/` in the repo.
3. Add at least one article MDX file with appropriate front matter.
4. Add a product `<Card>` to the root `support.mdx` with `href="/support/<slug>"` and a marker-wrapped count line:

   ```
   <Card title="W&B NewProduct" href="/support/<slug>" arrow="true" icon="/icons/cropped-newproduct.svg">
     {/* auto-generated counts */}
     0 articles &middot; 0 tags
     {/* end auto-generated counts */}
   </Card>
   ```

   Without the markers (or the bare count-line pattern for migration), the generator warns and leaves that card unchanged.

5. Run the generator (or open a PR). It creates the `tags/` directory as needed, generates tag pages and the product index page, adds or updates the hidden support tab in `docs.json`, and refreshes counts on `support.mdx`.

### How docs.json merging works

The W&B docs site uses a multi-language navigation structure:

```json
{
  "navigation": {
    "languages": [
      { "language": "en", "tabs": [ ... ] },
      { "language": "ja", "tabs": [ ... ] }
    ]
  }
}
```

The generator only modifies the English (`"en"`) language entry. Within that entry, it finds or creates hidden tabs named `"Support: <display_name>"` (for example `"Support: W&B Models"`). Each tab's `pages` list is set to:

```
["support/<slug>", "support/<slug>/tags/tag-a", "support/<slug>/tags/tag-b", ...]
```

All other tabs, groups, and non-support navigation entries are preserved untouched.

### How the CI workflow triggers

Workflow name: **Knowledgebase Nav** (file `.github/workflows/knowledgebase-nav.yml`). The job does not run `pytest`; run tests locally as above.

- **Pull requests**: Runs when a PR is opened, synchronized (new pushes to the branch), or reopened, and the path filter matches. Only runs that include changes under `support/**` or `scripts/knowledgebase-nav/**` trigger the workflow (see GitHub path filters for `pull_request`). After generation, `stefanzweifel/git-auto-commit-action` commits only if something changed, using `file_pattern`: `support.mdx`, `support/*/articles/*.mdx`, `support/*/tags/*.mdx`, `support/*.mdx`, and `docs.json`.

- **workflow_dispatch**: Run from the Actions tab. Optional input `branch`: when set, checkout uses that branch name. When empty, checkout uses `github.ref` (the branch selected in the "Use workflow from" dropdown). On `pull_request` events, checkout uses the PR head repository and head commit SHA (including forks); auto-commit runs only for same-repo PRs, not forks.

### Troubleshooting

**"Unknown keyword" warnings in CI logs**: An article uses a keyword not listed in `config.yaml`. The tag page is still generated, but add the keyword to config.yaml to suppress the warning.

**Allowed keyword in config but no tag page file**: A keyword is listed in `config.yaml` but no article lists it in `keywords`. The generator only writes tag pages for tags that appear on at least one article, so no file is created for unused keywords. If a tag page previously existed but all articles stopped using that keyword, the generator deletes the stale file and removes it from `docs.json`.

**Tag page not appearing in docs.json**: Verify the keyword is used in at least one article's `keywords` front matter and that the generator ran successfully.

**Golden test failures after intentional changes**: If you intentionally changed templates, slug logic, body preview rules, keyword footer formatting (including marker comments), product index output, `docs.json` support tabs, or `support.mdx` count lines or featured-article markers, the golden test will fail. Run the generator from the repo root, commit the updated generated files, then re-run the golden tests.

**Warning about a missing product card in `support.mdx`**: A product in `config.yaml` has no matching `<Card href="/support/<slug>">` or the count line does not match the expected pattern. Add or fix the card so counts can update.

**Article skipped in the log**: Front matter is missing a closing `---` or the file does not start with `---`. The generator skips that file with a warning.
