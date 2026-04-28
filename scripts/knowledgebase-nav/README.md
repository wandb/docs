# Knowledgebase Nav Generator

A standalone script that regenerates knowledgebase nav pages for a Mintlify documentation repository.

This utility lives at `<utility>/knowledgebase-nav/` inside the docs repo (for example `scripts/knowledgebase-nav/` or `utils/knowledgebase-nav/`); throughout this README, paths shown as `<utility>/knowledgebase-nav/...` mean the same directory the README itself lives in. The script locates its own `config.yaml` and resolves the **Mintlify root** (the directory containing `support/` and `support.mdx`) from the `mintlify_root` key in that file, so it does not need a path passed on the command line.

The generator reads MDX article files from `support/<product>/articles/`, aggregates them by keyword tags, and:

- **Updates tab-page Badges on articles.** Only `<Badge>` components whose link goes to `/support/<product>/tags/<tag-slug>` are rewritten from `keywords` (order preserved). Managed Badges are wrapped in MDX comment markers — any `{/* ... */}` comment that contains `AUTO-GENERATED: tab badges` anywhere inside it is the start marker; any comment that contains `END AUTO-GENERATED: tab badges` anywhere inside it is the end marker. You can add notes anywhere inside these comments without breaking the generator. Other Badges, prose, and anything outside the markers stay as you wrote them. If a new article has no tab Badges yet, the generator will insert them for you when `keywords` is non-empty.
- **Produces tag pages** at `support/<product>/tags/<tag-slug>.mdx`. Each lists the articles tagged with that keyword as Mintlify Card components.
- **Product index pages** at `support/<product>.mdx`. Each shows a "Featured articles" section (if any) and a "Browse by category" listing of all tags with article counts.
- **Updated root `support.mdx`.** The generator replaces the article and tag count lines inside each product `<Card>` (matched by `href="/support/<slug>"`) so the landing page stays in sync with the crawl. Count lines are wrapped between `{/* AUTO-GENERATED: counts */}` and `{/* END AUTO-GENERATED: counts */}` markers. The featured-articles section is managed between `{/* AUTO-GENERATED: featured articles */}` and `{/* END AUTO-GENERATED: featured articles */}` markers. For all markers: matching is case-insensitive, the colon after "generated" is optional, and the keyword can appear anywhere inside the comment — so you can add notes without breaking the generator.

The generator never reads, parses, or writes `docs.json`. When tag pages are added or removed, a human must update the matching `Support: <display_name>` tab under `navigation.languages[language="en"].tabs[]` by hand. The PR comment posted by `pr_report.py` lists the exact page ids to add or remove, grouped by product, so the edit can be made with copy and paste.

The generator runs automatically through GitHub Actions (workflow file `.github/workflows/knowledgebase-nav.yml`) when a pull request is opened, updated with new commits, or reopened, and at least one changed file matches the Mintlify `support/**` directory or the utility directory. You can also run that workflow manually from the Actions tab for previews.

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

   The generator recognises any `{/* ... */}` comment that contains `AUTO-GENERATED: tab badges` anywhere inside it as the start marker, and any comment that contains `END AUTO-GENERATED: tab badges` anywhere inside it as the end marker. You can add notes anywhere inside these comments — before the keyword, after it, or both — without breaking the generator. The canonical marker text is always written on output regardless of what was in the original comment.

4. Open a pull request. The workflow checks out your branch, runs the generator, and commits any updates to article footers, tag pages, product index pages, and `support.mdx` when those files change. The generator does not edit `docs.json`. If your change adds or removes any tag pages, the workflow's PR comment lists the exact page ids that need to be added to or removed from the matching `Support: <display_name>` tab in `docs.json`; a human (you or a reviewer) must make that edit by hand and push it on the PR branch. You do not need to edit any other generated files by hand. **Pull requests from forks** still run the generator (so logs show problems), but GitHub cannot push commits back to your fork. Run the generator locally and push the regenerated files, or ask a maintainer to regenerate after merge.

   If you remove every keyword from front matter (`keywords: []` or omit the field), the generator removes tab-page Badges only. Other Badges are unchanged.

### Adding a new keyword (tag) that does not exist yet

If you want to use a keyword that is not yet recognized for a product, you need to add it to the configuration file:

1. Open `<utility>/knowledgebase-nav/config.yaml` (the `config.yaml` file next to this README).

2. Find the product entry under `products:` and add your new keyword to its `allowed_keywords` list, in alphabetical order.

   **Before:**
   ```yaml
   - slug: models
     display_name: "Models"
     allowed_keywords:
       - Academic
       - Administrator
       - Alerts
   ```

   **After (adding "API Keys"):**
   ```yaml
   - slug: models
     display_name: "Models"
     allowed_keywords:
       - Academic
       - Administrator
       - Alerts
       - API Keys
   ```

3. Use the keyword in your article's `keywords` front matter. On the next PR, the generator creates a new tag page at `support/<product>/tags/api-keys.mdx`. The PR comment posted by the workflow lists the new page id (for example `support/<product>/tags/api-keys`) under "docs.json update required". Add that line to the matching `Support: <display_name>` tab's `pages` array in `docs.json` and push it on the PR branch — the generator will not do this for you.

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
docengineDescription: "Step-by-step instructions for resetting your API key from the user settings page."
description: "Reset your API key."
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

1. Open Terminal and go to the root of the docs repo clone.

2. Create a virtual environment in a folder named `.venv` (you can pick another name, but keep that folder out of git commits), then activate it:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   After activation, your prompt usually starts with `(.venv)`.

3. Upgrade `pip` (recommended once per new venv), then install the generator dependencies (replace `<utility>/knowledgebase-nav` below with the actual directory; it is the directory that contains this README):

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r <utility>/knowledgebase-nav/requirements.txt
   ```

4. Run the script (the path it scans is read from `mintlify_root` in `config.yaml`):

   ```bash
   python <utility>/knowledgebase-nav/generate_tags.py
   ```

   The command updates article footers, tag pages, product indexes, and `support.mdx` in your working tree. It does not modify `docs.json`. Review the changes with `git diff` before you commit.

5. When you are done, leave the virtual environment:

   ```bash
   deactivate
   ```

Next time you want to run the generator, activate the same `.venv` again (repeat only the activation command from step 2), then run step 4.

---

## For developers

### File layout

```
<utility>/knowledgebase-nav/
  generate_tags.py          Main generator script (single file, all phases below)
  pr_report.py              Builds the GitHub PR comment / job summary Markdown
  config.yaml               mintlify_root, badge_color, products, and allowed keywords
  requirements.txt          Python dependencies (pyyaml, jinja2)
  README.md                 This file
  Architecture.md           Architecture overview and Mermaid diagrams for developers
  templates/
    support_tag.mdx.j2      Jinja2 template for tag pages
    support_product_index.mdx.j2  Jinja2 template for product index pages
  tests/
    __init__.py             Package marker for tests
    conftest.py             Registers the `integration` pytest marker
    test_generate_tags.py   Unit tests for generate_tags.py (mocked filesystem)
    test_pr_report.py       Unit tests for pr_report.py (synthetic git diff input)
    test_golden_output.py   Golden-file integration tests (real repo layout)
```

### How the generator works

The script runs one pipeline after loading `config.yaml` and Jinja2 templates. The template environment registers a `tojson_unicode` filter for YAML front matter in the MDX templates (Jinja's default `tojson` uses HTML-oriented escapes that this project avoids).

1. **Crawl and parse** (`crawl_articles`, `parse_frontmatter`, `build_tag_index`, `get_featured_articles`): For each product, reads every `.mdx` file in `support/<product>/articles/`, parses YAML front matter (`title`, `keywords`, `featured`), and extracts the article body (everything before the first `_BADGE_START_RE` match). The `keywords` field is normalized with `_normalize_keywords` (YAML list of strings; a single string is coerced to a one-item list with a warning; other shapes warn and become an empty list). Body text is turned into a Card preview with `plain_text` and `extract_body_preview`. The `plain_text` step removes fenced code, horizontal rules, links and image syntax (keeping link labels), autolinks, bare `http(s)` URLs, HTML and MDX or JSX tags and simple `{...}` expressions, emphasis markers, common list or heading prefixes, decodes HTML entities, replaces non-breaking spaces (U+00A0) with a normal space, maps typographic quotes and apostrophes to ASCII, then applies an allowlist of safe characters (including `_` and `=` for identifiers) and collapses whitespace. `extract_body_preview` truncates to 120 characters and appends ` ...` when longer. Unknown `keywords` values warn once per keyword but still get tag pages.

2. **Generate tag pages** (`render_tag_pages`, `cleanup_stale_tag_pages`): For each tag that appears in at least one article, renders `support/<product>/tags/<tag-slug>.mdx` from `support_tag.mdx.j2`. Tags present only in `config.yaml` and not used by any article do not get a file. After writing current pages, `cleanup_stale_tag_pages` deletes any `.mdx` files in the tags directory that no longer correspond to a keyword used by any article, keeping the tags directory free of stale entries.

3. **Generate product index pages** (`render_product_index`): Renders `support/<product>.mdx` with optional "Featured articles" and a "Browse by category" section from `support_product_index.mdx.j2`.

4. **Sync tab Badges** (`sync_all_support_article_footers`, `sync_support_article_footer`, `build_tab_badges_mdx`, `build_keyword_footer_mdx`): For each `support/<product>/articles/*.mdx` file, replaces managed `<Badge>` links with one Badge per `keywords` entry (in list order). Managed Badges are located via `_BADGE_START_RE` and `_BADGE_END_RE`: any `{/* ... */}` comment containing `AUTO-GENERATED: tab badges` anywhere inside it is the start; any comment containing `END AUTO-GENERATED: tab badges` anywhere inside it is the end — authors can add notes anywhere in these comments without breaking the generator. The generator falls back to regex matching for articles that predate markers. Other Badges and the rest of the body are not edited. If there are no such Badges yet and `keywords` is non-empty, appends a blank line, canonical markers, and the tab Badges (no `---`). If `keywords` is empty, removes the marker block (or bare tab-page Badges). Runs after tag pages are generated so articles are not modified if earlier phases fail.

5. **Update support landing page** (`update_support_index`, `update_support_featured`): Edits the Mintlify root `support.mdx` in place. Count lines inside each product `<Card>` are located via `_COUNTS_START_RE` / `_COUNTS_END_RE` (any `{/* ... */}` comment containing `AUTO-GENERATED: counts` / `END AUTO-GENERATED: counts`, falling back to a bare count-line pattern for migration) and replaced with current article and tag counts (including singular or plural labels). The featured-articles section is regenerated between markers located via `_FEATURED_START_RE` / `_FEATURED_END_RE` (any comment containing `AUTO-GENERATED: featured articles` / `END AUTO-GENERATED: featured articles`). All marker matching is case-insensitive with an optional colon after "generated".

`docs.json` is intentionally not on this list. The generator does not read or write that file. After the generator runs, the workflow's PR comment (built by `pr_report.py`) lists the page ids of any tag pages that were added or removed, grouped by `Support: <display_name>`, so a human can update the matching tab in `docs.json` by hand.

### Running locally

On macOS, set up a virtual environment as in [Running the generator locally](#running-the-generator-locally) (same steps as in the tech writers section), then from the repo root:

```bash
python <utility>/knowledgebase-nav/generate_tags.py
```

Use `--config /path/to/config.yaml` to point at a different config file. If you omit `--config`, the script uses `config.yaml` next to `generate_tags.py`. The Mintlify root scanned by the generator is always read from `mintlify_root` in that config file.

The script prints a progress summary to stdout. The `warnings` module emits unknown keywords, skipped articles, and missing `support.mdx` product cards to stderr.

### Running tests

With the same virtual environment activated, install `pytest` and run:

```bash
python -m pip install pytest
pytest <utility>/knowledgebase-nav/tests/test_generate_tags.py -v
pytest <utility>/knowledgebase-nav/tests/test_pr_report.py -v
pytest <utility>/knowledgebase-nav/tests/test_golden_output.py -v -m integration
```

The unit tests use mocked file systems and run fast.

The golden-file module (`test_golden_output.py`) is marked `@pytest.mark.integration`. The marker is registered in `tests/conftest.py` so pytest does not warn about an unknown mark. It copies `support/` and `support.mdx` from the real repo into a temporary directory, runs the full pipeline there, and asserts:

- Every tag page under `support/<product>/tags/*.mdx` matches the real repo byte-for-byte (including that generated and existing tag file sets match).
- Each product index `support/<product>.mdx` matches the real repo byte-for-byte.
- Every article under `support/<product>/articles/*.mdx` matches the real repo byte-for-byte (catches regressions in footer sync and marker formatting).
- No `docs.json` file is created in the temp tree, so the pipeline never reads or writes it.
- Root `support.mdx` matches the real file byte-for-byte.

If `support.mdx` is not found at the Mintlify root resolved from `mintlify_root` in `config.yaml`, the golden tests skip.

### Adding a new product

1. Add a new entry to `config.yaml` under `products:` with `slug`, `display_name`, and `allowed_keywords`.
2. Create the directory `support/<slug>/articles/` in the repo.
3. Add at least one article MDX file with appropriate front matter.
4. Add a product `<Card>` to the root `support.mdx` with `href="/support/<slug>"` and a marker-wrapped count line:

   ```
   <Card title="NewProduct" href="/support/<slug>" arrow="true" icon="/icons/cropped-newproduct.svg">
     {/* AUTO-GENERATED: counts */}
     0 articles &middot; 0 tags
     {/* END AUTO-GENERATED: counts */}
   </Card>
   ```

   Without the markers (or the bare count-line pattern for migration), the generator warns and leaves that card unchanged.

5. Run the generator (or open a PR). It creates the `tags/` directory as needed, generates tag pages and the product index page, and refreshes counts on `support.mdx`. The generator does not edit `docs.json`. Add a new hidden tab to `docs.json` by hand at the same time you add the product:

   ```json
   {
     "tab": "Support: NewProduct",
     "hidden": true,
     "pages": [
       "support/<slug>"
     ]
   }
   ```

   Append the tag page ids to `pages` as the workflow's PR comment lists them.

### How docs.json edits are coordinated

The docs site uses a multi-language navigation structure:

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

Hidden tabs named `"Support: <display_name>"` (for example `"Support: Models"`) live under the English (`"en"`) language entry. Each tab's `pages` list looks like:

```
["support/<slug>", "support/<slug>/tags/tag-a", "support/<slug>/tags/tag-b", ...]
```

The generator never edits this file. When tag pages change, the workflow's PR comment (built by `pr_report.py`) shows a "docs.json update required" section grouped by `Support: <display_name>` listing the page ids to add or remove from each tab's `pages` array. A human (the PR author or a reviewer) makes those edits by hand and pushes them on the PR branch.

### How the CI workflow triggers

Workflow name: **Knowledgebase Nav** (file `.github/workflows/knowledgebase-nav.yml`). The job does not run `pytest`; run tests locally as above.

- **Pull requests**: Runs when a PR is opened, synchronized (new pushes to the branch), or reopened, and the path filter matches. Only runs that include changes under the Mintlify `support/**` directory or the utility directory trigger the workflow (see GitHub path filters for `pull_request`). After generation, `stefanzweifel/git-auto-commit-action` commits only if something changed, using `file_pattern`: `support.mdx`, `support/*/articles/*.mdx`, `support/*/tags/*.mdx`, and `support/*.mdx`. `docs.json` is intentionally not in this list — it is edited by humans only.

- **workflow_dispatch**: Run from the Actions tab. Optional input `branch`: when set, checkout uses that branch name. When empty, checkout uses `github.ref` (the branch selected in the "Use workflow from" dropdown). On `pull_request` events, checkout uses the PR head repository and head commit SHA (including forks); auto-commit runs only for same-repo PRs, not forks.

### Troubleshooting

**"Unknown keyword" warnings in CI logs**: An article uses a keyword not listed in `config.yaml`. The tag page is still generated, but add the keyword to config.yaml to suppress the warning.

**Allowed keyword in config but no tag page file**: A keyword is listed in `config.yaml` but no article lists it in `keywords`. The generator only writes tag pages for tags that appear on at least one article, so no file is created for unused keywords. If a tag page previously existed but all articles stopped using that keyword, the generator deletes the stale file. The PR comment will list the removed page id under "docs.json update required" so a human can also remove it from the matching `Support:` tab in `docs.json`.

**Tag page not appearing in docs.json**: The generator does not edit `docs.json`. After a new tag page is created, copy the page id (for example `support/models/tags/api-keys`) from the workflow's "docs.json update required" PR comment into the matching `Support: <display_name>` tab's `pages` array in `docs.json`, then commit and push the edit.

**Golden test failures after intentional changes**: If you intentionally changed templates, slug logic, body preview rules, keyword footer formatting (including marker comments), product index output, or `support.mdx` count lines or featured-article markers, the golden test will fail. Run the generator from the repo root, commit the updated generated files, then re-run the golden tests.

**Warning about a missing product card in `support.mdx`**: A product in `config.yaml` has no matching `<Card href="/support/<slug>">` or the count line does not match the expected pattern. Add or fix the card so counts can update.

**Article skipped in the log**: Front matter is missing a closing `---` or the file does not start with `---`. The generator skips that file with a warning.
