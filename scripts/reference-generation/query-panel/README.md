# Query Expression Language reference generation

Generates English Mintlify pages under `models/ref/query-panel/` from the TypeScript implementation in [`wandb/weave-internal`](https://github.com/wandb/weave-internal) (`weave-js/`).

This pipeline is **not** related to W&B Weave. It targets the Models **query panel** expression language (legacy internal name: Weave).

## Output schema

- One MDX file per datatype, for example `models/ref/query-panel/run.mdx`.
- Front matter: `title` is the **datatype display name** when listed on the QEL overview (for example `artifactType` for the `artifact-type` URL slug), so the nav title matches the spelling used in the expression language and in the page body. If the overview has no label yet, the converter falls back to the upstream `#` heading text, then the slug.
- Body sections: `## Chainable Ops` and optional `## List Ops` (upstream `generateDocs.ts`).
- Operation headings use `### <a id="..."></a>\`op-name\`` so anchors stay stable. List Ops reuse the same op names as Chainable Ops, so List section anchors get a `-list` suffix to avoid duplicate HTML ids.
- Internal links point to `/models/ref/query-panel/<slug>` with **no trailing slash** (Mintlify canonical paths; avoids extra redirects). Legacy `https://docs.wandb.ai/ref/weave/...` URLs from upstream `docType()` are rewritten and any trailing slash on these internal paths is stripped.
- Argument tables use `| Argument | Description |` headers.
- Table cells that say `A [artifactType](...` are normalized to `An [...]` when the link text starts with a vowel letter (simple English rule; upstream strings are otherwise unchanged).
- The overview page `models/ref/query-panel.mdx` wraps the generated datatype bullet list in MDX comments `{/* query-panel-generated-data-types:start */}` / `{/* ... end */}` so Mintlify can parse the file (HTML `<!-- -->` comments are not valid in MDX here).

## Prerequisites

- `git`, `yarn` (classic v1), `python3.11+`, Node **20** (matches CI; avoids native `canvas` install on dev machines).
- Read access to `wandb/weave-internal` (SSH, HTTPS, or a GitHub App token).

## Local usage

From the repository root, with a clone of `weave-internal` already present:

```bash
export WEAVE_JS_ROOT=/path/to/weave-internal/weave-js
./scripts/reference-generation/query-panel/generate_query_panel_reference.sh
```

Or let the script clone using a read token:

```bash
export QUERY_PANELS_READ_TOKEN="..."  # contents:read on weave-internal
./scripts/reference-generation/query-panel/generate_query_panel_reference.sh
```

## What the shell script does

1. Optionally shallow-clones `wandb/weave-internal` when `WEAVE_JS_ROOT` is unset, using `QUERY_PANELS_READ_TOKEN` in the clone URL when set.
2. Copies `src/core/docs/README.md` to `docs/README.md` inside `weave-js` (upstream `generateDocs.ts` expects that path).
3. Runs `yarn install --ignore-scripts` in `weave-js` to avoid optional native builds (for example `canvas`) that are not required for doc generation.
4. Runs `npx vite-node@3.1.3 src/core/generateDocs.ts` to produce Markdown under `weave-js/docs_gen/`.
5. Runs `convert_query_panel_md.py` to write Mintlify MDX, refresh the data-type list in `models/ref/query-panel.mdx`, and align the **English** `docs.json` navigation entry for Query Expression Language (localized nav is not modified).

## CI

See `.github/workflows/generate-query-panel-reference.yml`. CI creates a GitHub App installation token and passes it as `QUERY_PANELS_READ_TOKEN`.

## Tests

```bash
python3 -m unittest discover -s scripts/reference-generation/query-panel/tests -p 'test_*.py' -v
```
