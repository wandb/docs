# GitHub Actions Workflows

## GitHub App authentication

Several workflows need permissions that the native `GITHUB_TOKEN` cannot provide. Use GitHub App installation tokens instead of personal access tokens for those cases.

Created these GitHub Apps under the `wandb` organization:

- `wandb-docs-source-reader`: install only on `wandb/docs-code-eval` and `wandb/weave-internal` with **Contents** read access.
- `wandb-docs-pr-writer`: install only on `wandb/docs` with **Contents** read and write access and **Pull requests** read and write access.

Stored the app credentials in `wandb/docs`:

- Repository variable `DOCS_SOURCE_READER_CLIENT_ID`: client ID for `wandb-docs-source-reader` (alphanumeric string shown on the app's settings page, for example `Iv1.abc123`; not the numeric App ID).
- Repository secret `DOCS_SOURCE_READER_PRIVATE_KEY`: private key for `wandb-docs-source-reader`.
- Repository variable `DOCS_PR_WRITER_CLIENT_ID`: client ID for `wandb-docs-pr-writer` (same format as above).
- Repository secret `DOCS_PR_WRITER_PRIVATE_KEY`: private key for `wandb-docs-pr-writer`.

The workflows use `actions/create-github-app-token@v3` to create short-lived installation tokens from these credentials.

Workflows that push back to a same-repo PR branch with this token (instead of the default workflow `GITHUB_TOKEN`) include **Compress Images** (`calibreapp-image-actions.yml`), **Build CSS** (`build-css.yml`), and **Knowledgebase Nav** (`knowledgebase-nav.yml`). That way downstream `pull_request` checks (for example **Doc quality**) still run on the automation commit.

## Action pinning

Every third-party `uses:` in workflows and composite actions pins a full commit SHA with a trailing version comment (`@3d3c42e5… # v7`), never a mutable tag or branch — a tag can be moved to different code after review; a SHA cannot. Do not add tag- or branch-pinned actions.

Renovate keeps the SHAs and version comments current, configured by `.github/renovate.json5` via the org-wide [`wandb/renovate-config`](https://github.com/wandb/renovate-config) preset. That preset also enforces update policy: a 7-day minimum release age (defense against tag-repointing attacks) and Dependency Dashboard approval for major bumps. Do not add a `github-actions` entry to `.github/dependabot.yml` for this because Dependabot would duplicate Renovate's PRs and propose updates before the 7-day gate, bypassing that protection.

## Sync Code Examples

**Workflow**: `sync-code-examples.yml`

Automatically syncs ground truth code examples from the [docs-code-eval](https://github.com/wandb/docs-code-eval) repository.

### Triggers

- **Manual**: Trigger on-demand from the [Actions tab](../../actions/workflows/sync-code-examples.yml)
- **Scheduled**: Runs weekly on Mondays at 9 AM UTC (optional)

### What It Does

1. Clones the latest `docs-code-eval` repository
2. Copies ground truth Python examples from `ground_truth/`
3. Copies task metadata CSV
4. Regenerates the SDK coding cheat sheet at `models/ref/sdk-coding-cheat-sheet.mdx`
5. Checks for changes
6. If changes detected, creates a **draft PR** with:
   - Clear change summary
   - Review checklist
   - Automatic labels

### Manual Triggering

1. Go to [Actions tab](../../actions)
2. Click "Sync Code Examples from docs-code-eval"
3. Click "Run workflow"
4. Select branch (usually `main`)
5. Click "Run workflow" button

### When to Use

- **After docs-code-eval updates**: New examples added or existing ones modified
- **Periodic sync**: Keep docs in sync with evaluation benchmark
- **Before releases**: Ensure latest examples are included

### Output

If changes are detected:
- Creates a draft PR named: `🔄 Sync code examples from docs-code-eval`
- **Base branch**: the repository default branch (for example `main`), even if you ran the workflow from another branch
- Branch name: `sync-code-examples-{run_number}`
- Status: Draft (must be marked ready for review)
- Labels: `documentation`, `automated`, `code-examples`

If no changes:
- Workflow completes successfully
- No PR created
- Message: "Code examples are already up to date"

### Review Process

When the draft PR is created:

1. **Review Changes**: Check the PR for accuracy
   - Verify code examples are correct
   - Check cheat sheet rendering
   - Ensure placeholders are appropriate
2. **Test Locally**: Build docs and verify appearance
3. **Mark Ready**: Convert from draft to ready for review
4. **Merge**: Standard review and merge process

### Configuration

The workflow uses:
- **Python**: 3.11
- **Permissions**: `contents: write`, `pull-requests: write`
- **Token action**: `actions/create-github-app-token@v3`
- **Action**: `peter-evans/create-pull-request@v8`

**Authentication and `docs-code-eval`**

The automatic `GITHUB_TOKEN` is scoped to this repository (`wandb/docs`) only. It does not grant read access to other private repositories in the org, so it cannot clone a private `wandb/docs-code-eval`.

To sync from a private `docs-code-eval`, install `wandb-docs-source-reader` on `wandb/docs-code-eval`. The workflow creates a GitHub App installation token and passes it to the sync script as `DOCS_CODE_EVAL_READ_TOKEN`.

**Clone fails with `could not read Username for 'https://github.com'`**

That usually means Git tried to prompt for credentials (no TTY in Actions) or a credential helper failed. The sync script clears `credential.helper` for the clone and uses HTTPS with `x-access-token` when `DOCS_CODE_EVAL_READ_TOKEN` is set. If the log shows an anonymous clone but the repo is private, check the GitHub App installation and the `DOCS_SOURCE_READER_APP_ID` and `DOCS_SOURCE_READER_PRIVATE_KEY` credentials.

### Troubleshooting

**Workflow fails to run:**
- Check repository permissions
- Verify `GITHUB_TOKEN` has required scopes

**PR not created despite changes:**
- Check workflow logs for errors
- Verify `peter-evans/create-pull-request` action succeeded

**Empty code blocks in cheat sheet:**
- Check that Python files in `snippets/code-examples/` are valid
- Verify docstring format in source files
- Re-run the workflow

### Related Files

- **Sync Script**: `scripts/sync_code_examples.sh`
- **Generator**: `scripts/generate_cheat_sheet.py`
- **Examples**: `snippets/code-examples/*.py`
- **Cheat Sheet**: `models/ref/sdk-coding-cheat-sheet.mdx`
- **Documentation**: `snippets/code-examples/README.md`

### Disabling Scheduled Runs

To disable the weekly automatic sync, remove or comment out these lines:

```yaml
schedule:
  - cron: '0 9 * * 1'
```

### Manual Sync Alternative

You can also run the sync script locally:

```bash
cd /path/to/docs
./scripts/sync_code_examples.sh
git add .
git commit -m "Sync code examples from docs-code-eval"
git push
```

## Doc quality

**Workflow**: `doc-quality.yml`

One workflow, one runner job, one upserted PR comment for the three checkout-based doc-quality checks. It replaces the former `validate-mdx.yml`, `linkcheck-pr.yml`, and `readability-delta.yml` workflows. The job id stays `validate-mdx` so the branch-protection required check keeps its name.

The three checks:

1. **MDX validation** (**blocking**): `mint validate` + `mint broken-links` via `scripts/mdx-validation/validate-mdx-mintlify.sh`, using the Mintlify CLI from npm (cached ~4 days). `mint broken-links` owns *internal* links because it resolves them against the real route table (docs.json navigation and redirects).
2. **External link check** (informational): `lychee` over the changed `.mdx` files only, external URLs only (`lychee.toml` excludes `^file://`). Runs on plain `pull_request` from the git checkout — it does not wait for the Mintlify preview deployment (the old `deployment_status` trigger was vestigial scheduling and carried the DOCS-3041 wrong-PR-resolution bug).
3. **Readability delta** (informational, DOCS-2626): reports how the PR affects the readability of the English docs it changes — the *delta* (before/after) for well-established formulas (Flesch-Kincaid grade, Flesch reading ease, Gunning fog, SMOG), word-weighted across the changed pages, plus an optional AI-agent-comprehension rating from a W&B Inference LLM judge. Localized content under `ja/`, `ko/`, and `fr/` is skipped.

Only MDX validation can fail the job; the other two never fail a PR. All three results land in a single PR comment identified by the `<!-- doc-quality-report -->` marker (status table plus collapsed details). A migration step deletes leftover comments from the replaced workflows (`<!-- lychee-link-checker-comment -->` and `<!-- readability-delta-report -->` markers) on open PRs.

### Triggers

- **Pull request**: the default activity types (`opened`, `synchronize`, `reopened`), deliberately with **no `paths` filter** — a paths filter would leave the required check pending on PRs it skips. A scope-gate `git diff` inside the job fast-exits instead when a PR touches no Mintlify-relevant files (`.mdx`, `.json`, `.yaml`/`.yml`, images, or the validation script).
- **Push**: to `main` (full validation)
- **Manual**: `workflow_dispatch` (writes the report to the job summary instead of a comment). Dispatch runs the full MDX validation only: the link and readability checks diff against `main`, so on `main` itself the change list is empty and they skip. The old `linkcheck-pr.yml` dispatch fell back to a full-tree external-link sweep in that case; that behavior was **not** carried over — the closest substitute is `linkcheck-prod.yml`, which checks the live site's sitemap (monthly cron, also manually dispatchable).

### Configuration

- **Node**: 20 (Mintlify CLI), **Python**: 3.11 (readability)
- **Permissions**: `contents: read`, `pull-requests: write`
- **Validation script**: `scripts/mdx-validation/validate-mdx-mintlify.sh`
- **Link-check config**: `lychee.toml` (repo root)
- **Readability report glue**: `scripts/readability/pr_report.py`
- **Readability scoring logic**: `.claude/scripts/_readability.py` and `_docs_eval_lib.py` (submodule)

### Authentication

- The main checkout, the lychee rate-limit token, and PR comments use the default `GITHUB_TOKEN`.
- The private, cross-org `coreweave/docs-skills` submodule is initialized in a separate step with the `DOCENGINE_TOKEN` secret (the same `x-access-token` credential used for the `gitsubmodule` ecosystem in `.github/dependabot.yml`; it rotates ~every 30 days and needs no `wandb/docs` scope). Readability only.
- The AI agent comprehension judge calls W&B Inference with the `WANDB_DOCS_INFERENCE_API_KEY` secret (a W&B API key whose entity has Inference credits), passed to the scorer as `WANDB_API_KEY`. When that secret is absent, the deterministic `textstat` delta still runs.

### Forks

Fork PRs have no access to repo secrets and get a read-only `GITHUB_TOKEN`. MDX validation and the external link check need no secrets, so they still run (and validation still blocks). The readability delta skips (it needs the private submodule and the Inference key), and the report goes to the workflow step summary instead of a PR comment. Forks are uncommon in `wandb/docs` and coreweave repos cannot use forks at all.

### Related workflows (not consolidated)

- `mintlify-deployment-preview.yml` needs the deployment's `environment_url`, so it stays on `deployment_status`.
- `linkcheck-prod.yml` checks the live site on a monthly cron.
- The PinGuard pair and the commit-pushing bots (Compress Images, Build CSS, Knowledgebase Nav) use the `wandb-docs-pr-writer` App token — a different trust category that should not share a job with fork-facing validation.

### Related Files

- **Readability report glue**: `scripts/readability/pr_report.py`
- **Readability dependencies**: `scripts/readability/requirements.txt`
- **Readability tests**: `scripts/readability/tests/`
- **Readability documentation**: `scripts/readability/README.md`
