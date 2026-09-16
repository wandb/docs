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

Workflows that push back to a same-repo PR branch with this token (instead of the default workflow `GITHUB_TOKEN`) include **Compress Images** (`calibreapp-image-actions.yml`), **Build CSS** (`build-css.yml`), and **Knowledgebase Nav** (`knowledgebase-nav.yml`). That way downstream `pull_request` checks (for example **Validate MDX**) still run on the automation commit.

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

## UI label drift

**Workflow**: `uidrift-scan.yml`

Watches `wandb/core` for user-facing label changes that leave this repo's docs stale, and carries the resulting report in one rolling draft PR. The detector is `scripts/uidrift`; see [`scripts/uidrift/ADAPTING.md`](../../scripts/uidrift/ADAPTING.md) for what it looks for and why. This workflow is only the sink.

### Setup required before the first run

`wandb-docs-source-reader` is installed on `wandb/docs-code-eval` and `wandb/weave-internal` only, so **it cannot read `wandb/core` yet**. Pick one:

- **Preferred**: install `wandb-docs-source-reader` on `wandb/core` with **Contents: read**. Needs a `wandb` org owner. No secret changes here; the workflow already asks for `repositories: core`.
- **Fallback**: add a repository secret `WANDB_CORE_TOKEN` holding a token that can read `wandb/core`. The workflow prefers the App and falls back to this, so adding the App install later needs no edit.

With neither in place the first step fails immediately and names both options, rather than burning four minutes on a clone that cannot authenticate.

### Triggers

- **Scheduled**: weekdays at 13:00 UTC (6am PT), so a report is waiting at standup
- **Manual**: `workflow_dispatch` with `since` (window start for a non-incremental run), `seed` (ignore existing reports and rescan the whole window), and `dry-run` (report to the job summary, open no PR)

### What it does

1. Clones `wandb/core` — full history, single branch, no working tree. The ADAPTING.md table records why shallow and blobless clones were both rejected; do not "optimize" this without reading it.
2. Runs the scan. `--incremental` by default, taking its base from the head SHA in the newest report filename under `uidrift/reports/`; falls back to `--since` when no report exists yet.
3. Writes the report to the job summary, so a run is readable even when it opens no PR.
4. If there are findings (or a reopened decision), opens or updates a **draft PR** on the rolling branch `uidrift/drift-report` with the funnel counts, lane breakdown, and how to record a decision.
5. Fails the run — after the PR exists — if any stored decision reopened. That means a writer's earlier dismissal no longer matches the docs, which only a human can settle.

Merging the PR advances the watermark. Closing it unmerged is also safe: the next run rescans the same range and supersedes the report.

### Reviewing a report

Each row lands in one of three lanes: **agent** (mechanical rename, safe to apply), **pair** (a writer scopes it, an agent applies it), **human** (prose has to be written). Rows that are wrong get recorded rather than deleted:

```bash
PYTHONPATH=scripts python3 -m uidrift.scan decide <id> \
    --status dismissed --by <you> --agreement false_positive --note '<why>'
```

`--agreement` is the detector's only feedback channel and cannot be reconstructed later. A dismissal reopens by itself if docs later start covering that surface, so it suppresses a row without hiding it forever.

## Readability delta

**Workflow**: `readability-delta.yml`

Posts an informational, **non-blocking** PR comment describing how the PR affects the readability of the English docs it changes (DOCS-2626). It reports the *delta* (before/after) for well-established formulas (Flesch-Kincaid grade, Flesch reading ease, Gunning fog, SMOG), word-weighted across the changed pages, plus an optional AI-agent-comprehension rating from a W&B Inference LLM judge.

### Triggers

- **Pull request**: `opened`, `synchronize`, `reopened` on PRs that touch `**/*.mdx`
- **Manual**: `workflow_dispatch` (writes the report to the job summary instead of a comment)

### What it does

1. Diffs the PR base and head, scoring each changed English `.mdx` file (localized content under `ja/`, `ko/`, and `fr/` is skipped).
2. Extracts narrative prose and scores it with `textstat` via the analyzer in the `coreweave/docs-skills` submodule (`.claude/scripts/_readability.py`).
3. Optionally runs the AI agent comprehension judge (W&B Inference) when `WANDB_API_KEY` is set.
4. Upserts a single PR comment identified by the `<!-- readability-delta-report -->` marker.

The check **never fails** a PR. If scoring is unavailable it posts a brief notice and exits successfully.

### Configuration

- **Python**: 3.11
- **Permissions**: `contents: read`, `pull-requests: write`
- **Report glue**: `scripts/readability/pr_report.py`
- **Scoring logic**: `.claude/scripts/_readability.py` and `_docs_eval_lib.py` (submodule)

### Authentication

- The main checkout uses the default `GITHUB_TOKEN`.
- The private, cross-org `coreweave/docs-skills` submodule is initialized in a separate step with the `DOCENGINE_TOKEN` secret (the same `x-access-token` credential used for the `gitsubmodule` ecosystem in `.github/dependabot.yml`; it rotates ~every 30 days and needs no `wandb/docs` scope).
- The AI agent comprehension judge calls W&B Inference with the `WANDB_DOCS_INFERENCE_API_KEY` secret (a W&B API key whose entity has Inference credits), passed to the scorer as `WANDB_API_KEY`. When that secret is absent, the deterministic `textstat` delta still runs.

### Forks

Fork PRs have no access to repo secrets, so the first step detects a fork, posts an Actions notice, and makes the whole job a no-op (still reporting success). Forks are uncommon in `wandb/docs` and coreweave repos cannot use forks at all.

### Related Files

- **Report glue**: `scripts/readability/pr_report.py`
- **Dependencies**: `scripts/readability/requirements.txt`
- **Tests**: `scripts/readability/tests/`
- **Documentation**: `scripts/readability/README.md`
