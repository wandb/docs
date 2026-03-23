# GitHub Actions Workflows

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
- **Action**: `peter-evans/create-pull-request@v6`

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

