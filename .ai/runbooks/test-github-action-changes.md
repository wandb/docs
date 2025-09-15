# Agent Prompt: Testing GitHub Actions Changes in wandb/docs

## Task Overview
Test changes to GitHub Actions workflows in the wandb/docs repository, particularly the PR preview link generation workflows that depend on Cloudflare Pages deployments.

> **Note**: If you are testing changes to an action that doesn't depend on Cloudflare, adjust your interpretation of this runbook accordingly.

## Context and Constraints

### Repository Setup
- **Main repository**: `wandb/docs` (origin)
- **Fork for testing**: `<username>/docs` (fork remote)
- **Important**: GitHub Actions in PRs always run from the base branch (main), not from the PR branch
- **Cloudflare limitation**: Cloudflare Pages only builds for the main wandb/docs repository, not for forks

These instructions assume that you are using a fork that doesn't contain real work in `main`. An agent can have trouble committing to a fork, so the agent may need to temporarily push a branch to the main repo so the human can push the same branch to the fork.

### Key Workflows
1. `.github/workflows/pr-preview-links.yml` - Runs on PR open/sync
2. `.github/workflows/pr-preview-links-on-comment.yml` - Triggered by Cloudflare comments

### Testing Requirements
To test workflow changes, you must:
1. Sync the fork's `main` with the main repo's `main`, throwing away all temporary commits.
2. Apply changes to the fork's main branch (not just a feature branch)
2. Override Cloudflare URLs since they won't generate for forks
3. Create a test PR with content changes to trigger the workflows

## Step-by-Step Testing Process

### 1. Initial Setup
```bash
# Ensure you have both remotes configured
git remote -v  # Should show 'origin' (wandb/docs) and 'fork' (mdlinville/docs)

# If fork remote is missing:
git remote add fork https://github.com/<username>/docs.git
```

### 2. Prepare Test Branch
```bash
# Start from latest main
git checkout main
git pull origin main

# Create test branch for the workflow changes
git checkout -b test-[description]-[date]
```

### 3. Apply Workflow Changes
Make your changes to the workflow files. For dependency upgrades:
- Update version numbers in `uses:` statements
- Check both workflow files if the dependency is used in multiple places

### 4. Add Cloudflare URL Override
Since Cloudflare won't build for forks, add this override to BOTH workflow files:

For `pr-preview-links.yml`, after the URL extraction logic (around line ~220):
```javascript
// TEMPORARY OVERRIDE FOR FORK TESTING
// Since Cloudflare won't run on forks, use a hardcoded URL
if (!base && context.repo.owner === 'mdlinville') {
  base = 'https://main.docodile.pages.dev';
  core.warning('Using temporary override URL for fork testing: ' + base);
}
```

For `pr-preview-links-on-comment.yml`, after the URL extraction (around line ~126):
```javascript
// TEMPORARY OVERRIDE FOR FORK TESTING
// Since Cloudflare won't run on forks, use a hardcoded URL
if (!branchUrl && context.repo.owner === 'mdlinville') {
  branchUrl = 'https://main.docodile.pages.dev';
  core.warning('Using temporary override URL for fork testing: ' + branchUrl);
}
```

### 5. Commit and Push to Fork's Main
```bash
# Commit all changes
git add -A
git commit -m "test: [Description of what you're testing]

- Add Cloudflare URL override for fork testing
- [Other changes made]"

# Push to fork's main branch
git push fork HEAD:main --force-with-lease
```

Agents: If you can't push to the fork yourself, give the human the command to push from your temporary branch in `wandb/docs` to the fork, then give them the URL to create the PR in the fork. At the end of this runbook, don't forget to prompt them to remove the temporary branch from `wandb/docs`.


### 6. Create Test PR
```bash
# Create new branch from the updated fork main
git checkout -b test-pr-[description]

# Make a small content change to trigger workflows
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# Commit and push
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

Then create PR via GitHub UI from `<username>:test-pr-[description]` to `<username>:main`

### 7. Monitor and Verify

Expected behavior:
1. GitHub Actions bot creates initial comment with "Generating preview links..."
2. Workflow should complete without errors
3. Comment should update with preview links pointing to `https://main.docodile.pages.dev/...`

Check for:
- ✅ Workflow completes successfully
- ✅ Preview comment is created and updated
- ✅ Links use the override URL
- ✅ File categorization works (Added/Modified/Deleted/Renamed)
- ❌ Any errors in the Actions logs
- ❌ Security warnings or exposed secrets

### 8. Cleanup
After testing:
```bash
# Reset fork's main to match upstream
git checkout main
git fetch origin
git reset --hard origin/main
git push fork main --force

# Delete test branches from fork and origin
git branch -D test-[description]-[date] test-pr-[description]
```

## Common Issues and Solutions

### Issue: Permission denied when pushing to fork
- The GitHub token might be read-only
- Solution: Use SSH or push manually from your local machine

### Issue: Workflows not triggering
- Remember: Workflows run from the base branch (main), not the PR branch
- Ensure changes are in fork's main branch

### Issue: Preview links not generating
- Check if Cloudflare override is properly added
- Verify the override URL is correct: `https://main.docodile.pages.dev`

### Issue: Changed files not detected
- Ensure content changes are in tracked directories (content/, static/, assets/, etc.)
- Check the `files:` filter in the workflow configuration

## Testing Checklist

- [ ] Both remotes (origin and fork) are configured
- [ ] Workflow changes applied to both relevant files
- [ ] Cloudflare URL override added with fork owner check
- [ ] Changes pushed to fork's main branch
- [ ] Test PR created with content changes
- [ ] Preview comment generated successfully
- [ ] No errors in GitHub Actions logs
- [ ] Fork's main branch reset after testing

## Notes
- The Cloudflare preview domain is `docodile.pages.dev`
- Branch previews normally use pattern: `https://[branch-name].docodile.pages.dev`
- The override uses the main branch preview as a stable fallback
- Always remember to remove the Cloudflare override before merging to production