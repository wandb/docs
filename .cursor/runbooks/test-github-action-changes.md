# Testing GitHub Actions changes in wandb/docs

## Overview

<task_overview>
This runbook guides you through testing changes to GitHub Actions workflows in the wandb/docs repository, particularly the PR preview link generation workflows that depend on Cloudflare Pages deployments.

**Note**: If you are testing changes to an action that doesn't depend on Cloudflare, adjust your interpretation of this runbook accordingly.
</task_overview>

## Requirements

<requirements>
- [ ] **W&B employee access**: You must be a W&B employee with access to internal W&B systems
- [ ] **GitHub fork**: A personal fork of wandb/docs for testing workflow changes
- [ ] **Git push access**: Ability to push to either the fork or wandb/docs repository
</requirements>

## Prerequisites

<prerequisites>
Information to gather from the user before starting:

1. **GitHub username** - Needed to identify their fork
2. **Fork status** - Confirm they have a fork of wandb/docs that can be used for testing
3. **Test scope** - What specific changes are being tested (dependency upgrade, functionality change, etc.)
4. **Push access** - Whether you can push directly to their fork or need to use wandb/docs
</prerequisites>

## Context and constraints

<context>
### Repository setup
- **Main repository**: `wandb/docs` (origin)
- **Fork for testing**: `<username>/docs` (fork remote)
- **Important**: GitHub Actions in PRs always run from the base branch (main), not from the PR branch.
- **Cloudflare limitation**: Cloudflare Pages only builds for the main wandb/docs repository, not for forks.

### Key workflows
1. `.github/workflows/pr-preview-links.yml` - Runs on PR open/sync
2. `.github/workflows/pr-preview-links-on-comment.yml` - Triggered by Cloudflare comments

### Testing requirements
To test workflow changes, you must:
1. Sync the fork's main with the upstream main
2. Apply changes to the fork's main branch (not just a feature branch)
3. Override Cloudflare URLs since they won't generate for forks
4. Create a test PR with content changes to trigger the workflows

### Agent limitations
- May not have direct push access to user's fork
- Will need to create temporary branches in wandb/docs for user to push
</context>

## Step-by-step process

<process>
### Step 1: Initial setup and information gathering

First, gather required information:
```bash
# Ask the user for their GitHub username
# Example prompt: "What is your GitHub username for the fork we'll use for testing?"
```

Configure remotes:
```bash
# Check existing remotes
git remote -v

# Add fork remote if missing
git remote add fork https://github.com/<username>/docs.git
```

**Expected result**: Both 'origin' (wandb/docs) and 'fork' (<username>/docs) remotes configured.

### Step 2: Prepare test branch

Create a branch for the workflow changes:
```bash
# Start from latest main
git checkout main
git pull origin main

# Create test branch with descriptive name
git checkout -b test-[description]-[date]
# Example: test-dependabot-upgrade-20250127
```

**Agent note**: Use descriptive branch names that indicate what's being tested and when.

### Step 3: Apply workflow changes

Make the necessary changes to workflow files. Common scenarios:

**For dependency upgrades**:
- Update version numbers in `uses:` statements
- Check both workflow files if the dependency is used in multiple places

**For functionality changes**:
- Modify the workflow logic as needed
- Ensure changes are compatible with GitHub Actions syntax

### Step 4: Add Cloudflare URL override

Since Cloudflare won't build for forks, add temporary overrides.

**For `pr-preview-links.yml`** (after URL extraction logic, around line ~220):
```javascript
// TEMPORARY OVERRIDE FOR FORK TESTING
// Since Cloudflare won't run on forks, use a hardcoded URL
if (!base && context.repo.owner === '<username>') {  // Replace <username>
  base = 'https://main.docodile.pages.dev';
  core.warning('Using temporary override URL for fork testing: ' + base);
}
```

**For `pr-preview-links-on-comment.yml`** (after URL extraction, around line ~126):
```javascript
// TEMPORARY OVERRIDE FOR FORK TESTING
// Since Cloudflare won't run on forks, use a hardcoded URL
if (!branchUrl && context.repo.owner === '<username>') {  // Replace <username>
  branchUrl = 'https://main.docodile.pages.dev';
  core.warning('Using temporary override URL for fork testing: ' + branchUrl);
}
```

**Agent note**: Replace `<username>` with the actual GitHub username provided by the user.

### Step 5: Commit and push to fork's main

```bash
# Stage all changes
git add -A

# Create descriptive commit
git commit -m "test: [Description of what you're testing]

- Add Cloudflare URL override for fork testing
- [Other changes made]"

# Push to fork's main branch
git push fork HEAD:main --force-with-lease
```

**Agent note**: If you cannot push to the fork directly:
1. Push to a temporary branch in wandb/docs
2. Provide the user with commands to push from there to their fork
3. Guide them through creating the PR
4. Remember to clean up the temporary branch afterward

**Fallback instructions for users**:
```bash
# If agent created temp-branch-name in wandb/docs:
git fetch origin temp-branch-name
git push fork origin/temp-branch-name:main --force
```

### Step 6: Create test PR

Create a branch with content changes to trigger workflows:
```bash
# Create new branch from updated fork main
git checkout -b test-pr-[description]

# Add small content change
echo "<!-- Test change for PR preview -->" >> content/en/guides/quickstart.md

# Commit and push
git add content/en/guides/quickstart.md
git commit -m "test: Add content change to trigger PR preview"
git push fork test-pr-[description]
```

**Agent note**: Direct user to create PR at: `https://github.com/<username>/docs/compare/main...test-pr-[description]`

### Step 7: Monitor and verify

Monitor the PR for expected behavior:
1. GitHub Actions bot should create initial comment with "Generating preview links..."
2. Workflow should complete without errors
3. Comment should update with preview links pointing to override URL

Check Actions tab for any errors or warnings.
</process>

## Verification

<verification>
### Expected outcomes
- ✓ Workflow completes successfully
- ✓ Preview comment is created and updated
- ✓ Links use the override URL (https://main.docodile.pages.dev)
- ✓ File categorization works (Added/Modified/Deleted/Renamed)
- ✓ No security warnings or exposed secrets

### How to verify success
1. Check the Actions tab in the PR for green checkmarks
2. Verify the preview comment contains correct links
3. Review workflow logs for any warnings
4. Confirm file change detection is accurate
</verification>

## Troubleshooting

<troubleshooting>
### Issue: Permission denied when pushing to fork
- **Symptoms**: Git push fails with permission error.
- **Cause**: No write access to user's fork.
- **Solution**: Create temporary branch in wandb/docs and provide push instructions to user.

### Issue: Workflows not triggering
- **Symptoms**: No GitHub Actions run after creating PR.
- **Cause**: Workflows run from base branch (main), not PR branch.
- **Solution**: Ensure changes are pushed to fork's main branch, not just feature branch.

### Issue: Preview links not generating
- **Symptoms**: Comment stuck on "Generating preview links...".
- **Cause**: Missing or incorrect Cloudflare override.
- **Solution**: Verify override is added with correct username in owner check.

### Issue: Changed files not detected
- **Symptoms**: Preview comment shows no files changed.
- **Cause**: Content changes not in tracked directories.
- **Solution**: Ensure changes are in content/, static/, assets/, or other tracked directories.
</troubleshooting>

## Cleanup

<cleanup>
After testing is complete:

1. **Reset fork's main to match upstream**:
   ```bash
   git checkout main
   git fetch origin
   git reset --hard origin/main
   git push fork main --force
   ```

2. **Delete test branches**:
   ```bash
   # Delete local branches
   git branch -D test-[description]-[date] test-pr-[description]
   
   # Delete remote branches if needed
   git push fork --delete test-pr-[description]
   ```

3. **Clean up any temporary branches in wandb/docs**:
   ```bash
   # If you created temporary branches
   git push origin --delete temp-branch-name
   ```

4. **Close test PR** (or ask user to close it)
</cleanup>

## Summary checklist

<checklist>
- [ ] Asked user for GitHub username and fork details
- [ ] Configured both remotes (origin and fork)
- [ ] Applied workflow changes to both relevant files
- [ ] Added Cloudflare URL override with correct username
- [ ] Pushed changes to fork's main branch
- [ ] Created test PR with content changes
- [ ] Verified preview comment generated successfully
- [ ] Checked Actions logs for errors
- [ ] Reset fork's main branch after testing
- [ ] Cleaned up all temporary branches
- [ ] Closed test PR
</checklist>

## Additional notes

<notes>
- The Cloudflare preview domain is `docodile.pages.dev`
- Branch previews normally use pattern: `https://[branch-name].docodile.pages.dev`
- The override uses the main branch preview as a stable fallback
- Always remove the Cloudflare override before merging to production
- This runbook is for W&B employees only due to internal system dependencies
- For non-Cloudflare workflows, skip steps 4 and adjust verification accordingly
</notes>
