# Agent prompt: Testing GitHub Actions changes in wandb/docs

## Requirements
- **W&B employee access**: You must be a W&B employee with access to internal W&B systems.
- **GitHub fork**: A personal fork of wandb/docs for testing workflow changes. In the fork, you need permission to push to the default branch and bypass branch protection rules.

## Agent prerequisites
Before starting, gather this information:
1. **GitHub username** - Check `git remote -v` first for fork remote, then `git config` for username. Only ask the user if not found in either location.
2. **Fork status** - Confirm they have a fork of wandb/docs with permission to push to the default branch and bypass branch protection.
3. **Test scope** - Ask what specific changes are being tested (dependency upgrade, functionality change, etc.).

## Task overview
Test changes to GitHub Actions workflows in the wandb/docs repository.


## Context and constraints

### Repository setup
- **Main repository**: `wandb/docs` (origin)
- **Fork for testing**: `<username>/docs` (fork remote) - If not clear from `git remoter -v`, ask the user for their fork's endpoint.
- **Important**: GitHub Actions in PRs always run from the base branch (main), not from the PR branch.
- **Mintlify deploy limitation**: Mintlify deployments and the `link-rot` check only build for the main wandb/docs repository, not for forks. In a fork, the `validate-mdx` Github Action checks the status of `mint dev` and `mint broken-links` commands in a fork PR.

**Agent note**: You need to:
1. Check `git remote -v` for existing fork remote and extract username from the URL if present.
2. If username not found in remotes, check `git config` for GitHub username.
3. Only ask the user for their GitHub username if not found in either location.
4. Verify they have a fork of wandb/docs that can be used for testing.
5. If you can't push to the fork directly, create a temporary branch in wandb/docs for the user to push from.

### Testing requirements
To test workflow changes, you must:
1. Sync the fork's `main` with the main repo's `main`, throwing away all temporary commits.
2. Apply changes to the fork's main branch (not just a feature branch)
3. Create a test PR against the fork's `main` with content changes to trigger the workflows.

## Step-by-step testing process

### 1. Initial setup
```bash
# Check existing remotes
git remote -v

# If fork remote exists, note the username from the fork URL
# If fork remote is missing, check git config for username
git config user.name  # or git config github.user

# Only ask user for their GitHub username or fork details if not found in remotes or config
# Example question: "What is your GitHub username for the fork we'll use for testing?"

# If fork remote is missing, add it:
git remote add fork https://github.com/<username>/docs.git  # Replace <username> with actual username
```

### 2. Sync fork and prepare test branch
```bash
# Fetch latest from origin
git fetch origin

# Checkout main and hard reset to origin/main to ensure clean sync
git checkout main
git reset --hard origin/main

# Force push to fork to sync it (throwing away any temporary commits in fork)
git push fork main --force

# Create test branch for the workflow changes
git checkout -b test-[description]-[date]
```

### 3. Apply workflow changes
Make your changes to the workflow files. For dependency upgrades:
- Update version numbers in `uses:` statements
- Check both workflow files if the dependency is used in multiple places

**Pro tip**: Before finalizing any runbook, ask an AI agent to review it with a prompt like:
> "Please review this runbook and suggest improvements to make it more useful for AI agents. Focus on clarity, completeness, and removing ambiguity."

### 5. Commit and push to fork's main
```bash
# Commit all changes
git add -A
git commit -m "test: [Description of what you're testing]"

# Push to fork's main branch
git push fork HEAD:main --force-with-lease
```

**Agent instructions for fork access**:
If you can't push to the fork directly:
1. Create a temporary branch in wandb/docs with the changes
2. Provide the user with this command:
   ```bash
   git fetch origin temp-branch-name
   git push fork origin/temp-branch-name:main --force
   ```
3. Guide them to create the PR at: `https://github.com/<username>/docs/compare/main...test-pr-[description]`
4. Remember to delete the temporary branch from wandb/docs after testing


### 6. Create test PR
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

Then create PR through GitHub UI from `<username>:test-pr-[description]` to `<username>:main`

### 7. Monitor and verify

Expected behavior:
1. GitHub Actions bot creates initial comment with "Generating preview links..."
2. Workflow should complete without errors

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

## Common issues and solutions

### Issue: Permission denied when pushing to fork
- The GitHub token might be read-only
- Solution: Use SSH or push manually from your local machine

### Issue: Workflows not triggering
- Remember: Workflows run from the base branch (main), not the PR branch
- Ensure changes are in fork's main branch

### Issue: Changed files not detected
- Ensure content changes are in tracked directories (content/, static/, assets/, etc.)
- Check the `files:` filter in the workflow configuration

## Testing checklist

- [ ] Asked user for their GitHub username and fork details
- [ ] Both remotes (origin and fork) are configured
- [ ] Workflow changes applied to both relevant files
- [ ] Changes pushed to fork's main branch (directly or through user)
- [ ] Test PR created with content changes
- [ ] Preview comment generated successfully
- [ ] No errors in GitHub Actions logs
- [ ] Fork's main branch reset after testing
- [ ] Temporary branches cleaned up from wandb/docs (if created)
