# Testing Dependabot PR #1647 - tj-actions/changed-files v46 to v47 Upgrade

## Summary
This test verifies that upgrading `tj-actions/changed-files` from v46 to v47 doesn't break the PR preview links GitHub Actions.

## Changes Made

1. **Upgraded tj-actions/changed-files from v46 to v47** in:
   - `.github/workflows/pr-preview-links.yml`
   - `.github/workflows/pr-preview-links-on-comment.yml`

2. **Added temporary Cloudflare URL override** for testing in forks:
   - Both workflows now check if `context.repo.owner === 'mdlinville'`
   - If true and no Cloudflare URL is found, uses `https://main.docodile.pages.dev` as fallback
   - This allows testing without actual Cloudflare builds

3. **Added test content change**:
   - Modified `content/en/guides/quickstart.md` description to trigger the preview workflow

## Steps to Test

1. **Apply the changes to your fork's main branch**:
   ```bash
   # In your local mdlinville/docs clone
   git checkout main
   git pull origin main  # Make sure you're up to date with wandb/docs
   
   # Apply the patch
   git apply test-dependabot-pr-1647.patch
   
   # Or cherry-pick the commits
   git cherry-pick 12ea7b4795174228a52557c5dbc4ae1f25bebfb3
   git cherry-pick 0158d72
   git cherry-pick 233dc3e
   
   # Push to your fork's main
   git push fork main
   ```

2. **Create a test PR in your fork**:
   ```bash
   # Create a new branch
   git checkout -b test-pr-preview-v47
   
   # Make a small content change
   echo "Test change" >> content/en/guides/quickstart.md
   git add content/en/guides/quickstart.md
   git commit -m "test: Verify PR preview with tj-actions v47"
   
   # Push and create PR
   git push fork test-pr-preview-v47
   ```

3. **Verify the PR preview comment**:
   - The GitHub Action should run and create a preview comment
   - Links should use `https://main.docodile.pages.dev` as the base URL
   - The action should complete successfully without errors

4. **Clean up after testing**:
   - Reset your fork's main branch to match wandb/docs main
   - Close the test PR

## Expected Results

- The PR preview GitHub Actions should run successfully
- A comment should be created with preview links
- Links should point to `https://main.docodile.pages.dev/...`
- No errors related to tj-actions/changed-files v47

## Notes

- The Cloudflare URL override is temporary and should be removed after testing
- The override only activates for the `mdlinville` fork to avoid affecting the main repo