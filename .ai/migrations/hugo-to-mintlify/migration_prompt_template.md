# LLM Prompt Template for Migrating Hugo PRs to Mintlify

## Context
You are helping migrate a pull request that was created when the W&B docs used Hugo as the static site generator, but the repository has since been migrated to Mintlify. The file paths and structure have changed significantly during this migration.

## Task
Migrate the changes from a Hugo-based branch to work with the current Mintlify structure.

## Step-by-Step Instructions

### 1. Identify the Changes
First, identify what changes were made in the original PR:
```bash
# View the commits in the branch
git log --oneline origin/main..HEAD

# See which files were modified
git diff origin/main...HEAD --name-only

# Save the actual changes for reference
git diff origin/main...HEAD > /tmp/pr-changes.diff
```

### 2. Understand the File Structure Migration
The main file structure changes from Hugo to Mintlify are:
- **Hugo**: `content/en/guides/` → **Mintlify**: root level directories
- **Hugo**: `content/en/guides/hosting/` → **Mintlify**: `platform/hosting/`
- **Hugo**: `content/en/guides/` → **Mintlify**: `models/`
- **Hugo**: `.md` files → **Mintlify**: `.mdx` files
- **Hugo**: Uses `{{< >}}` shortcodes → **Mintlify**: Uses React components like `<Tabs>`, `<Tab>`

### 3. Find the New File Locations
For each file that was modified in the Hugo version:
```bash
# Find the file in the new structure (example for org_dashboard)
git ls-tree -r origin/main --name-only | grep -E "org_dashboard\.(md|mdx)"
```

### 4. Apply the Changes

**Critical**: If you encounter Hugo template errors when running `mint dev`, you likely have leftover Hugo files. The cleanest approach is to hard reset and recreate the commit with only Mintlify content:

1. **Initial attempt - Apply changes to new locations**:
   ```bash
   git checkout origin/main -- <new-file-path>
   ```
   - Apply the changes to the new file structure
   - Update any Hugo-specific syntax to Mintlify syntax

2. **If Hugo errors persist, hard reset and recreate**:
   ```bash
   # Save your work (optional - Git keeps it in reflog anyway)
   git commit -m "temp: migration attempt"
   
   # Hard reset to clean main
   git reset --hard origin/main
   
   # Retrieve your Mintlify files from the previous commit
   git checkout HEAD@{1} -- models/<your-new-files>
   # Or cherry-pick specific files from the commit
   
   # Stage only the Mintlify files
   git add models/ docs.json
   ```

3. **Clean approach - Avoid mixing Hugo and Mintlify**:
   - Never stage or commit old Hugo files from `content/en/guides/`
   - Only work with files in the new Mintlify structure
   - If you accidentally commit Hugo files, hard reset and start fresh

### 5. Common Syntax Conversions

#### Tabs
**Hugo**:
```markdown
{{< tabpane text=true >}}
{{% tab header="Dedicated / Self-Managed" value="dedicated" %}}
Content for dedicated deployment
{{% /tab %}}
{{% tab header="Multi-tenant Cloud" value="saas" %}}
Content for multi-tenant cloud
{{% /tab %}}
{{< /tabpane >}}
```

**Mintlify**:
```markdown
<Tabs>
<Tab title="Dedicated / Self-Managed">
Content for dedicated deployment
</Tab>
<Tab title="Multi-tenant Cloud">
Content for multi-tenant cloud
</Tab>
</Tabs>
```

Note: 
- The `header` attribute in Hugo becomes the `title` attribute in Mintlify
- The `value` attribute from Hugo is not needed in Mintlify
- Tab titles commonly used: "Dedicated Cloud / Self-Managed" and "Multi-tenant Cloud"

#### Links
**Hugo**:
```markdown
[Link text]({{< relref "/path/to/file.md" >}})
```

**Mintlify**:
```markdown
[Link text](/path/to/file)
```

#### Comments
**Hugo**:
```markdown
<!-- HTML comment -->
```

**Mintlify**:
```markdown
{/* JSX comment */}
```

### 6. Update Navigation in docs.json

**Critical**: New pages must be added to the navigation in `docs.json` or they won't appear in the documentation.

```bash
# Find the appropriate section in docs.json
grep -n "automations" docs.json  # or your relevant section

# Add your new page to the appropriate group's "pages" array
# Example: adding view-automation-history to automations section:
"models/automations/view-automation-history",
```

The navigation structure typically mirrors the file structure but needs explicit configuration.

### 7. Validate the Changes
```bash
# Check for broken links and syntax errors
npx mint broken-links

# Run local development server to preview
mint dev

# If you see Hugo template errors like:
# "error missed comma between flow collection entries"
# This means Hugo files are still present - do a hard reset and recreate
```

### 8. Commit the Migration
```bash
# Stage ONLY Mintlify files - never stage content/en/guides/
git add models/ docs.json

git commit -m "[TICKET-ID] Description of changes

- Brief summary of what was changed
- Note that this was migrated from Hugo to Mintlify structure
- Port from PR #XXXX which was created during Hugo era"
```

### 9. Handle Mintlify Preview Deployment Issue

**Important**: If your PR was originally created when the repo used Hugo, Mintlify's preview deployment will be skipped, even after rebasing or force-pushing. This is a known Mintlify limitation.

**Workaround**:
1. Close the original PR (don't delete the branch)
2. Rebase your branch to a single commit against current `origin/main`:
   ```bash
   git rebase -i origin/main
   # Squash all commits into one
   ```
   Or completely recreate the commit:
   ```bash
   git reset --soft origin/main
   git add <your-files>
   git commit -m "[TICKET-ID] Your commit message"
   git push --force origin <your-branch>
   ```
3. Create a new PR from the same branch
4. Add a comment in the original (closed) PR linking to the new one:
   ```
   Moved to #[NEW_PR_NUMBER] due to Mintlify preview deployment requirements
   ```

This ensures the Mintlify bot will properly deploy a preview for review.

## Important Notes

1. **Don't try to merge**: The structural changes are too significant. Instead, apply changes manually to the new structure.

2. **Hard reset if needed**: If you encounter Hugo template errors, the cleanest solution is to `git reset --hard origin/main` and recreate the commit with only Mintlify files. You can retrieve your work from git reflog or the previous commit.

3. **Never mix Hugo and Mintlify files**: Only stage and commit files from the new Mintlify structure (`models/`, `platform/`, etc.). Never commit files from `content/en/guides/`.

4. **Update navigation**: New pages must be added to `docs.json` or they won't appear in the documentation.

5. **Mintlify preview limitation**: PRs created during the Hugo era won't get Mintlify preview deployments. You must close the old PR and create a new one from the same branch (see step 9).

6. **Test thoroughly**: The syntax differences between Hugo and Mintlify can cause rendering issues. Always test locally with `mint dev`.

7. **Check for broken links**: File paths have changed, so internal links may need updating.

8. **Preserve the intent**: Focus on preserving the intent and improvements of the original PR, not necessarily the exact same implementation.

## Example Migration

Real example: [PR #1627](https://github.com/wandb/docs/pull/1627) → [PR #1802](https://github.com/wandb/docs/pull/1802)

If the original PR changed `content/en/guides/hosting/monitoring-usage/org_dashboard.md` to improve user activity documentation:

1. Find the new location: `platform/hosting/monitoring-usage/org_dashboard.mdx`
2. Check out the current version from main
3. Apply the content improvements (reorganized sections, clearer descriptions, CSV schema details, etc.)
4. Convert Hugo shortcodes to Mintlify components:
   - `{{< tabpane >}}` → `<Tabs>`
   - `{{% tab header="..." %}}` → `<Tab title="...">`
5. Update internal links to use new paths:
   - `{{< relref "/guides/hosting/..." >}}` → `/platform/hosting/...`
6. Delete the old Hugo file after migration
7. Test with `mint dev`
8. If the original PR was from the Hugo era, close it and create a new PR from the same branch to enable Mintlify previews

This approach ensures that the valuable content improvements from the original PR are preserved while working within the new Mintlify structure.
