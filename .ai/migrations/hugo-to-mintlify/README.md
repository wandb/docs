# Hugo to Mintlify Migration Guide

This directory contains templates and prompts for migrating documentation from Hugo to Mintlify structure.

## Quick Start for Humans

If you have a PR or branch that was created when the docs used Hugo, follow these steps:

### 1. Identify Your Situation

- **Have a PR number?** → Use Option 1 in `migration_user_prompt.md`
- **Have a local branch?** → Use Option 2 in `migration_user_prompt.md`  
- **Have uncommitted changes?** → Use Option 3 in `migration_user_prompt.md`

### 2. Request the Migration

1. Open your AI assistant (Cursor, GitHub Copilot Chat, etc.)
2. Copy the appropriate prompt from `migration_user_prompt.md`
3. Fill in the placeholders (PR number, branch name, or file list)
4. Paste to your AI assistant

### 3. What to Expect

The AI agent will:
- Identify your Hugo changes
- Find where files moved in Mintlify structure
- Convert Hugo syntax to Mintlify components
- Update navigation in `docs.json`
- Create a clean commit with only Mintlify files

### 4. After Migration

- **If you had a PR**: Close the old PR and create a new one from the migrated branch
- **If you had a branch**: Create a new PR directly
- **Why?** PRs created during Hugo era don't get Mintlify preview deployments

## Files in This Directory

### `migration_prompt_template.md`
Detailed step-by-step instructions for AI agents to perform the migration. This is the "how-to" guide the AI follows.

### `migration_user_prompt.md`
Ready-to-use prompts for humans to copy/paste when requesting a migration. Pick the option that matches your situation.

## Common Issues and Solutions

### "Hugo template errors when running mint dev"
The AI needs to do a hard reset and recreate the commit with only Mintlify files. Ask:
```
I'm getting Hugo template errors. Please hard reset to origin/main and recreate the commit with only Mintlify files.
```

### "My new page doesn't appear in the navigation"
The AI needs to update `docs.json`. Ask:
```
Please add my new page to the navigation in docs.json
```

### "The PR preview isn't working"
This is expected for PRs created during Hugo era. You must:
1. Close the old PR (don't delete the branch)
2. Create a new PR from the same branch
3. The new PR will get Mintlify previews

## Prerequisites

Before starting a migration:
1. Install Mintlify CLI: `npm i -g mintlify`
2. Have your changes ready (in a PR, branch, or locally)
3. Be in the docs repository

## Examples of Successful Migrations

- PR #1627 → PR #1802 (org_dashboard documentation)
- PR #1563 (automation history documentation)

## Need Help?

If the migration fails or you encounter issues:
1. Check that you're on the latest `main` branch
2. Ensure no Hugo files are staged (only `models/`, `platform/`, etc.)
3. Verify `mint dev` runs without errors
4. Ask the AI to show you what files it's trying to commit

## Technical Notes

### File Structure Changes
- **Hugo**: `content/en/guides/` → **Mintlify**: Root-level directories
- **Hugo**: `.md` files → **Mintlify**: `.mdx` files
- **Hugo**: `content/en/guides/core/` → **Mintlify**: `models/`
- **Hugo**: `content/en/guides/hosting/` → **Mintlify**: `platform/`

### Syntax Conversions
- **Tabs**: `{{< tabpane >}}` → `<Tabs>`
- **Links**: `{{< relref >}}` → Direct paths
- **Comments**: `<!-- -->` → `{/* */}`
- **Includes**: `{{< readfile >}}` → Import statements or inline content
