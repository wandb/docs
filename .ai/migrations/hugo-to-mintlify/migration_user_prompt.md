# User Prompt for Hugo to Mintlify Migration

## For Contributors: How to Request a Migration

Copy and paste one of these prompts to your AI agent, depending on your situation:

---

### Option 1: If you have a PR number

```
Please migrate PR #[YOUR_PR_NUMBER] from Hugo to Mintlify structure. 

The PR was created when the docs used Hugo, but the repo has since migrated to Mintlify. I've checked out the branch locally for you.

Use the migration prompt template in migration_prompt_template.md to guide you through the process. Start by doing the port locally and validate with the Mintlify CLI before committing.
```

---

### Option 2: If you have a local branch (no PR yet)

```
Please migrate my current branch from Hugo to Mintlify structure.

I have changes in the branch [BRANCH_NAME] that were created for Hugo, but the repo has since migrated to Mintlify. The changes are in:
[List your changed files, e.g., content/en/guides/...]

Use the migration prompt template in migration_prompt_template.md to guide you through the process. Start by doing the port locally and validate with the Mintlify CLI before committing.
```

---

### Option 3: If you have uncommitted local changes

```
Please help me migrate my uncommitted Hugo changes to Mintlify structure.

I have changes in these files that need to be migrated:
[List your files]

The main changes are:
[Brief description of what you changed]

Use the migration prompt template in migration_prompt_template.md to guide you through the process. Please create the appropriate Mintlify files and help me stage them properly.
```

---

## What the Agent Will Do

The agent will:
1. Identify your changes (from PR, branch, or your description)
2. Find where the files moved to in the Mintlify structure
3. Apply your changes with proper syntax conversion
4. Update the navigation in docs.json
5. Test with `mint dev` to ensure no Hugo files remain
6. Create a clean commit with only Mintlify files

## Important Notes for Contributors

- **If you have a PR from the Hugo era**: The agent will help you close it and create a new one after migration (required for Mintlify previews)
- **If you haven't created a PR yet**: Perfect! You can create it directly after migration
- **If you have uncommitted work**: The agent will help you organize it into the proper Mintlify structure

## Prerequisites

Before asking the agent to migrate:
1. Ensure you have the Mintlify CLI installed: `npm i -g mintlify`
2. Make sure you're in the docs repository
3. Have your changes ready (either in a PR, branch, or locally)

## After Migration

The agent will provide you with:
- A clean commit with only Mintlify files
- Instructions for creating a new PR if needed
- Confirmation that `mint dev` runs without errors

## Troubleshooting

If the agent encounters issues:
- **Hugo template errors**: Ask the agent to do a hard reset and recreate the commit
- **Missing content**: Provide the agent with more context about your changes
- **Navigation issues**: Ask the agent to verify the docs.json updates

---

## For Agents: Additional Context

When a user provides one of these prompts, you should:
1. Check if `migration_prompt_template.md` exists in the repo
2. Follow the template systematically
3. If given a PR number, you can fetch the branch with:
   ```bash
   gh pr checkout [PR_NUMBER]  # if using GitHub CLI
   # or
   git fetch origin pull/[PR_NUMBER]/head:pr-[PR_NUMBER]
   git checkout pr-[PR_NUMBER]
   ```
4. If only given a branch name, ensure it's checked out
5. Always validate with `mint dev` before committing
6. Never mix Hugo and Mintlify files in the same commit
