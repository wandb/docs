# SDK Release Notes GitHub Action

This workflow automates the generation of W&B SDK release notes from the `wandb/wandb` repository's CHANGELOG.md file.

## Features

- Generates release notes for a specific SDK version by getting correct release from wandb/wandb with some light post-processing
- Creates a new branch with the generated markdown file
- Opens a draft PR for review
- Can be triggered via PR comments for testing

## Usage

### Method 1: Manual Workflow Dispatch

1. Go to the [Actions tab](../../actions) in the docs repository
2. Select "Generate SDK Release Notes" workflow
3. Click "Run workflow"
4. Enter the SDK version (e.g., `0.21.1`)
5. Click "Run workflow"

### Method 2: PR Comment Trigger (for testing)

While developing or testing, you can trigger the workflow from a PR comment:

```
/sdk-release 0.21.1
```

The bot will:
1. Respond with a confirmation comment
2. Generate the release notes
3. Create a new branch and draft PR
4. Post a success/failure comment with links

## How It Works

1. **Fetches the CHANGELOG.md** from `wandb/wandb` repository
2. **Runs the Python script** to parse and transform the changelog entry
3. **Creates a new branch** named `sdk-release-notes-v{VERSION}`
4. **Commits the generated file** to `content/en/ref/release-notes/sdk/{VERSION}.md`
5. **Opens a draft PR** with a checklist for review

## File Transformations

The Python script (`scripts/sdk-changelog-to-hugo.py`) applies these transformations:
- Removes emojis
- Moves PR links and author info to HTML comments
- Adds periods to sentences
- Fixes whitespace issues
- Removes "Contributors" sections
- Filters out `docs(sdk)` items
- Removes conventional commit prefixes
- Adds a GitHub changelog link at the top

## Local Testing

Test the script locally with a specific version:

```bash
./scripts/test-sdk-single-version.sh 0.21.1
```

Or run the Python script directly:

```bash
python3 scripts/sdk-changelog-to-hugo.py \
  --changelog /path/to/wandb/CHANGELOG.md \
  --output ./test-output \
  --version 0.21.1
```

## Requirements

- Python 3.11+
- `emoji` package (`pip install emoji`)
- Access to `wandb/wandb` repository (public)
- Write access to docs repository (for creating branches/PRs)

## Permissions

The workflow requires these GitHub token permissions:
- `contents: write` - To create branches and commits
- `pull-requests: write` - To create PRs
- `issues: write` - To post comments (for PR trigger)

## Troubleshooting

### Version not found
- Check that the version exists in the wandb/wandb CHANGELOG.md
- Version format should be like `0.21.1` (without 'v' prefix)

### PR comment trigger not working
- Ensure the comment starts exactly with `/sdk-release `
- The workflow only runs on PR comments, not issue comments
- Check that the workflow has the necessary permissions

### File not generated
- Review the workflow logs for Python script errors
- Verify the CHANGELOG.md format hasn't changed
- Test locally with the test script first
