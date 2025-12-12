# W&B SDK code examples

This directory contains code examples synced from the [docs-code-eval](https://github.com/wandb/docs-code-eval) repository. These examples serve dual purposes:

1. **Ground truth for LLM evaluation**: For benchmarking a code generation model's capability to generate W&B code examples.
2. **SDK coding cheat sheet reference**: To help readers and LLMs write W&B code.

## Files

- **`snippets/en/_includes/code-examples/`**: Python code examples synced from docs-code-eval.
- **`snippets/CodeSnippet.jsx`**: Auto-generated React component that imports all code examples.

## Syncing examples

### Github action

Use the GitHub Action workflow to sync examples automatically:

1. Go to the [`sync-code-examples` workflow](https://github.com/wandb/docs/actions/workflows/sync-code-examples.yml).
2. Click **Run workflow**.
3. Wait for workflow to complete.
4. If changes detected, review the draft PR linked in the workflow log.
5. Mark as ready for review, get peer review, and merge the PR.

The workflow does not run on a schedule.

### Python script

To update these examples manually from the latest version in `docs-code-eval`, run the script:

```bash
# From the docs repository root
./scripts/sync_code_examples.sh
```

This script will:

1. Clone the latest `docs-code-eval` repository (temporarily).
2. Copy Python files from `ground_truth/` to `snippets/en/_includes/code-examples/`.
3. Generate the `CodeSnippet.jsx` component with imports for all Python files.
4. Generate the cheat sheet pages at `models/ref/sdk-coding-cheat-sheet/` by extracting metadata directly from the Python docstrings.
5. Clean up the temporary clone.

The changes are ready to commit, push the branch, and create a PR.

## Manual updates

To update individual examples:

1. Make changes in the `wandb/docs-code-eval` repository first.
2. Run the sync script in the `wandb/docs` repository to sync the code examples.
3. Commit the updated files, push the branch, and create a PR.

**Do not edit the Python files or CodeSnippet.jsx directly in the docs repo** as they will be overwritten on the next sync. Make changes in the upstream `docs-code-eval` repository.

## Example structure

Each example in `docs-code-eval` follows this format:

```python
"""
Task description that becomes the prompt for LLM evaluation.

Additional context about the task.
"""
import wandb

# Code demonstrating the W&B pattern
# Uses <placeholder> syntax for values users should replace
```

During sync, these are copied directly as `.py` files and imported into the `CodeSnippet.jsx` component.

Examples are organized into these categories, each with a landing page:

- **Artifacts**: Creating, logging, managing, and downloading artifacts and managing their metadata.
- **Logging**: Logging metrics, hyperparameters, and tables.
- **Registry**: Creating and managing registries and working with collections.
- **Runs**: Initializing and managing runs.
