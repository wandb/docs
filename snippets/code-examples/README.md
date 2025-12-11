# W&B SDK code examples

This directory contains metadata for code examples synced from the [docs-code-eval](https://github.com/wandb/docs-code-eval) repository. These examples serve dual purposes:

1. **Ground truth for LLM evaluation**: For benchmarking a code generation model's capability to generate W&B code examples.
2. **SDK coding cheat sheet reference**: To help readers and LLMs write W&B code.

## Files

- **`llm_evaluation_tasks.csv`**: Metadata about each example (category, difficulty, description)

- **`snippets/en/_incxludes/code-examplees`**: The actual code examples.

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
2. Create MDX snippet wrappers from ground truth examples in `snippets/en/_includes/code-examples/`.
3. Copy the task metadata CSV to `snippets/code-examples/`.
4. Use the MDX snippets and CSV metadata to generate the cheat sheet pages at `models/ref/sdk-coding-cheat-sheet/` and the landing page at `models/ref/sdk-coding-cheat-sheet.mdx`.
5. Clean up the temporary clone.

The changes are ready to commit, push the branch, and create a PR.

## Manual updates

To update individual examples:

1. Make changes in the `wandb/docs-code-eval` repository first.
2. Run the sync script in the `wandb/docs` repository to sync the code examples.
3. Commit the updated files, push the branch, and create a PR.

**Do not edit the MDX snippets directly in the docs repo** (`snippets/en/_includes/code-examples/`) because they will be overwritten on the next sync.

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

During sync, these are converted to MDX snippets that wrap the code in a Python code block.

Examples are organized into these categories, each with a landing page:

- **Artifacts**: Creating, logging, managing, and downloading artifacts and managing their metadata.
- **Logging**: Logging metrics, hyperparameters, and tables.
- **Registry**: Creating and managing registries and working with collections.
- **Runs**: Initializing and managing runs.
