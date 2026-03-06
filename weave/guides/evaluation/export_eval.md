---
title: "Programmatically export evaluations"
description: "Export evaluation data from W&B Weave using the v2 Evaluation REST API"
---


Teams that run evaluations in W&B Weave often need evaluation results outside of the Weave UI. Common use cases include:

- Pulling metrics into spreadsheets or notebooks for custom analysis and visualization.
- Feeding evaluation results into CI/CD pipelines to gate deployments.
- Sharing results with stakeholders who don't have W&B seats, through BI tools like Looker or internal dashboards.
- Building automated reporting pipelines that aggregate scores across projects.

The [`weave_export_evals.py`](https://github.com/wandb/docs/blob/main/scripts/weave_export_evals.py) script demonstrates extracting evaluation data from a Weave project using the [v2 Evaluation REST API](https://trace.wandb.ai/docs). Unlike the general-purpose Calls API, these endpoints surface focused evaluation concepts: evaluation runs, predictions, scores, and scorers. The result is richer, more structured output with typed scorer statistics and resolved dataset inputs. The script requires only Python and the `requests` library.

### What the script exports

The script operates in two modes: **list** and **export**.

**List mode** queries a project for recent evaluation runs and displays a summary of each one, including the model, evaluation name, status, and timestamps. It tries the v2 `evaluation_runs` endpoint first and falls back to the general-purpose Calls API to search for Call objects with `Evaluation.evaluate` in their Op name when the v2 endpoint returns no data.

**Export mode** retrieves the full details of a single evaluation run and writes them to JSON or CSV. The export includes:

- **Evaluation run details**: The run ID, display name, evaluation reference, model reference, status, and timestamps.
- **Scorer statistics**: Aggregated stats for each scorer dimension, including value type (binary or continuous), pass rate and pass counts for binary scorers, and numeric mean for continuous scorers.
- **Per-prediction data**: For each row in the evaluation dataset, the export includes:
  - The predict-and-score Call ID and row digest.
  - Resolved dataset row inputs (the actual data, not just a reference).
  - The model's output.
  - All scorer results, broken down by scorer and sub-field.
  - Model latency and token usage when available.

### How to use row digests

Each prediction in the export includes a `row_digest`, a content hash that uniquely identifies a specific input in the evaluation dataset based on its contents, not its position. Row digests are useful for:

- **Cross-evaluation comparison**: When you run two different models against the same dataset, rows with the same digest represent the same input. You can join on `row_digest` to compare how different models performed on the exact same task.
- **Deduplication**: If the same task appears in multiple evaluation suites, the digest lets you identify it.
- **Reproducibility**: The digest is content-addressable, so if someone modifies a dataset row (changes the instruction text, rubric, or other fields), it gets a new digest. You can verify whether two evaluation runs used identical inputs or slightly different versions.

### Prerequisites

- Python 3.7 or later.
- The `requests` library. Install it with `pip install requests`.
- A W&B API key, set as the `WANDB_API_KEY` environment variable. Get your key at [wandb.ai/settings](https://wandb.ai/settings).

### Usage

**List recent evaluation runs in a project:**

```bash
python weave_export_evals.py --entity my-team --project my-project
```

**Export a specific evaluation run to JSON (by UUID or list index):**

```bash
python weave_export_evals.py --entity my-team --project my-project --eval-run-id <id>
python weave_export_evals.py --entity my-team --project my-project --eval-run-id 0
```

**Export to a JSON file:**

```bash
python weave_export_evals.py --entity my-team --project my-project --eval-run-id <id> -o results.json
```

**Export to CSV:**

```bash
python weave_export_evals.py --entity my-team --project my-project --eval-run-id <id>
```

### Script options

| Flag | Description | Default |
|---|---|---|
| `--entity` | W&B entity (team or username). Required. | |
| `--project` | W&B project name. Required. | |
| `--eval-run-id` | Evaluation run ID (UUID) or list index (for example, `0`, `1`) to export. Omit to list runs. | |
| `--format` | Output format: `json` or `csv`. | `json` |
| `-o`, `--output` | Output file path. JSON defaults to stdout. CSV defaults to `eval_<id>.csv`. | |
| `--limit` | Maximum number of evaluation runs to list. | `20` |

### API endpoints used

The script uses the following endpoints from the [v2 Evaluation REST API](https://trace.wandb.ai/docs):

- `GET /v2/{entity}/{project}/evaluation_runs`: Lists evaluation runs in a project, with optional filters by evaluation reference, model reference, or run ID.
- `GET /v2/{entity}/{project}/evaluation_runs/{evaluation_run_id}`: Reads a single evaluation run to retrieve its model, evaluation reference, status, timestamps, and summary.
- `GET /v2/{entity}/{project}/predictions/{prediction_id}`: Reads an individual prediction with its inputs, output, and model reference.
- `GET /v2/{entity}/{project}/scorers/{object_id}/versions/{digest}`: Reads a scorer definition including its name, description, and score Op reference.
- `POST /v2/{entity}/{project}/eval_results/query`: Retrieves grouped evaluation result rows for one or more evaluations. Returns per-row trials with model output, scores, and optionally resolved dataset row inputs. Also returns aggregated scorer statistics when requested.

Authentication uses HTTP Basic with `api` as the username and your W&B API key as the password.

### Output structure

**JSON output** contains three top-level keys:

- `evaluation_run`: An object with `evaluation_run_id`, `evaluation_ref`, `evaluation_name`, `model_ref`, `model_name`, `display_name`, `status`, `started_at`, `finished_at`, and `total_rows`.
- `scorer_stats`: An array of scorer statistics, each with `scorer` (the scorer name and path), `value_type` (`binary` or `continuous`), `trial_count`, and either `pass_rate`/`pass_true_count`/`pass_known_count` for binary scorers or `numeric_mean`/`numeric_count` for continuous scorers.
- `predictions`: An array of per-prediction objects, each with `predict_and_score_call_id`, `predict_call_id`, `row_digest`, `inputs` (the resolved dataset row), `output` (the model's response), `scores` (nested by scorer and sub-field), `model_latency_seconds`, and `total_tokens`.

**CSV output** contains one row per prediction with columns for `predict_and_score_call_id`, `row_digest`, `inputs` (JSON string), `output` (JSON string), and one column per flattened score path (for example, `score.check_concrete_fields.city_match`, `score.check_value_fields.avg_temp_f_err`).

