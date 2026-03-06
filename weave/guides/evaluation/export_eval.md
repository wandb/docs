---
title: "Export evaluation data"
description: "Programmatically export evaluation results using the Evaluation REST API."
---

Teams that run evaluations in W&B Weave often need evaluation results outside of the Weave UI. Common use cases include:

- Pulling metrics into spreadsheets or notebooks for custom analysis and visualization.
- Feeding evaluation results into CI/CD pipelines to gate deployments.
- Sharing results with stakeholders who don't have W&B seats, through BI tools like Looker or internal dashboards.
- Building automated reporting pipelines that aggregate scores across projects.

The [v2 Evaluation REST API](https://trace.wandb.ai/docs) surfaces focused evaluation concepts: evaluation runs, predictions, scores, and scorers. The result is richer, more structured output with typed scorer statistics and resolved dataset inputs compared to the general-purpose Calls API.

## API endpoints used

The snippets on this page use the following endpoints from the [v2 Evaluation REST API](https://trace.wandb.ai/docs):

- `GET /v2/{entity}/{project}/evaluation_runs`: Lists evaluation runs in a project, with optional filters by evaluation reference, model reference, or run ID.
- `GET /v2/{entity}/{project}/evaluation_runs/{evaluation_run_id}`: Reads a single evaluation run to retrieve its model, evaluation reference, status, timestamps, and summary.
- `POST /v2/{entity}/{project}/eval_results/query`: Retrieves grouped evaluation result rows for one or more evaluations. Returns per-row trials with model output, scores, and optionally resolved dataset row inputs. Also returns aggregated scorer statistics when requested.
- `GET /v2/{entity}/{project}/predictions/{prediction_id}`: Reads an individual prediction with its inputs, output, and model reference.

Authentication uses HTTP Basic with `api` as the username and your W&B API key as the password.

## Prerequisites

- Python 3.7 or later.
- The `requests` library. Install it with `pip install requests`.
- A W&B API key, set as the `WANDB_API_KEY` environment variable. Get your key at [wandb.ai/settings](https://wandb.ai/settings).

## Set up authentication

```python
import json
import os

import requests

TRACE_BASE = "https://trace.wandb.ai"
AUTH = ("api", os.environ["WANDB_API_KEY"])

entity = "my-team"
project = "my-project"
```

## List evaluation runs

Retrieve recent evaluation runs in a project and list details for each run, such as ID and status.

```python
resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/evaluation_runs",
    auth=AUTH,
)
runs = [json.loads(line) for line in resp.text.strip().splitlines()]

for run in runs:
    print(run["evaluation_run_id"], run.get("status"))
```

## Read a single evaluation run

Retrieve details for a specific evaluation run, including its model, evaluation reference, status, and timestamps.

```python
eval_run_id = "<evaluation-run-id>"

resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/evaluation_runs/{eval_run_id}",
    auth=AUTH,
)
eval_run = resp.json()
print(eval_run["evaluation_run_id"], eval_run.get("status"), eval_run.get("model"))
```

## Get predictions and scores

Use the `eval_results/query` endpoint to retrieve per-row results for an evaluation run. Each row includes the resolved dataset inputs, model output, and individual scorer results. Set `include_rows`, `include_raw_data_rows`, and `resolve_row_refs` to get the full per-row detail.

```python
eval_run_id = "<evaluation-run-id>"

resp = requests.post(
    f"{TRACE_BASE}/v2/{entity}/{project}/eval_results/query",
    json={
        "evaluation_run_ids": [eval_run_id],
        "include_rows": True,
        "include_raw_data_rows": True,
        "resolve_row_refs": True,
    },
    auth=AUTH,
)
results = resp.json()

for row in results["rows"]:
    inputs = row.get("raw_data_row")
    for ev in row.get("evaluations", []):
        for trial in ev.get("trials", []):
            output = trial.get("model_output")
            scores = trial.get("scores", {})
            print("Input:", inputs)
            print("Output:", output)
            print("Scores:", scores)
```

## Get aggregated scores

The same `eval_results/query` endpoint can also return aggregated scorer statistics instead of per-row data. Set `include_summary` to get summary-level metrics like pass rates for binary scorers and means for continuous scorers.

```python
resp = requests.post(
    f"{TRACE_BASE}/v2/{entity}/{project}/eval_results/query",
    json={
        "evaluation_run_ids": [eval_run_id],
        "include_summary": True,
        "include_rows": False,
    },
    auth=AUTH,
)
results = resp.json()

for ev in results["summary"]["evaluations"]:
    for stat in ev["scorer_stats"]:
        print(stat["scorer_key"], stat.get("value_type"), stat.get("pass_rate") or stat.get("numeric_mean"))
```

## Read a single prediction

Retrieve the full details of an individual prediction, including its inputs, output, and model reference.

```python
prediction_id = "<predict-call-id>"

resp = requests.get(
    f"{TRACE_BASE}/v2/{entity}/{project}/predictions/{prediction_id}",
    auth=AUTH,
)
prediction = resp.json()
print(prediction)
```

## How to use row digests

Each result row from the `eval_results/query` endpoint includes a `row_digest`, a content hash that uniquely identifies a specific input in the evaluation dataset based on its contents, not its position. Row digests are useful for:

- **Cross-evaluation comparison**: When you run two different models against the same dataset, rows with the same digest represent the same input. You can join on `row_digest` to compare how different models performed on the exact same task.
- **Deduplication**: If the same task appears in multiple evaluation suites, the digest lets you identify it.
- **Reproducibility**: The digest is content-addressable, so if someone modifies a dataset row (changes the instruction text, rubric, or other fields), it gets a new digest. You can verify whether two evaluation runs used identical inputs or slightly different versions.

