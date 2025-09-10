---
title: Public API
module: wandb.apis.public
weight: 6
no_list: true
---

The W&B Public API provides programmatic access to query, export, and update data stored in W&B. Use this API for post-hoc analysis, data export, and programmatic management of runs, artifacts, and sweeps. While the main SDK handles real-time logging during training, the Public API enables you to retrieve historical data, update metadata, manage artifacts, and perform analysis on completed experiments. Access is provided through the main `Api` class which serves as the entry point to all functionality.

> Use the Public API for querying and managing data after it has been logged to W&B.

## Available Components

| Component | Description |
|-----------|-------------|
| [`Api`](./api/) | Main entry point for the Public API. Query runs, projects, and artifacts across your organization. |
| [`Runs`](./runs/) | Access and manage individual training runs, including history, logs, and metrics. |
| [`Artifacts`](./artifacts/) | Query and download model artifacts, datasets, and other versioned files. |
| [`Sweeps`](./sweeps/) | Access hyperparameter sweep data and analyze optimization results. |
| [`Projects`](./projects/) | Manage projects and access project-level metadata and settings. |
| [`Reports`](./reports/) | Programmatically access and manage W&B Reports. |
| [`Teams`](./teams/) | Query team information and manage team-level resources. |
| [`Users`](./users/) | Access user profiles and user-specific data. |
| [`Files`](./files/) | Download and manage files associated with runs. |
| [`History`](./history/) | Access detailed time-series metrics logged during training. |
| [`Automations`](./automations/) | Manage automated workflows and actions. |
| [`Integrations`](./integrations/) | Configure and manage third-party integrations. |

## Common Use Cases

### Data Export and Analysis
- Export run history as DataFrames for analysis in Jupyter notebooks
- Download metrics for custom visualization or reporting
- Aggregate results across multiple experiments

### Post-Hoc Updates
- Update run metadata after completion
- Add tags or notes to completed experiments
- Modify run configurations or summaries

### Artifact Management
- Query artifacts by version or alias
- Download model checkpoints programmatically
- Track artifact lineage and dependencies

### Sweep Analysis
- Access sweep results and best performing runs
- Export hyperparameter search results
- Analyze parameter importance

## Authentication

The Public API uses the same authentication mechanism as the Python SDK. You can authenticate in several ways:

Use the `WANDB_API_KEY` environment variable to set your API key:

```bash
export WANDB_API_KEY=your_api_key
```

Pass the API key directly when initializing the `Api` class:

```python
api = Api(api_key="your_api_key")
```

Or use `wandb.login()` to authenticate the current session:
```python
import wandb

wandb.login()
api = Api()
```


## Example Usage

```python
from wandb.apis.public import Api

# Initialize the API client
api = Api()

# Query runs with filters
runs = api.runs(
    path="entity/project",
    filters={"state": "finished", "config.learning_rate": {"$gte": 0.001}}
)

# Analyze run metrics
for run in runs:
    print(f"Run: {run.name}")
    print(f"Final accuracy: {run.summary.get('accuracy')}")
    
    # Get detailed history
    history = run.history(keys=["loss", "accuracy"])
    
    # Update run metadata
    run.tags.append("reviewed")
    run.update()

# Access artifacts
artifact = api.artifact("entity/project/model:v1")
artifact_dir = artifact.download()

# Query sweep results
sweep = api.sweep("entity/project/sweep_id")
best_run = sweep.best_run()
print(f"Best parameters: {best_run.config}")

# Export data as DataFrame
import pandas as pd
runs_df = pd.DataFrame([
    {**run.config, **run.summary} 
    for run in runs
])
```

