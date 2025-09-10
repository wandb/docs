---
title: Python SDK
module: 
weight: 2
---
The W&B Python SDK, accessible at `wandb`, enables you to train and fine-tune models, and manage models from experimentation to production. 

> After performing your training and fine-tuning operations with this SDK, you can use [the Public API]({{< relref "/ref/python/sdk/public-api" >}}) to query and analyze the data that was logged, and [the Reports and Workspaces API]({{< relref "/ref/python/wandb_workspaces" >}}) to generate a web-publishable [report]({{< relref "/guides/core/reports" >}}) summarizing your work.

## Installation and setup

### Sign up and create an API key

To authenticate your machine with W&B, you must first generate an API key at https://wandb.ai/authorize.

### Install and import packages

Install the W&B library.

```
pip install wandb
```

### Import W&B Python SDK:

```python
import wandb

# Specify your team entity
entity = "<team_entity>"

# Project that the run is recorded to
project = "my-awesome-project"

with wandb.init(entity=entity, project=project) as run:
   run.log({"accuracy": .90, "loss": .10})
````

## Optional extras 

The `wandb` package supports optional extras for specific features. To install these, use square brackets with the package name. 

> In **zsh** (default on macOS), you need to escape the brackets or use quotes, e.g. `pip install "wandb[media]"`

**Available extras:**
- `wandb[media]` - Media logging support (installs bokeh, moviepy, pillow, plotly)
- `wandb[workspaces]` - Workspaces functionality (installs wandb-workspaces)
- `wandb[sweeps]` - Hyperparameter sweeps support
- `wandb[launch]` - W&B Launch support
- `wandb[models]` - Model management features
- `wandb[aws]` - AWS integrations
- `wandb[azure]` - Azure integrations
- `wandb[gcp]` - Google Cloud Platform integrations
- `wandb[kubeflow]` - Kubeflow integrations
- `wandb[importers]` - Data importers
- `wandb[perf]` - Performance monitoring tools

**Installation examples:**
```bash
# Using quotes (works in all shells)
pip install "wandb[media]"
pip install "wandb[workspaces,media]"

# Escaping brackets (for zsh/bash)
pip install wandb\[media\]

# Install all common extras
pip install "wandb[media,workspaces,sweeps,launch]"
```

For detailed documentation on each extra, see the [Optional Extras Reference]({{< relref "./extras" >}}).
