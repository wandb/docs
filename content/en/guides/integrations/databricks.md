---
description: How to integrate W&B with Databricks.
menu:
  default:
    identifier: databricks
    parent: integrations
title: Databricks
weight: 50
---

W&B integrates with [Databricks](https://www.databricks.com/) by customizing the W&B Jupyter notebook experience in the Databricks environment.

## Configure Databricks

1. Install wandb in the cluster

    Navigate to your cluster configuration, choose your cluster, click **Libraries**. Click **Install New**, choose **PyPI**, and add the package `wandb`.

2. Set up authentication

    To authenticate your W&B account you can add a Databricks secret which your notebooks can query.

    ```bash
    # install databricks cli
    pip install databricks-cli

    # Generate a token from databricks UI
    databricks configure --token

    # Create a scope with one of the two commands (depending if you have security features enabled on databricks):
    # with security add-on
    databricks secrets create-scope --scope wandb
    # without security add-on
    databricks secrets create-scope --scope wandb --initial-manage-principal users

    # Add your api_key from: https://app.wandb.ai/authorize
    databricks secrets put --scope wandb --key api_key
    ```

## Examples

### Simple example

```python
import os
import wandb

api_key = dbutils.secrets.get("wandb", "api_key")
wandb.login(key=api_key)

with wandb.init() as run:
    run.log({"foo": 1})
```

### Sweeps

Setup required (temporary) for notebooks attempting to use wandb.sweep() or wandb.agent():

```python
import os

# These will not be necessary in the future
os.environ["WANDB_ENTITY"] = "my-entity"
os.environ["WANDB_PROJECT"] = "my-project-that-exists"
```