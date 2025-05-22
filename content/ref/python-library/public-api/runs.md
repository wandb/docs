---
title: runs
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/runs.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Runs. 

This module provides classes for interacting with W&B runs and their associated data. 



**Example:**
 ```python
from wandb.apis.public import Api

# Initialize API
api = Api()

# Get runs matching filters
runs = api.runs(
     path="entity/project", filters={"state": "finished", "config.batch_size": 32}
)

# Access run data
for run in runs:
     print(f"Run: {run.name}")
     print(f"Config: {run.config}")
     print(f"Metrics: {run.summary}")

     # Get history with pandas
     history_df = run.history(keys=["loss", "accuracy"], pandas=True)

     # Work with artifacts
     for artifact in run.logged_artifacts():
         print(f"Artifact: {artifact.name}")
``` 



**Note:**

> This module is part of the W&B Public API and provides read/write access to run data. For logging new runs, use the wandb.init() function from the main wandb package. 

