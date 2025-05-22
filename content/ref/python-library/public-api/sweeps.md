---
title: sweeps
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/sweeps.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Sweeps. 

This module provides classes for interacting with W&B hyperparameter optimization sweeps. 



**Example:**
 ```python
from wandb.apis.public import Api

# Initialize API
api = Api()

# Get a specific sweep
sweep = api.sweep("entity/project/sweep_id")

# Access sweep properties
print(f"Sweep: {sweep.name}")
print(f"State: {sweep.state}")
print(f"Best Loss: {sweep.best_loss}")

# Get best performing run
best_run = sweep.best_run()
print(f"Best Run: {best_run.name}")
print(f"Metrics: {best_run.summary}")
``` 



**Note:**

> This module is part of the W&B Public API and provides read-only access to sweep data. For creating and controlling sweeps, use the wandb.sweep() and wandb.agent() functions from the main wandb package. 

