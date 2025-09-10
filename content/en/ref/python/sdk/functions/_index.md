---
title: Global Functions
module: 
weight: 1
---

The W&B Python SDK provides a set of global functions that serve as the primary entry points for interacting with the platform. These functions are called directly on the `wandb` module and form the foundation of most W&B workflows.

## Overview

Global functions in W&B are top-level functions that you call directly, such as `wandb.init()` or `wandb.login()`. Unlike methods that belong to specific classes, these functions provide direct access to W&B's core functionality without needing to instantiate objects first.

## Available Functions

| Function | Description |
|----------|-------------|
| [`init()`](./init/) | Start a new run to track and log to W&B. This is typically the first function you'll call in your ML training pipeline. |
| [`login()`](./login/) | Set up W&B login credentials to authenticate your machine with the platform. |
| [`finish()`](./finish/) | Complete a run and upload any remaining data to ensure all information is synced to the server. |
| [`setup()`](./setup/) | Prepare W&B for use in the current process and its children. Useful for multi-process applications. |
| [`teardown()`](./teardown/) | Clean up W&B resources and shut down the backend process. |
| [`sweep()`](./sweep/) | Initialize a hyperparameter sweep to search for optimal model configurations. |
| [`agent()`](./agent/) | Create a sweep agent to run hyperparameter optimization experiments. |
| [`controller()`](./controller/) | Manage and control sweep agents and their execution. |
| [`restore()`](./restore/) | Restore a previous run or experiment state for resuming work. |

## Getting Started

The most common workflow begins with authenticating with W&B, initializing a run, and logging values (such as accuracy and loss) from your training loop:

```python
import wandb

# Authenticate with W&B
wandb.login()

config = {
   "learning_rate": 0.01,
   "epochs": 10,
}

# Project where logs
project = "my-awesome-project"

# Start a new run
with wandb.init(project=project, config=config) as run:
   # Your training code here...
   
   # Log values to W&B
   run.log({"accuracy": acc, "loss": loss})
```

The previous code example demonstrates the following key concepts:

- **Authentication**: Required to sync data with the W&B platform
- **Configuration**: Store hyperparameters and metadata for your experiments
- **[Runs](/guides/runs)**: The fundamental unit of tracking in W&B. Log metrics, artifacts, and more to runs.

For detailed information about each function, click on the function names above to view their complete documentation, including parameters, examples, and usage patterns.
