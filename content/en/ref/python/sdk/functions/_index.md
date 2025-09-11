---
title: Global Functions
module: 
weight: 1
no_list: true
---

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

## Example

The most common workflow begins with authenticating with W&B, initializing a run, and logging values (such as accuracy and loss) from your training loop. The first steps are to import `wandb` and use the global functions `login()` and `init()`:

```python
import wandb

# Authenticate with W&B
wandb.login()

# Hyperparameters and metadata
config = {
   "learning_rate": 0.01,
   "epochs": 10,
}

# Project that the run is recorded to
project = "my-awesome-project"

# Initialize a new run
with wandb.init(project=project, config=config) as run:
   # Your training code here...
   
   # Log values to W&B
   run.log({"accuracy": 0.9, "loss": 0.1})
```