---
description: Forking a W&B run
displayed_sidebar: default
---

# Fork Runs

## Forking a Run
 The `fork_from` initialization parameter streamlines the creation of new experiment runs by "forking" from existing runs. It automatically creates a new run using the `run ID` and `step` of the source run, facilitating the creation of run chains and easy tracking of their lineage. This feature enables efficient exploration of different parameters or models from a specific point in an experiment without data loss impact on the original run.


### Starting a Forked Run

To fork a run, use the `fork_from` argument in `wandb.init()` and specify the source `run ID` and the `step` from the source run to fork from:

```python
import wandb

# Initialize a run to be forked later
original_run = wandb.init(project="your_project_name", entity="your_entity_name")
# ... perform training or logging ...
original_run.finish()

# Fork the run from a specific step
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{original_run.id}?_step=200",
)
```


### Continuing from a Forked Run
After initializing a forked run, you can continue logging to the new run. You can log the same metrics for continuity and introduce new metrics:

```python
import wandb
import math

# Initialize the first run and log some metrics
run1 = wandb.init("your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# Fork from the first run at a specific step and log the metric starting from step 200
run2 = wandb.init(
    "your_project_name", entity="your_entity_name", fork_from=f"{run1.id}?_step=200"
)

# Continue logging in the new run
# For the first few steps, log the metric as is from run1
# After step 250, start logging the spikey pattern
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i})  # Continue logging from run1 without spikes
    else:
        # Introduce the spikey behavior starting from step 250
        subtle_spike = i + (2 * math.sin(i / 3.0))  # Apply a subtle spikey pattern
        run2.log({"metric": subtle_spike})
    # Additionally log the new metric at all steps
    run2.log({"additional_metric": i * 1.1})
run2.finish()
```

## Arguments for `wandb.init()` related to Forking Runs

When initializing a new run with the intention of forking from an existing run, `wandb.init()` accepts specific arguments for forking a run:


| Argument     | Description |
|--------------|-------------|
| `fork_from`  | (str, optional) A unique `run.id` identifier of the run you want to fork. Append `?_step=` to the `run.id` with the step to fork from. |


:::info
Forking a run requires monotonically increasing steps. Non-monotonic steps defined with `define_metric()` cannot be used to set a fork point, as it would disrupt the essential chronological order of run history and system metrics.
:::

