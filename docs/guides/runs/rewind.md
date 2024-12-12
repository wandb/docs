---
description: Rewind
title: Rewind a run
---

# Rewind a run
{{% alert color="secondary" %}}
The option to rewind a run is in private preview. Contact W&B Support at support@wandb.com to request access to this feature.

W&B currently does not support:
* **Log rewind**: Logs are reset in the new run segment.
* **System metrics rewind**: W&B logs only new system metrics after the rewind point.
* **Artifact association**: W&B associates artifacts with the source run that produces them.
{{% /alert %}}

{{% alert %}}
* To rewind a run, you must have [W&B Python SDK](https://pypi.org/project/wandb/) version >= `0.17.1`.
* You must use monotonically increasing steps. You can not use non-monotonic steps defined with [`define_metric()`](/ref/python/run#define_metric) because it disrupts the required chronological order of run history and system metrics.
{{% /alert %}}

Rewind a run to correct or modify the history of a run without losing the original data. In addition, when you 
rewind a run, you can log new data from that point in time. W&B recomputes the summary metrics for the run you rewind based on the newly logged history. This means the following behavior:
- **History truncation**: W&B truncates the history to the rewind point, allowing new data logging.
- **Summary metrics**: Recomputed based on the newly logged history.
- **Configuration preservation**: W&B preserves the original configurations and you can merge new configurations.

<!-- #### Manage runs -->
When you rewind a run, W&B resets the state of the run to the specified step, preserving the original data and maintaining a consistent run ID. This means that:

- **Run archiving**: W&B archives the original runs. Runs are accessible from the [**Run Overview**](./intro.md#overview-tab) tab.
- **Artifact association**: Associates artifacts with the run that produce them.
- **Immutable run IDs**: Introduced for consistent forking from a precise state.
- **Copy immutable run ID**: A button to copy the immutable run ID for improved run management.

{{% alert title="Rewind and forking compatibility" %}}
Forking compliments a rewind.

When you fork from a run, W&B creates a new branch off a run at a specific point to try different parameters or models. 

When you  rewind a run, W&B lets you correct or modify the run history itself.
{{% /alert %}}



## Rewind a run

Use `resume_from` with [`wandb.init()`](/ref/python/init) to "rewind" a run’s history to a specific step. Specify the name of the run and the step you want to rewind from:

```python
import wandb
import math

# Initialize the first run and log some metrics
# Replace with your_project_name and your_entity_name!
run1 = wandb.init(project="your_project_name", entity="your_entity_name")
for i in range(300):
    run1.log({"metric": i})
run1.finish()

# Rewind from the first run at a specific step and log the metric starting from step 200
run2 = wandb.init(project="your_project_name", entity="your_entity_name", resume_from=f"{run1.id}?_step=200")

# Continue logging in the new run
# For the first few steps, log the metric as is from run1
# After step 250, start logging the spikey pattern
for i in range(200, 300):
    if i < 250:
        run2.log({"metric": i, "step": i})  # Continue logging from run1 without spikes
    else:
        # Introduce the spikey behavior starting from step 250
        subtle_spike = i + (2 * math.sin(i / 3.0))  # Apply a subtle spikey pattern
        run2.log({"metric": subtle_spike, "step": i})
    # Additionally log the new metric at all steps
    run2.log({"additional_metric": i * 1.1, "step": i})
run2.finish()
```

## View an archived run


After you rewind a run, you can explore archived run with the W&B App UI. Follow these steps to view archived runs:

1. **Access the Overview Tab:** Navigate to the [**Overview tab**](./intro.md#overview-tab) on the run's page. This tab provides a comprehensive view of the run's details and history.
2. **Locate the Forked From field:** Within the **Overview** tab, find the `Forked From` field. This field captures the history of the resumptions. The **Forked From** field includes a link to the source run, allowing you to trace back to the original run and understand the entire rewind history.

By using the `Forked From` field, you can effortlessly navigate the tree of archived resumptions and gain insights into the sequence and origin of each rewind. 

## Fork from a run that you rewind

To fork from a rewound run, use the [**`fork_from`**](/guides/runs/forking) argument in `wandb.init()` and specify the source run ID and the step from the source run to fork from:

```python 
import wandb

# Fork the run from a specific step
forked_run = wandb.init(
    project="your_project_name",
    entity="your_entity_name",
    fork_from=f"{rewind_run.id}?_step=500",
)

# Continue logging in the new run
for i in range(500, 1000):
    forked_run.log({"metric": i*3})
forked_run.finish()
```

