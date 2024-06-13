---
description: Rewind
displayed_sidebar: default
---

# Rewind Runs
:::caution
The ability to rewind a run is in private preview. Contact W&B Support at support@wandb.com to request access to this feature.
:::

Use `resume_from` with [`wandb.init()`](https://docs.wandb.ai/ref/python/init) to "rewind" a run’s history to a specific step. When you rewind a run, W&B resets the state of the run to the specified step, preserving the original data and maintaining a consistent run ID. This feature allows for correction or modification of the run history without losing the original data, and enables new data logging from that point. Summary metrics are recomputed based on the newly logged history.

:::info
Rewinding a run requires monotonically increasing steps. You can not use non-monotonic steps defined with [`define_metric()`](https://docs.wandb.ai/ref/python/run#define_metric) to set a resume point because it would disrupt the essential chronological order of run history and system metrics.
:::

:::info
Rewinding a run requires the [`wandb`](https://pypi.org/project/wandb/) SDK version >= 0.17.1.
:::

### History and Config Management

- **History truncation**: The history is truncated to the rewind point, allowing new data logging.
- **Summary metrics**: Recomputed based on the newly logged history.
- **Configuration preservation**: Original configurations are preserved and can be merged with new configurations.

### Run Management

- **Run archiving**: Original runs are archived and accessible from the [**`Run Overview`**](https://docs.wandb.ai/guides/app/pages/run-page#overview-tab).
- **Artifact inheritance**: New runs inherit artifacts from the original run.
- **Artifact association**: Artifacts are associated with the latest version of the rewound run.
- **Immutable run IDs**: Introduced for consistent rewinding from a precise state.
- **Copy immutable run ID**: A button to copy the immutable run ID for improved run management.

### Rewind and Forking Compatibility

Rewind compliments the [**`Forking`**](https://docs.wandb.ai/guides/runs/forking) feature by providing more flexibility in managing and experimenting with your runs. While Forking allows you to create a new branch off a run at a specific point to try different parameters or models, Rewinding a run allows you to correct or modify the run history itself.

### Rewind a Run

To rewind a run, use the `resume_from` argument in `wandb.init()` and specify the run name and the step from which you want to rewind:

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

#### Navigating to a Run Archive

After a run has been rewound, you can easily explore the archived resumptions through the user interface. Follow these steps to navigate the run archive:

1. **Access the Overview Tab:**
   - Navigate to the [**Overview tab**](https://docs.wandb.ai/guides/app/pages/run-page#overview-tab) on the run's page. This tab provides a comprehensive view of the run's details and history.

2. **Locate the Forked From Field:**
   - Within the Overview tab, find the `Forked From` field. This field captures the history of the resumptions.
   - The `Forked From` field includes a link to the source run, allowing you to trace back to the original run and understand the entire rewind history.

By using the `Forked From` field, you can effortlessly navigate the tree of archived resumptions and gain insights into the sequence and origin of each rewind. 

### Fork from a Rewound Run

To fork from a rewound run, use the [**`fork_from`**](https://docs.wandb.ai/guides/runs/forking) argument in `wandb.init()` and specify the source run ID and the step from the source run to fork from:

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

### Unsupported Functionality
- **Log rewind**: Logs are reset in the new run segment.
- **System metrics rewind**: Only new system metrics after the rewind point are logged.
- **Artifact association with specific run segments**: Artifacts are associated with the latest run segment, not the segment that produced them.

