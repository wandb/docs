---
description: Create semantic legends for charts
menu:
  default:
    identifier: color-code-runs
    parent: what-are-runs
title: Semantic run plot legends
---

Create visually meaningful line plots and plot legends by color-coding your W&B runs based on metrics or configuration parameters. Identify patterns and trends across experiments by coloring runs according to their performance metrics (highest, lowest, or latest values). W&B automatically groups your runs into color-coded buckets based on the values of your selected parameter.

To use metric or configuration-based colors for your runs, you need to configure two settings:

### Turn on key-based colors

1. Navigate to your W&B project.
2. Select the **Workspace** tab from the project sidebar.
3. Click on the **Settings** icon in the top right corner.
4. From the drawer, select **Runs**.
5. In the **Run colors** section, select **Key-based colors**.
6. Configure the following options:
    - From the **Key** dropdown, select the metric you want to use for assigning colors to runs.
    - From the **Y value** dropdown, select the y value you want to use for assigning colors to runs.
    - Set the number of buckets to a value from 2 to 8.

### Choose a color palette

1. Navigate to your W&B project.
2. Select the **Workspace** tab from the project sidebar.
3. Click on the **Settings** icon in the top right corner.
4. From the drawer, select **Runs**.
5. In the **Color palette** section, choose from:
    - **Default**: Standard color palette.
    - **Colorblind-safe (deuteranomaly)**: Optimized for red-green color blindness.
    - **Colorblind-safe (all other forms)**: Optimized for other forms of color vision difference.

The following sections describe how to set the metric and y value and as how to customize the buckets used for assigning colors to runs.


### Example: Key-based coloring with loss metric

In this example plot, runs are colored with a gradient where darker colors represent higher loss values and lighter colors represent lower loss values. The Y value is set to `latest` to use the most recent loss value for each run.
 
{{< img src="/images/app_ui/run-colors-key-based.png" alt="W&B workspace showing runs colored based on their loss values using key-based coloring.">}}

## Set a metric

The metric options in your **Key** dropdown are derived from the key-value pairs [you log to W&B]({{< relref "guides/models/track/runs/color-code-runs/#custom-metrics" >}}) and [default metrics]({{< relref "guides/models/track/runs/color-code-runs/#default-metrics" >}}) defined by W&B.

### Default metrics

* `Relative Time (Process)`: The relative time of the run, measured in seconds since the start of the run.
* `Relative Time (Wall)`: The relative time of the run, measured in seconds since the start of the run, adjusted for wall clock time.
* `Wall Time`: The wall clock time of the run, measured in seconds since the epoch.
* `Step`: The step number of the run, which is typically used to track the progress of training or evaluation.

### Custom metrics

Color runs and create meaningful plot legends based on custom metrics logged by your training or evaluation scripts. Custom metrics are logged as key-value pairs, where the key is the name of the metric and the value is the metric value.

For example, the following code snippet logs accuracy (`"acc"` key) and loss (`"loss"` key) during a training loop:

```python
import wandb
import random

epochs = 10

with wandb.init(project="basic-intro") as run:
  # Block simulates a training loop logging metrics
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # Log metrics from your script to W&B
      run.log({"acc": acc, "loss": loss})
```

Within the **Key** dropdown, both `"acc"` and `"loss"` are available options.

## Set a configuration key

The configuration options in your **Key** dropdown are derived from the key-value pairs you pass to the `config` parameter when you initialize a W&B run. Configuration keys are typically used to log hyperparameters or other settings used in your training or evaluation scripts.

```python
import wandb

config = {
  "learning_rate": 0.01,
  "batch_size": 32,
  "optimizer": "adam"
}

with wandb.init(project="basic-intro", config=config) as run:
  # Your training code here
  pass
```

Within the **Key** dropdown, `"learning_rate"`, `"batch_size"`, and `"optimizer"` are available options.

## Set a y value

You can choose from the following options:

- **Latest**: Determine color based on Y value at last logged step for each line.
- **Max**: Color based on highest Y value logged against the metric.
- **Min**: Color based on lowest Y value logged against the metric.

## Customize buckets

Buckets are ranges of values that W&B uses to categorize runs based on the metric or configuration key you select. Buckets are evenly distributed across the range of values for the specified metric or configuration key and each bucket is assigned a unique color. Runs that fall within that bucket's range are displayed in that color. 

Consider the following:

{{< img src="/images/track/color-coding-runs.png" alt="Color coded runs" >}}

- **Key** is set to `"Accuracy"` (abbreviated as `"acc"`).
- **Y value** is set to `"Max"`

With this configuration, W&B colors each run based on their accuracy values. The colors vary from a light yellow color to a deep color. Lighter colors represent lower accuracy values, while deeper colors represent higher accuracy values.

Six buckets are defined for the metric, with each bucket representing a range of accuracy values. Within the **Buckets** section, the following range of buckets are defined:

- Bucket 1: (Min - 0.7629)
- Bucket 2: (0.7629 - 0.7824)
- Bucket 3: (0.7824 - 0.8019)
- Bucket 4: (0.8019 - 0.8214)
- Bucket 5: (0.8214 - 0.8409)
- Bucket 6: (0.8409 - Max)

In the line plot below, the run with the highest accuracy (0.8232) is colored in a deep purple (Bucket 5), while the run with the lowest accuracy (0.7684) is colored in a light orange (Bucket 2). The other runs are colored based on their accuracy values, with the color gradient indicating their relative performance. 

{{< img src="/images/track/color-code-runs-plot.png" alt="Color coded runs plot" >}}
