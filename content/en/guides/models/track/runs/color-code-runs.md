---
description: Color code runs based on metrics 
menu:
  default:
    identifier: color-code-runs
    parent: what-are-runs
title: Color code runs
---

Color code runs to visually distinguish them based on metrics you log to W&B. Identify trends and patterns based highest, lowest, or latest values of a metric across your training and evaluation runs. The metric, y value, and the number of buckets you select determine how the runs are grouped and colored in a plot. 

Navigate to your workspace's settings page to configure the metric-based colors for runs:

1. Navigate to your W&B project.
2. Select the **Workspace** tab from the project sidebar.
3. Click on the **Settings** icon (⚙️) in the top right corner of the workspace.
4. From the drawer, select **Runs**.
5. Select **Metric-based colors**.
6. From the **Metric** dropdown, select the metric you want to use for color coding.
7. From the **Y value** dropdown, select the y value you want to use for color coding.
8. Select the number of buckets. Minimum is 2, maximum is 8.

The following sections describe how to set the metric and y value, as well as how to customize the buckets used for color coding.

## Configure a metric

The options in your **Metric** dropdown are derived from the key-value pairs [you log to W&B]({{< relref "guides/models/track/runs/color-code-runs/#custom-metrics" >}}) and [default metrics]({{< relref "guides/models/track/runs/color-code-runs/#default-metrics" >}}) provided by W&B.

### Default metrics

* `Relative Time (Process)`: The relative time of the run, measured in seconds since the start of the run.
* `Relative Time (Wall)`: The relative time of the run, measured in seconds since the start of the run, adjusted for wall clock time.
* `Wall Time`: The wall clock time of the run, measured in seconds since the epoch.
* `Step`: The step number of the run, which is typically used to track the progress of training or evaluation.

### Custom metrics

Use custom metrics that you log to W&B in your training or evaluation scripts to color code runs. Custom metrics are logged as key-value pairs, where the key is the name of the metric and the value is the metric value.

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

Within the **Metric** dropdown, both `"acc"` and `"loss"` are available options.

## Configure a Y value

You can choose from the following options:

- **Latest**: Determine color based on Y value at last logged step for each line.
- **Max**: Color based on highest Y value logged against the metric.
- **Min**: Color based on lowest Y value logged against the metric.

## Customize buckets

Buckets are ranges of values that W&B uses to categorize runs based on the metric you select. Buckets are evenly distributed across the range of values for the specified metric and each bucket is assigned a unique color. Runs that fall within that bucket's range are displayed in that color. 

Consider the following configuration:

{{< img src="/images/track/color-coding-runs.png" alt="" >}}

**Metric** is set to `"Accuracy"` (abbreviated as `"acc"`). **Y value** is set to `"Max"`, meaning the maximum value of the accuracy metric is used to determine the color coding for the runs. 

Six buckets are defined for the metric, with each bucket representing a range of accuracy values. Within the **Buckets** section, the following range of buckets are defined:

- Bucket 1: (Min - 0.5798)
- Bucket 2: (0.5798 - 0.6629)
- Bucket 3: (0.6629 - 0.7460)
- Bucket 4: (0.7460 - 0.8292)
- Bucket 5: (0.8292 - 0.9123)
- Bucket 6: (0.9123 - Max)

In the line plot below, the run with the highest accuracy (0.957) is colored in a deep purple (Bucket 6), while the run with the lowest accuracy (0.7993) is colored in a lighter purple (Bucket 4). The other runs are colored based on their accuracy values, with the color gradient indicating their relative performance. 

{{< img src="/images/track/color-code-runs-plot.png" alt="Line plot with color coded runs based on a specific metric" >}}