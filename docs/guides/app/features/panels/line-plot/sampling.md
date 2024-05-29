---
displayed_sidebar: default
---

# W&B Line Plots Point Aggregation

## Overview
W&B Line Plots feature selection of point aggregation methods designed to improve data visualization accuracy and performance: **Random sampling** and **Full fidelity**.

## Random sampling (default)
For performance reasons, when over 1500 points were chosen for a line plot metric, this point aggregation method returns 1500 randomly sampled points. Each metric is sampled separately, and only steps where the metric was actually logged were considered. Because it is sampling non-deterministically, this method sometimes excluded important outliers or spikes.


#### Example Code: Accessing Run History

To access the complete history of metrics logged during a run, you can use the W&B API. The following example demonstrates how to retrieve and process the loss values from a specific run:

```python
# Initialize the W&B API
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# Retrieve the history of the 'Loss' metric
history = run.scan_history(keys=["Loss"])

# Extract the loss values from the history
losses = [row["Loss"] for row in history]
```

## Full fidelity

When grouping or using expressions with multiple runs with possibly not-aligned x axis values, bucketing is used to downsample the points.  The x-axis is divided into 200 evenly sized segments and then within each segments all points for a given metric are averaged. When grouping or using expressions to combine metrics, this average inside a segment is used as the value of the metric.

The new full fidelity point aggregation method replaces random sampling with an averaging approach that maintains the integrity of critical visual insights, such as outliers and spikes. This mode guarantees the inclusion of minimum and maximum values within each bucket on your chart, allowing for high-detail zoom capabilities.

**Key Benefits:**
1. Accurate Data Representation: Ensures all critical outlier spikes are displayed.
2. High-Density Visualization: Maintains full data resolution beyond the 1,500 point limit.
3. Enhanced Zoom: Users can zoom into data points without losing detail due to downsampling.

**Enabling Full fidelity mode:**
1. Navigate to your workspace settings or panel settings.
2. Select the Runs tab.
3. Under Point aggregation method, choose Full fidelity.


_Note: By default, workspaces have **Random sampling** selected. Switching to **Full fidelity** applies the chart setting per user._

:::info Line Plot Grouping or Expressions
When using Line Plot Grouping or Expressions with runs that have non-aligned x-axis values, we downsample points using bucketing. The x-axis is divided into 200 segments, and points within each segment are averaged. These averages represent the metric values when grouping or combining metrics.
:::

:::caution Active feature development
Applying Grouping or Expressions will revert to Random sampling instead of Full fidelity. We are actively working on achieving full feature parity with the Run Plots settings for Full fidelity mode, including enabling Grouping and Custom Expressions, while also optimizing performance. For now, panels with grouping or expressions will use Random sampling. This feature is available early because it was highly requested and provided value to users, even though improvements are still ongoing. Please reachout to support@wandb.com if you have any issues. 
:::