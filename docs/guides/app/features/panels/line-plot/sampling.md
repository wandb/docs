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
1. Navigate to your workspace settings.
2. Select the Runs tab.
3. Under Point aggregation method, choose Full fidelity.


_Note: By default, workspaces have **Random sampling** selected. Switching to Full fidelity applies the chart setting per user._

:::info Line Plot Grouping or Expressions
When applying Line Plot Grouping or using Expressions with multiple runs with possibly not-aligned x axis values, bucketing is used to downsample the points.  The x-axis is divided into 200 evenly sized segments and then within each segments all points for a given metric are averaged. When grouping or using expressions to combine metrics, this average inside a segment is used as the value of the metric.
:::