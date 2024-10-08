---
displayed_sidebar: default
title: Use point aggregation
---

Use point aggregation methods within your line plots for improved data visualization accuracy and performance. There are two types of point aggregation modes: [full fidelity](#full-fidelity) and random sampling](#random-sampling). W&B uses full fidelity mode by default.

## Full fidelity

When you use full fidelity mode, W&B breaks the x-axis into dynamic buckets based on the number of data points. It then calculates the minimum, maximum, and average values within each bucket while rendering a point aggregation for the line plot. 

There are three main advantages to using full fidelity mode for point aggregation:

* Preserve outliers and spikes: Unlike random sampling, full fidelity mode preserves outliers and spikes in the data. 
* Create shaded areas: use the W&B App to interactively decide whether you want to shade extreme values outliers, such as the minimum and maximum.
* Zoom in on data points: Zoom in on data points without losing detail due to downsampling. 


Full Fidelity provides an accurate and detailed representation of the dataset while maintaining performance for charts with extremely large data.



### Manage shaded areas

Manage shaded areas within your line plots to highlight minimum and maximum points in your data. For example, you can shade the area between the minimum and maximum values to emphasize the range of the data. 

The proceeding image shows a line plot in purple with maximum values shaded in gray.
![](/images/app_ui/shaded-areas.png)

There are three ways to manage render shaded areas within your line plots:

* **Never**: The min/max values are not displayed as a shaded area. Only show the aggregated line across the x-axis bucket.
* **On hover**: The shaded area for min/max values appears dynamically when you hover over the chart. This option keeps the view uncluttered while allowing you to inspect ranges interactively.
* **Always**: The min/max shaded area is consistently displayed for every bucket in the chart, helping you visualize the full range of values at all times. This can introduce visual noise if there are many runs visualized in the chart.


By default, the minimum and maximum values are not displayed as shaded areas. To enable one of the shaded area options, follow these steps:

1. Navigate to your W&B project
2. Select on the **Workspace** icon on the left tab
3. Select the gear icon on the top right corner of the screen next to the left of the **Add panels** button.
4. From the UI slider that appears, select **Line plots**
5. Within the **Point aggregation** section, choose **On over** or **Always** from the **Show min/max values as a shaded area** dropdown menu.


### Zoom in on data points

Analyze specific regions of the dataset without missing critical points like extreme values or spikes. When you zoom in on a line plot, W&B adjusts the buckets sizes used to calculate the minimum, maximum, and average values within each bucket. 

![](/images/app_ui/zoom_in.gif)


W&B divides the x-axis is dynamically into 1000 buckets by default. For each bucket, W&B calculates the following values:

- **Minimum**: The lowest value in that bucket.
- **Maximum**: The highest value in that bucket.
- **Average**: The mean value of all points in that bucket.

:::info
W&B does not aggregate data points if you zoom in on a plot with less than 1000 points.
:::

To zoom in on a line plot, follow these steps:

1. Navigate to your W&B project
2. Select on the **Workspace** icon on the left tab
3. Optionally add a line plot panel to your workspace or navigate to an existing line plot panel.
4. Click and drag to select a specific region to zoom in on.

<!-- BEFORE -->



 
## Random sampling

Random sampling point aggregation method uses 1500 randomly sampled points to render line plots. Random sampling is useful for performance reasons when you have a large number of data points. 

However, because random sampling samples non-deterministically, this method can sometimes exclude important outliers or spikes in the data and therefore reduce data accuracy.

### Enable random sampling
By default, W&B uses full fidelity mode for point aggregation. To enable random sampling, follow these steps:

1. Navigate to your W&B project
2. Select on the **Workspace** icon on the left tab
3. Select the gear icon on the top right corner of the screen next to the left of the **Add panels** button.
4. From the UI slider that appears, select **Line plots**
5. Choose **Random sampling** from the **Point aggregation** section

:::info Line plot grouping and expressions
When you use Line Plot Grouping, W&B applies the following based on the mode selected:

- **Non-windowed sampling (grouping)**: Aligns points across runs on the x-axis. The average is taken if multiple points share the same x-value; otherwise, they appear as discrete points.
- **Windowed sampling (grouping and expressions)**: Divides the x-axis either into 250 buckets or the number of points in the longest line (whichever is smaller). Points within each bucket are averaged.
- **Full fidelity (grouping and expressions)**: Similar to non-windowed sampling, but fetches up to 500 points per run to balance performance and detail.
:::


### Access run history

To access the complete history of metrics logged during a run, you can use the [W&B Run API](../../../../../ref/python/public-api/run.md). The following example demonstrates how to retrieve and process the loss values from a specific run:

```python
# Initialize the W&B API
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# Retrieve the history of the 'Loss' metric
history = run.scan_history(keys=["Loss"])

# Extract the loss values from the history
losses = [row["Loss"] for row in history]
``` 