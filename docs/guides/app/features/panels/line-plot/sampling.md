---
displayed_sidebar: default
title: Use point aggregation
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Use point aggregation methods within your line plots for improved data visualization accuracy and performance. There are two types of point aggregation modes: [full fidelity](#full-fidelity) and [random sampling](#random-sampling). W&B uses full fidelity mode by default.

## Full fidelity

When you use full fidelity mode, W&B breaks the x-axis into dynamic buckets based on the number of data points. It then calculates the minimum, maximum, and average values within each bucket while rendering a point aggregation for the line plot. 

There are three main advantages to using full fidelity mode for point aggregation:

* Preserve extreme values and spikes: retain extreme values and spikes in your data 
* Configure how minimum and maximum points render: use the W&B App to interactively decide whether you want to show extreme (min/max) values as a shaded area.
* Explore your data without losing data fidelity: W&B recalculate x-axis bucket sizes when you zoom into specific data points. This helps ensure that you can explore your data without losing accuracy. Caching is used to store previously computed aggregations to help reduce loading times which is particularly useful if you are navigating through large datasets.



### Configure how minimum and maximum points render

Show or hide minimum and maximum values with shaded areas around your line plots.

The proceeding image shows a blue line plot. The light blue shaded area represents the minimum and maximum values for each bucket.

![](/images/app_ui/shaded-areas.png)

There are three ways to render minimum and maximum values in your line plots:

* **Never**: The min/max values are not displayed as a shaded area. Only show the aggregated line across the x-axis bucket.
* **On hover**: The shaded area for min/max values appears dynamically when you hover over the chart. This option keeps the view uncluttered while allowing you to inspect ranges interactively.
* **Always**: The min/max shaded area is consistently displayed for every bucket in the chart, helping you visualize the full range of values at all times. This can introduce visual noise if there are many runs visualized in the chart.

By default, the minimum and maximum values are not displayed as shaded areas. To view one of the shaded area options, follow these steps:


<Tabs
  defaultValue="all_charts"
  values={[
    {label: 'All charts in  a workspace', value: 'all_charts'},
    {label: 'Individual chart in a workspace', value: 'single_chart'},
  ]}>
  <TabItem value="all_charts">

1. Navigate to your W&B project
2. Select on the **Workspace** icon on the left tab
3. Select the gear icon on the top right corner of the screen next to the left of the **Add panels** button.
4. From the UI slider that appears, select **Line plots**
5. Within the **Point aggregation** section, choose **On over** or **Always** from the **Show min/max values as a shaded area** dropdown menu.


  </TabItem>
  <TabItem value="single_chart">

1. Navigate to your W&B project
2. Select on the **Workspace** icon on the left tab
3. Select the line plot panel you want to enable full fidelity mode for
4. Within the modal that appears, select **On hover** or **Always** from the **Show min/max values as a shaded area** dropdown menu.


  </TabItem>
</Tabs>


### Explore your data without losing data fidelity

Analyze specific regions of the dataset without missing critical points like extreme values or spikes. When you zoom in on a line plot, W&B adjusts the buckets sizes used to calculate the minimum, maximum, and average values within each bucket. 


![](/images/app_ui/zoom_in.gif)


W&B divides the x-axis is dynamically into 1000 buckets by default. For each bucket, W&B calculates the following values:

- **Minimum**: The lowest value in that bucket.
- **Maximum**: The highest value in that bucket.
- **Average**: The mean value of all points in that bucket.

W&B plots values in buckets in a way that preserves full data representation and includes extreme values in every plot. When zoomed in to 1,000 points or fewer, full fidelity mode renders every data point without additional aggregation.


To zoom in on a line plot, follow these steps:

1. Navigate to your W&B project
2. Select on the **Workspace** icon on the left tab
3. Optionally add a line plot panel to your workspace or navigate to an existing line plot panel.
4. Click and drag to select a specific region to zoom in on.

:::info Line plot grouping and expressions
When you use Line Plot Grouping, W&B applies the following based on the mode selected:

- **Non-windowed sampling (grouping)**: Aligns points across runs on the x-axis. The average is taken if multiple points share the same x-value; otherwise, they appear as discrete points.
- **Windowed sampling (grouping and expressions)**: Divides the x-axis either into 250 buckets or the number of points in the longest line (whichever is smaller). W&B takes an average of points within each bucket.
- **Full fidelity (grouping and expressions)**: Similar to non-windowed sampling, but fetches up to 500 points per run to balance performance and detail.
:::

 
## Random sampling

Random sampling uses 1500 randomly sampled points to render line plots. Random sampling is useful for performance reasons when you have a large number of data points. 

:::warning
Random sampling samples non-deterministically. This means that random sampling sometimes excludes important outliers or spikes in the data and therefore reduces data accuracy.
:::


### Enable random sampling
By default, W&B uses full fidelity mode. To enable random sampling, follow these steps:


<Tabs
  defaultValue="all_charts"
  values={[
    {label: 'All charts in a workspace', value: 'all_charts'},
    {label: 'Individual chart in a workspace', value: 'single_chart'},
  ]}>
  <TabItem value="all_charts">

1. Navigate to your W&B project
2. Select on the **Workspace** icon on the left tab
3. Select the gear icon on the top right corner of the screen next to the left of the **Add panels** button.
4. From the UI slider that appears, select **Line plots**
5. Choose **Random sampling** from the **Point aggregation** section


  </TabItem>
  <TabItem value="single_chart">

1. Navigate to your W&B project
2. Select on the **Workspace** icon on the left tab
3. Select the line plot panel you want to enable random sampling for
4. Within the modal that appears, select **Random sampling** from the **Point aggregation method** section


  </TabItem>
</Tabs>



### Access non sampled data

You can access the complete history of metrics logged during a run using the [W&B Run API](../../../../../ref/python/public-api/run.md). The following example demonstrates how to retrieve and process the loss values from a specific run:


```python
# Initialize the W&B API
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")

# Retrieve the history of the 'Loss' metric
history = run.scan_history(keys=["Loss"])

# Extract the loss values from the history
losses = [row["Loss"] for row in history]
``` 