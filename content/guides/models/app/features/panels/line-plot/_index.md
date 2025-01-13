---
description: Visualize metrics, customize axes, and compare multiple lines on a plot
url: guides/app/features/panels/line-plot
menu:
  default:
    identifier: intro_line_plot
    parent: panels
cascade:
- url: guides/app/features/panels/line-plot/:filename
title: Line plots
weight: 10
---

Line plots show up by default when you plot metrics over time with **wandb.log()**. Customize with chart settings to compare multiple lines on the same plot, calculate custom axes, and rename labels.

{{< img src="/images/app_ui/line_plot_example.png" alt="" >}}

## Edit line panel settings
 
Hover your mouse over the panel you want to edit its settings for. Select the pencil icon that appears. Within the modal that appears, select a tab to edit the Data, Grouping, Chart, Expressions, or Legend settings.

### Data 

Select the **Data** tab to edit the x-axis, y-axis, smoothing filter, point aggregation and more. The proceeding describes some of the options you can edit:

{{% alert %}}
If you'd like to use a custom x-axis, make sure it's logged in the same call to `wandb.log()` that you use to log the y-axis.
{{% /alert %}} 

* **X axis**: By default the x-axis is set to **Step**. You can change the x-axis to **Relative Time**, or select a custom axis based on values you log with W&B.
  * **Relative Time (Wall)** is clock time since the process started, so if you started a run and resumed it a day later and logged something that would be plotted a 24hrs.
  * **Relative Time (Process)** is time inside the running process, so if you started a run and ran for 10 seconds and resumed a day later that point would be plotted at 10s
  * **Wall Time** is minutes elapsed since the start of the first run on the graph
  * **Step** increments by default each time `wandb.log()` is called, and is supposed to reflect the number of training steps you've logged from your model
* **Y axes**: Select y-axes from the logged values, including metrics and hyperparameters that change over time.
* **Min, max, and log scale**: Minimum, maximum, and log scale settings for x axis and y axis in line plots
* **Smoothing**: Change the smoothing on the line plot.
* **Outliers**: Rescale to exclude outliers from the default plot min and max scale
* **Max runs to show**: Show more lines on the line plot at once by increasing this number, which defaults to 10 runs. You'll see the message "Showing first 10 runs" on the top of the chart if there are more than 10 runs available but the chart is constraining the number visible.
* **Chart type**: Change between a line plot, an area plot, and a percentage area plot

{{% alert %}}
Pick **multiple y-axes** in the line plot settings to compare different metrics on the same chart, like accuracy and validation accuracy for example.
{{% /alert %}}

### Grouping

Select the **Grouping** tab to use group by methods to organize your data.


* **Group by**: Select a column, and all the runs with the same value in that column will be grouped together.
* **Agg**: Aggregation— the value of the line on the graph. The options are mean, median, min, and max of the group.

### Chart
Select the Chart tab to edit the plot's title, axis titles, legend, and more.

* **Title**: Add a custom title for line plot, which shows up at the top of the chart
* **X-Axis title**: Add a custom title for the x-axis of the line plot, which shows up in the lower right corner of the chart.
* **Y-Axis title**: Add a custom title for the y-axis of the line plot, which shows up in the upper left corner of the chart.
* **Show legend**: Toggle legend on or off
* **Font size**: Change the font size of the chart title, x-axis title, and y-axis title
* **Legend position**: Change the position of the legend on the chart


### Legend


* **Legend**: Select field that you want to see in the legend of the plot for each line. You could, for example, show the name of the run and the learning rate.
* **Legend template**: Fully customizable, this powerful template allows you to specify exactly what text and variables you want to show up in the template at the top of the line plot as well as the legend that appears when you hover your mouse over the plot.

### Expressions

* **Y Axis Expressions**: Add calculated metrics to your graph. You can use any of the logged metrics as well as configuration values like hyperparameters to calculate custom lines.
* **X Axis Expressions**: Rescale the x-axis to use calculated values using custom expressions. Useful variables include\*\*_step\*\* for the default x-axis, and the syntax for referencing summary values is `${summary:value}`

## Visualize average values on a plot

If you have several different experiments and you'd like to see the average of their values on a plot, you can use the Grouping feature in the table. Click "Group" above the run table and select "All" to show averaged values in your graphs.

Here is what the graph looks like before averaging:

{{< img src="/images/app_ui/demo_precision_lines.png" alt="" >}}

The proceeding image shows a graph that represents average values across runs using grouped lines.

{{< img src="/images/app_ui/demo_average_precision_lines.png" alt="" >}}

## Visualize NaN value on a plot

You can also plot `NaN` values including PyTorch tensors on a line plot with `wandb.log`. For example:

```python
wandb.log({"test": [..., float("nan"), ...]})
```

{{< img src="/images/app_ui/visualize_nan.png" alt="" >}}

## Compare two metrics on one chart

{{< img src="/images/app_ui/visualization_add.gif" alt="" >}}

1. Select the **Add panels** button in the top right corner of the page.
2. From the left panel that appears, expand the Evaluation dropdown.
3. Select **Run comparer**


## Change the color of the line plots

Sometimes the default color of runs is not helpful for comparison. To help overcome this, wandb provides two instances with which one can manually change the colors.

{{< tabpane text=true >}}
{{% tab header="From the run table" value="run_table" %}}

  Each run is given a random color by default upon initialization.

  {{< img src="/images/app_ui/line_plots_run_table_random_colors.png" alt="Random colors given to runs" >}}

  Upon clicking any of the colors, a color palette appears from which we can manually choose the color we want.

  {{< img src="/images/app_ui/line_plots_run_table_color_palette.png" alt="The color palette" >}}

{{% /tab %}}

{{% tab header="From the chart legend settings" value="legend_settings" %}}

1. Hover your mouse over the panel you want to edit its settings for.
2. Select the pencil icon that appears.
3. Choose the **Legend** tab.

{{< img src="/images/app_ui/plot_style_line_plot_legend.png" alt="" >}}

{{% /tab %}}
{{< /tabpane >}}

## Visualize on different x axes

If you'd like to see the absolute time that an experiment has taken, or see what day an experiment ran, you can switch the x axis. Here's an example of switching from steps to relative time and then to wall time.

{{< img src="/images/app_ui/howto_use_relative_time_or_wall_time.gif" alt="" >}}

## Area plots

In the line plot settings, in the advanced tab, click on different plot styles to get an area plot or a percentage area plot.

{{< img src="/images/app_ui/line_plots_area_plots.gif" alt="" >}}

## Zoom

Click and drag a rectangle to zoom vertically and horizontally at the same time. This changes the x-axis and y-axis zoom.

{{< img src="/images/app_ui/line_plots_zoom.gif" alt="" >}}

## Hide chart legend

Turn off the legend in the line plot with this simple toggle:

{{< img src="/images/app_ui/demo_hide_legend.gif" alt="" >}}