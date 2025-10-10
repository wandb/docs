---
menu:
  default:
    identifier: intro
    parent: w-b-app-ui-reference
title: Custom charts
weight: 3
url: guides/app/features/custom-charts
cascade:
- url: guides/app/features/custom-charts/:filename
---

Custom charts are visualizations you create explicitly. Use custom charts to visualize complex data relationships and have greater control over the appearance and behavior of the chart.

To create a custom chart, you can either:

* [Build a chart in the W&B App](#create-a-custom-chart-in-the-wb-app) by pulling in data from your runs using a [GraphQL](https://graphql.org) query.
* Log a `wandb.Table` and call the `wandb.plot()` function from your script.

{{% alert title="Default charts" %}}
Default charts are generated visualizations based on the data logged from your script with `wandb.Run.log()`. For example, consider the following code snippet:

```python
import wandb
import math

with wandb.init() as run:
  offset = random.random()
  for run_step in range(20):
    run.log({
        "acc": math.log(1 + random.random() + run_step) + offset,
        "val_acc": math.log(1 + random.random() + run_step) + offset * random.random(),
    })
```

This code logs two metrics `acc` and `val_acc` at each training step (each `wandb.Run.log()` call). Within the W&B App, the line plots for `acc` and `val_acc` are automatically generated and appear in the `Charts` panel of the **Workspace** tab.
{{% /alert %}}


## Create a custom chart in the W&B App

To create a custom chart in the W&B App, you must first log data to one or more runs. This data can come from key-value pairs logged with `wandb.Run.log()`, or more complex data structures like `wandb.Table`. Once you log data, you can create a custom chart by pulling in the data using a [GraphQL](https://graphql.org) query.


Suppose you want to create a line plot that shows the accuracy as a function of training step. To do this, you first log accuracy values to a run using the following code snippet:

```python
import wandb
import math

with wandb.init() as run:
  offset = random.random()
  for run_step in range(20):
    run.log({
        "acc": math.log(1 + random.random() + run_step) + offset,
        "val_acc": math.log(1 + random.random() + run_step) + offset * random.random(),
    })
```

Each call to `wandb.Run.log()` creates a new entry in the run's history. The `step` value is automatically tracked by W&B and increments with each log call.

Once the code snippet completes, navigate to your project's workspace:

<!-- 
In the W&B App, navigate to the Workspace tab. Within the `Charts` panel, two line plots appear: one for `acc` and one for `val_acc`. The x-axis is the `step` value (automatically tracked by W&B), and the y-axis is the metric value you provide with `wandb.Run.log()`. -->

<!-- Suppose you want to create a line plot of the accuracy values you logged (`acc`) vs step. To do this: -->

1. Click the **Add panels** button in the top right corner, then select **Custom chart**.
2. Select **Line plot** from the list of chart types.
3. Within the [query editor](#query-editor), select `history` as the [data source](#query-data-sources). Next, select `acc` and type in `_step` as keys. 
4. Within the chart editor, select `_step` for the **X** field and `acc` for the **Y** field. Optionally, set a title for the chart. Your settings should look similar to the following:
{{< img src="/images/app_ui/custom-charts-query-example.png" alt="Custom line plot settings" max-width="90%" >}}


{{% alert %}}
Note that only the keys you specify in the query editor are available in the chart editor (below the query editor). If you do not see the expected keys, go back to the query editor and ensure you have selected the correct data source and specified the correct keys.
{{% /alert %}}

Your custom line plot should now appear in the panel, showing the accuracy values over training steps.

{{< img src="/images/app_ui/manual-line-plot.png" alt="Custom line plot example" max-width="90%" >}}

### Query data sources

When you create a custom chart, you can pull in data from your runs using a [GraphQL](https://graphql.org) query. The data you can query comes from:

* `config`: Initial settings of your experiment (your independent variables). This includes any named fields you’ve logged as keys to `wandb.Run.config` at the start of your training. For example: wandb.Run.config.learning_rate = 0.0001
* `summary`: A single-value metrics that summarize a run. It's populated by the final value of metrics logged with `wandb.Run.log()` or by directly updating the run.summary object. Think of it as a key-value store for your run's final results.
* `history`: Time series data logged with `wandb.Run.log()`. Each call to `log()` creates a new row in the history table. This is ideal for tracking metrics that change over time, like training and validation loss.
* `summaryTable`: A table of summary metrics. It's populated by logging a `wandb.Table` to the `summary` field. This is useful for logging metrics that are best represented in a tabular format, like confusion matrices or classification reports.
* `historyTable`: A table of time series data. It's populated by logging a `wandb.Table` to the `history` field. This is useful for logging complex metrics that change over time, like per-epoch evaluation metrics.

To recap, `summary` and `history` are the general locations for your run data, while `summaryTable` and `historyTable` are the specific query types needed to access tabular data stored in those respective locations.


### Query editor

Within the query editor, you can define a [GraphQL](https://graphql.org) query to pull in the data from available [data sources](#query-data-sources). The query editor consists of dropdowns and text fields that allow you to construct the query without needing to write raw GraphQL. You can include any combination of available data sources, depending on the data you want to visualize.

{{% alert %}}
Pay close attention to the data sources. For example, if you want to visualize time series data logged with `wandb.Run.log()`, you must select the `history` data source. If you select `summary`, you will not see the expected data because `summary` contains only single-value metrics that summarize a run, not time series data.
{{% /alert %}}

The `keys` argument acts as a filter to specify exactly which pieces of data you want to retrieve from a larger data object such as `summary`. The sub-fields are dictionary-like or key-value pairs.


The general structure of the query is as follows:

```graphql
query {
    runSets: (runSets: "${runSets}", limit: 500 ) {
        id: 
        name: 
        summary:
            (keys: [""])
        history:
            (keys: [""])
        summaryTable:
            (keys: [""])
        historyTable:
            (keys: [""])
    }
}
```

Here's a breakdown of the components:

* `runSets`: Top-level object, representing the set of runs you are currently viewing or have filtered in the UI.
* `summary(...)`, `history(...)`, `summaryTable(...)`, `historyTable(...)`: This tells the query to fetch data from the respective objects of each run.
* `keys: [""]`: An array of strings where each string is the name (or key) of the metric or object you want to retrieve.

## Log charts from a script

You can programmatically create a custom chart from your script by logging a `wandb.Table` of the data you want to visualize, then calling `wandb.plot.*` to create the chart. 

For example, consider the following code snippet:

```python
import wandb
import math

with wandb.init() as run:
    offset = random.random()

    # Set up data to log in custom charts
    data = []
    for i in range(100):
    data.append([i, random.random() + math.log(1 + i) + offset + random.random()])

    # Create a table with the columns to plot
    table = wandb.Table(data=data, columns=["step", "height"])

    # Use the table to populate various custom charts
    line_plot = wandb.plot.line(table, x='step', y='height', title='Line Plot')
    histogram = wandb.plot.histogram(table, value='height', title='Histogram')
    scatter = wandb.plot.scatter(table, x='step', y='height', title='Scatter Plot')

    # Log custom tables, which will show up in customizable charts in the UI
    run.log({'line_1': line_plot, 
                'histogram_1': histogram, 
                'scatter_1': scatter})
```

Within the W&B app, navigate to the **Workspace** tab. Within the `Custom Charts` panel, there are three charts with the following titles: **Scatter Plot**, **Histogram**, and **Line Plot**. These correspond to the three charts created in the script above. The x-axis and y-axis are set to the columns specified in the `wandb.plot.*` function calls (`height` and `step`). 

The following image shows the three custom charts created from the script:

{{< img src="/images/app_ui/custom-charts-script-plots.png" alt="Custom charts from script" max-width="90%" >}}

### Built-in chart types

W&B has a number of built-in chart presets that you can log directly from your script. These include line plots, scatter plots, bar charts, histograms, PR curves, and ROC curves. The following tabs show how to log each type of chart.

{{< tabpane text=true >}}
{{% tab header="Line plot" value="line-plot" %}}

  `wandb.plot.line()`

  Log a custom line plot—a list of connected and ordered points (x,y) on arbitrary axes x and y.

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=["x", "y"])
    run.log(
        {
            "my_custom_plot_id": wandb.plot.line(
                table, "x", "y", title="Custom Y vs X Line Plot"
            )
        }
    )
  ```

  A line plot logs curves on any two dimensions. If you plot two lists of values against each other, the number of values in the lists must match exactly (for example, each point must have an x and a y).

  {{< img src="/images/app_ui/line_plot.png" alt="Custom line plot" >}}

  [See an example report](https://wandb.ai/wandb/plots/reports/Custom-Line-Plots--VmlldzoyNjk5NTA) or [try an example Google Colab notebook](https://tiny.cc/custom-charts).

{{% /tab %}}

{{% tab header="Scatter plot" value="scatter-plot" %}}

  `wandb.plot.scatter()`

  Log a custom scatter plot—a list of points (x, y) on a pair of arbitrary axes x and y.

  ```python
  with wandb.init() as run:
    data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
    table = wandb.Table(data=data, columns=["class_x", "class_y"])
    run.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
  ```

  You can use this to log scatter points on any two dimensions. Note that if you're plotting two lists of values against each other, the number of values in the lists must match exactly (for example, each point must have an x and a y).

  {{< img src="/images/app_ui/demo_scatter_plot.png" alt="Scatter plot" >}}

  [See an example report](https://wandb.ai/wandb/plots/reports/Custom-Scatter-Plots--VmlldzoyNjk5NDQ) or [try an example Google Colab notebook](https://tiny.cc/custom-charts).

{{% /tab %}}

{{% tab header="Bar chart" value="bar-chart" %}}

  `wandb.plot.bar()`

  Log a custom bar chart—a list of labeled values as bars—natively in a few lines:

  ```python
  with wandb.init() as run:
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns=["label", "value"])
    run.log(
        {
            "my_bar_chart_id": wandb.plot.bar(
                table, "label", "value", title="Custom Bar Chart"
            )
        }
    )
  ```

  You can use this to log arbitrary bar charts. Note that the number of labels and values in the lists must match exactly (for example, each data point must have both).

{{< img src="/images/app_ui/demo_bar_plot.png" alt="Demo bar plot" >}}

  [See an example report](https://wandb.ai/wandb/plots/reports/Custom-Bar-Charts--VmlldzoyNzExNzk) or [try an example Google Colab notebook](https://tiny.cc/custom-charts).
{{% /tab %}}

{{% tab header="Histogram" value="histogram" %}}

  `wandb.plot.histogram()`

  Log a custom histogram—sort list of values into bins by count/frequency of occurrence—natively in a few lines. Let's say I have a list of prediction confidence scores (`scores`) and want to visualize their distribution:

  ```python
  with wandb.init() as run:
    data = [[s] for s in scores]
    table = wandb.Table(data=data, columns=["scores"])
    run.log({"my_histogram": wandb.plot.histogram(table, "scores", title=None)})
  ```

  You can use this to log arbitrary histograms. Note that `data` is a list of lists, intended to support a 2D array of rows and columns.

  {{< img src="/images/app_ui/demo_custom_chart_histogram.png" alt="Custom histogram" >}}

  [See an example report](https://wandb.ai/wandb/plots/reports/Custom-Histograms--VmlldzoyNzE0NzM) or [try an example Google Colab notebook](https://tiny.cc/custom-charts).

{{% /tab %}}

{{% tab header="PR curve" value="pr-curve" %}}

  `wandb.plot.pr_curve()`

  Create a [Precision-Recall curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) in one line:

  ```python
  with wandb.init() as run:
    plot = wandb.plot.pr_curve(ground_truth, predictions, labels=None, classes_to_plot=None)

    run.log({"pr": plot})
  ```

  You can log this whenever your code has access to:

  * a model's predicted scores (`predictions`) on a set of examples
  * the corresponding ground truth labels (`ground_truth`) for those examples
  * (optionally) a list of the labels/class names (`labels=["cat", "dog", "bird"...]` if label index 0 means cat, 1 = dog, 2 = bird, etc.)
  * (optionally) a subset (still in list format) of the labels to visualize in the plot

  {{< img src="/images/app_ui/demo_average_precision_lines.png" alt="Precision-recall curves" >}}


  [See an example report](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY) or [try an example Google Colab notebook](https://colab.research.google.com/drive/1mS8ogA3LcZWOXchfJoMrboW3opY1A8BY?usp=sharing).

{{% /tab %}}

{{% tab header="ROC curve" value="roc-curve" %}}

  `wandb.plot.roc_curve()`

  Create an [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve) in one line:

  ```python
  with wandb.init() as run:
    # ground_truth is a list of true labels, predictions is a list of predicted scores
    ground_truth = [0, 1, 0, 1, 0, 1]
    predictions = [0.1, 0.4, 0.35, 0.8, 0.7, 0.9]

    # Create the ROC curve plot
    # labels is an optional list of class names, classes_to_plot is an optional subset of those labels to visualize
    plot = wandb.plot.roc_curve(
        ground_truth, predictions, labels=None, classes_to_plot=None
    )

    run.log({"roc": plot})
  ```

  You can log this whenever your code has access to:

  * a model's predicted scores (`predictions`) on a set of examples
  * the corresponding ground truth labels (`ground_truth`) for those examples
  * (optionally) a list of the labels/ class names (`labels=["cat", "dog", "bird"...]` if label index 0 means cat, 1 = dog, 2 = bird, etc.)
  * (optionally) a subset (still in list format) of these labels to visualize on the plot

  {{< img src="/images/app_ui/demo_custom_chart_roc_curve.png" alt="ROC curve" >}}

  [See an example report](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE) or [try an example Google Colab notebook](https://colab.research.google.com/drive/1_RMppCqsA8XInV_jhJz32NCZG6Z5t1RO?usp=sharing).

{{% /tab %}}
{{< /tabpane >}}


<!-- Commnented out because I couldn't find this option anymore -->
<!-- ### Saving chart presets

Apply any changes to a specific visualization panel with the button at the bottom of the modal. Alternatively, you can save the Vega spec to use elsewhere in your project. To save the reusable chart definition, click **Save as** at the top of the Vega editor and give your preset a name. -->


## Additional resources

<!-- * Watch a [walkthrough video](https://www.youtube.com/watch?v=3-N9OV6bkSM). -->
* Log your first custom chart with this [Colab notebook](https://tiny.cc/custom-charts) or read its companion [W&B Machine Learning Visualization IDE](https://wandb.ai/wandb/posts/reports/The-W-B-Machine-Learning-Visualization-IDE--VmlldzoyNjk3Nzg) report.
* Try the [Logging custom visualizations with W&B using Keras and Sklearn Colab](https://colab.research.google.com/drive/1g-gNGokPWM2Qbc8p1Gofud0_5AoZdoSD?usp=sharing) notebook.
* Read the [Visualizing NLP Attention Based Models](https://wandb.ai/kylegoyette/gradientsandtranslation2/reports/Visualizing-NLP-Attention-Based-Models-Using-Custom-Charts--VmlldzoyNjg2MjM) report.
* Explore the [Visualizing The Effect of Attention on Gradient Flow](https://wandb.ai/kylegoyette/gradientsandtranslation/reports/Visualizing-The-Effect-of-Attention-on-Gradient-Flow-Using-Custom-Charts--VmlldzoyNjg1NDg) report.
* Read the [Logging arbitrary curves](https://wandb.ai/stacey/presets/reports/Logging-Arbitrary-Curves--VmlldzoyNzQyMzA) report.