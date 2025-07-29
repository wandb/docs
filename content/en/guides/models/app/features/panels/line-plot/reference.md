---
menu:
  default:
    identifier: reference
    parent: line-plot
title: Line plot reference
weight: 10
---

## X-Axis

{{< img src="/images/app_ui/reference_x_axis.png" alt="Selecting X-Axis" >}}

You can set the x-axis of a line plot to any value that you have logged with W&B.log as long as it's always logged as a number.

## Y-Axis variables

You can set the y-axis variables to any value you have logged with wandb.log as long as you were logging numbers, arrays of numbers or a histogram of numbers. If you logged more than 1500 points for a variable, W&B samples down to 1500 points.

{{% alert %}}
You can change the color of your y axis lines by changing the color of the run in the runs table.
{{% /alert %}}

## X range and Y range

You can change the maximum and minimum values of X and Y for the plot.

X range default is from the smallest value of your x-axis to the largest.

Y range default is from the smallest value of your metrics and zero to the largest value of your metrics.

## Max runs/groups

By default you will only plot 10 runs or groups of runs. The runs will be taken from the top of your runs table or run set, so if you sort your runs table or run set you can change the runs that are shown.

{{% alert %}}
A workspace is limited to displaying a maximum of 1000 runs, regardless of its configuration.
{{% /alert %}}

## Legend

You can control the legend of your chart to show for any run any config value that you logged and meta data from the runs such as the created at time or the user who created the run.

Example:

`${run:displayName} - ${config:dropout}` will make the legend name for each run something like `royal-sweep - 0.5` where `royal-sweep` is the run name and `0.5` is the config parameter named `dropout`.

You can set value inside`[[ ]]` to display point specific values in the crosshair when hovering over a chart. For example `\[\[ $x: $y ($original) ]]` would display something like "2: 3 (2.9)"

Supported values inside `[[ ]]` are as follows:

| Value         | Meaning                                    |
| ------------  | ------------------------------------------ |
| `${x}`        | X value                                    |
| `${y}`        | Y value (Including smoothing adjustment)   |
| `${original}` | Y value not including smoothing adjustment |
| `${mean}`     | Mean of grouped runs                       |
| `${stddev}`   | Standard Deviation of grouped runs         |
| `${min}`      | Min of grouped runs                        |
| `${max}`      | Max of grouped runs                        |
| `${percent}`  | Percent of total (for stacked area charts) |

## Grouping

You can aggregate all of the runs by turning on grouping, or group over an individual variable. You can also turn on grouping by grouping inside the table and the groups will automatically populate into the graph.

## Smoothing

You can set the [smoothing coefficient]({{< relref "/support/kb-articles/formula_smoothing_algorithm.md" >}}) to be between 0 and 1 where 0 is no smoothing and 1 is maximum smoothing.


## Ignore outliers

Rescale the plot to exclude outliers from the default plot min and max scale. The setting's impact on the plot depends on the plot's sampling mode.

- For plots that use [random sampling mode]({{< relref "sampling.md#random-sampling" >}}), when you enable **Ignore outliers**, only points from 5% to 95% are shown. When outliers are shown, they are not formatted differently from other points.
- For plots that use [full fidelity mode]({{< relref "sampling.md#full-fidelity" >}}), all points are always shown, condensed down to the last value in each bucket. When **Ignore outliers** is enabled, the minimum and maximum bounds of each bucket are shaded. Otherwise, no area is shaded.

## Expression

Use expressions in charts to create derived metrics by performing mathematical operations on your logged values. For example, you could calculate error rates, ratios, percentages, and other custom metrics directly in your charts. Expressions are computed on-the-fly in the W&B App.

### Supported operators
Expressions support the following operators:

- Basic arithmetic: `+`, `-`, `*`, `/`
- Modulo (remainder): `%`
- Exponent (power): `**`

### Limitations
- Expressions currently work only when plotting a single metric.
- Expressions must use the exact metric name that was logged. For example, `accuracy` rather than `run.accuracy`.
- Expressions do not support functions like `log()` or`sqrt()`.

### Examples
These examples illustrate some ways to use expressions in charts. For even more examples, refer to a [Colab notebook demo of chart expressions](https://raw.githubusercontent.com/wandb/examples/0151359bdede604088074b9817788fb87607f636/colabs/chart_expressions/chart_expressions_demo.ipynb). <!-- Temporarily points to the raw file inside the upstream PR -->

#### Error rate from accuracy
Convert a 0.95 accuracy into a 0.05 error rate:

```math
1 - accuracy
```

![Example chart showing an expression to derive error rate from accuracy](/images/app_ui/expressions_accuracy_error_rate.png)

#### Percentage from decimal value
Convert the decimal value `0.95` to `95%`:

```math
accuracy * 100
```

![Example chart showing an expression to derive a percentage from a decinal value](/images/app_ui/expressions_accuracy_percentage.png)

#### Calculate ratios of logged values
Detect overfitting by comparing training and validation losses:

```math
loss / val_loss
```

In this expression, a value greater than `1` indicates overfitting.

![Example chart showing an expression to calculate the ratio of two logged values](/images/app_ui/expressions_loss_overfitting.png)

#### Visualize small values more easily
Use an expression to scale a small value:

```math
learning_rate * 1000
```

Or a metric with a wide range of values:
```math
loss * 10000
```

![Example chart showing an expression that scales a small value](/images/app_ui/expressions_learning_rate_scaled.png)

#### Convert from one unit to another
Convert memory bytes used to a percentage of total memory:

```math
memory_used / memory_total * 100
```

![Example chart showing an expression that scales a small value](/images/app_ui/expressions_memory_usage.png)

#### Visualize training efficiency
Plot accuracy gained per epoch:

```math
accuracy / epoch
```

![Example chart that visualizes accuracy per epoch](/images/app_ui/expressions_accuracy_per_epoch.png).

#### Approximate F1 score
F1 score is a machine learning evaluation metric that combines precision and recall scores. Use an expression to approximate the F1 score:

```math
2 * (precision * recall) / (precision + recall)
```

![Example chart that calculates the approximate F1 score, given precision and recall](/images/app_ui/expressions_f1_approximation.png).

## Plot style

Select a style for your line plot.

**Line plot:**

{{< img src="/images/app_ui/plot_style_line_plot.png" alt="Line plot style" >}}

**Area plot:**

{{< img src="/images/app_ui/plot_style_area_plot.png" alt="Area plot style" >}}

**Percentage area plot:**

{{< img src="/images/app_ui/plot_style_percentage_plot.png" alt="Percentage plot style" >}}

### Interactive Tutorial

For a hands-on tutorial with runnable examples, check out our [W&B Expressions Demo Colab notebook](https://colab.research.google.com/github/wandb/docs/blob/main/wandb_expressions_demo.ipynb) that demonstrates various expression use cases with real code you can run and modify.
