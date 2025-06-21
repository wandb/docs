---
title: line_series()
object_type: python_sdk_custom_charts
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/plot/line_series.py >}}




### <kbd>function</kbd> `line_series`

```python
line_series(
    xs: 'Iterable[Iterable[Any]] | Iterable[Any]',
    ys: 'Iterable[Iterable[Any]]',
    keys: 'Iterable[str] | None' = None,
    title: 'str' = '',
    xname: 'str' = 'x',
    split_table: 'bool' = False
) → CustomChart
```

Constructs a line series chart. 



**Args:**
 
 - `xs`:  Sequence of x values. If a singular  array is provided, all y values are plotted against that x array. If  an array of arrays is provided, each y value is plotted against the  corresponding x array. 
 - `ys`:  Sequence of y values, where each iterable represents  a separate line series. 
 - `keys`:  Sequence of keys for labeling each line series. If  not provided, keys will be automatically generated as "line_1",  "line_2", etc. 
 - `title`:  Title of the chart. 
 - `xname`:  Label for the x-axis. 
 - `split_table`:  Whether the table should be split into a separate section  in the W&B UI. If `True`, the table will be displayed in a section named  "Custom Chart Tables". Default is `False`. 



**Returns:**
 
 - `CustomChart`:  A custom chart object that can be logged to W&B. To log the  chart, pass it to `wandb.log()`. 



**Examples:**
 Logging a single x array where all y series are plotted against the same x values: 

```python
import wandb

# Initialize W&B run
with wandb.init(project="line_series_example") as run:
    # x values shared across all y series
    xs = list(range(10))

    # Multiple y series to plot
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # Generate and log the line series chart
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         title="title",
         xname="step",
    )
    run.log({"line-series-single-x": line_series_chart})
``` 

In this example, a single `xs` series (shared x-values) is used for all `ys` series. This results in each y-series being plotted against the same x-values (0-9). 

Logging multiple x arrays where each y series is plotted against its corresponding x array: 

```python
import wandb

# Initialize W&B run
with wandb.init(project="line_series_example") as run:
    # Separate x values for each y series
    xs = [
         [i for i in range(10)],  # x for first series
         [2 * i for i in range(10)],  # x for second series (stretched)
         [3 * i for i in range(10)],  # x for third series (stretched more)
    ]

    # Corresponding y series
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # Generate and log the line series chart
    line_series_chart = wandb.plot.line_series(
         xs, ys, title="Multiple X Arrays Example", xname="Step"
    )
    run.log({"line-series-multiple-x": line_series_chart})
``` 

In this example, each y series is plotted against its own unique x series. This allows for more flexibility when the x values are not uniform across the data series. 

Customizing line labels using `keys`: 

```python
import wandb

# Initialize W&B run
with wandb.init(project="line_series_example") as run:
    xs = list(range(10))  # Single x array
    ys = [
         [i for i in range(10)],  # y = x
         [i**2 for i in range(10)],  # y = x^2
         [i**3 for i in range(10)],  # y = x^3
    ]

    # Custom labels for each line
    keys = ["Linear", "Quadratic", "Cubic"]

    # Generate and log the line series chart
    line_series_chart = wandb.plot.line_series(
         xs,
         ys,
         keys=keys,  # Custom keys (line labels)
         title="Custom Line Labels Example",
         xname="Step",
    )
    run.log({"line-series-custom-keys": line_series_chart})
``` 

This example shows how to provide custom labels for the lines using the `keys` argument. The keys will appear in the legend as "Linear", "Quadratic", and "Cubic". 
