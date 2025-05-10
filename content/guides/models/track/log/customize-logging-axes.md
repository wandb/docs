---
menu:
  default:
    identifier: customize-logging-axes
    parent: log-objects-and-media
title: Customize log axes
---

Set a custom x-axis when you log metrics to W&B. By default, W&B logs metrics as *steps*. Each step corresponds to a `wandb.log()` API call. 

For example, suppose you have a `for` loop that loops 10 times (see the following code block). In each `for` loop, you log a metric called `validation_loss` with `wandb.log()`. W&B increments the step number by 1 each time you log that metric. 

```python
import wandb

with wandb.init() as run:
  # range function creates a sequence of numbers from 0 to 9
  for i in range(10):
    log_dict = {
        "validation_loss": 1/(i+1)   
    }
    run.log(log_dict)
```

If you navigate to your project's workspace, you can see that the `validation_loss` metric is plotted against the step x-axis. Each step in the for loop monotonically increases by 1, so the x-axis shows the step numbers 0, 1, 2, ..., 9. (The Python `range()` function creates a sequence of numbers from 0 to 9.)

{{< img src="/images/experiments/standard_axes.png" alt="Line plot panel that uses default x-axis. Each step in monotonically increases by 1." >}}

In certain situations, it makes more sense to log metrics against a different x-axis. For example, you may want a logarithmic x-axis. Use the [`define_metric()`]({{< relref "ref/python/run/#define_metric" >}}) method to define a custom x-axis based on a metric you log.

Specify the metric that you want to appear as the y-axis with the `name` parameter. The `step_metric` parameter specifies the metric you want to use as the x-axis. When you log a metric, specify a value for both the x-axis and the y-axis as key-value pairs in a dictionary. 

Copy and paste the following code snippet to set a custom x-axis metric. Replace the values within `<>` with your own values:

```python
import wandb

custom_step = "<custom_step>"  # custom x-axis
metric_name = "<metric>"  # metric to log against custom x-axis

with wandb.init() as run:
    # define step metric (x-axis) and the metric to log against it (y-axis)
    run.define_metric(name = metric_name, step_metric = custom_step)

    for i in range(10):
        log_dict = {
            custom_step : int,  # Value of x-axis
            metric_name : int,  # Value of y-axis
        }
        run.log(log_dict)
```

As an example, the following code snippet demonstrates how to create a custom x-axis using a metric called `x_axis_squared` and a y-axis called `validation_loss`. The value of the custom x-axis is the square of the for loop index `i` (`i**2`). The `validation_loss` metric is logged as `1 / (i + 1)`.

```python
import wandb

with wandb.init() as run:
    # define which metrics will be plotted against it
    run.define_metric(name = "validation_loss", step_metric = "x_axis_squared")

    for i in range(10):
        log_dict = {
            "x_axis_squared": i**2,
            "validation_loss": 1 / (i + 1),
        }
        run.log(log_dict)
```

The following image shows the resulting plot in the W&B App UI. The `validation_loss` metric is plotted against the custom x-axis `x_axis_squared`, which is the square of the for loop index `i`. Note that the x-axis values are `0, 1, 4, 9, 16, 25, 36, 49, 64, 81`, which correspond to the squares of `0, 1, 2, ..., 9` respectively.

{{< img src="/images/experiments/custom_x_axes.png" alt="Line plot panel that uses a custom x axis. Values are logged to W&B as the square of the loop number." >}}

You can set a custom x-axis for multiple metrics using `globs` with string prefixes. As an example, the following code snippet plots logged metrics with the prefix `train/*` to the x-axis `train/step`:

```python
import wandb

with wandb.init() as run:

    # set all other train/ metrics to use this step
    run.define_metric("train/*", step_metric="train/step")

    for i in range(10):
        log_dict = {
            "train/step": 2**i,  # exponential growth w/ internal W&B step
            "train/loss": 1 / (i + 1),  # x-axis is train/step
            "train/accuracy": 1 - (1 / (1 + i)),  # x-axis is train/step
            "val/loss": 1 / (1 + i),  # x-axis is internal wandb step
        }
        run.log(log_dict)
```


<!-- [Try `define_metric` in Google Colab](http://wandb.me/define-metric-colab). -->