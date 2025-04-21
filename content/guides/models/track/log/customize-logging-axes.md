---
menu:
  default:
    identifier: customize-logging-axes
    parent: log-objects-and-media
title: Customize log axes
---

Set a custom x-axis when you log metrics to W&B. By default, W&B logs metrics as a function of *steps*. A step corresponds to the number of times you call `wandb.log()`. 

For example, suppose you log a metric in a for loop that iterates 10 times. W&B logs the metric against the step number, which is incremented by 1 each time you log a metric.

The code might look similar to the following:

```python
import wandb

with wandb.init() as run:
  for i in range(10):
    log_dict = {
        "validation/loss": 1/(i+1)   
    }
    run.log(log_dict)
```

The previous code snippet generates a line plot panel in the W&B UI that shows the `validation/loss` metric as a function of the step number. 

{{< img src="/images/experiments/standard_axes.png" alt="" >}}

However, there are cases where you might want to log metrics against a different x-axis. In this case, use the [`define_metric`]({{< relref "ref/python/run/#define_metric" >}}) method to define a custom x-axis. 


Specify the name of the metric you want to create a custom x-axis for with the `name` parameter. Then, set the `step_metric` parameter to the name of the metric you want to use as the x-axis.

The following code snippet demonstrates how to set a custom x-axis metric called `"custom_step"` and log the `"validation_loss"` metric against it. With each iteration in the for loop, the `"custom_step"` is set to `i**2`:

```python
import wandb

with wandb.init() as run:
    # define which metrics will be plotted against it
    run.define_metric(name = "validation_loss", step_metric = "custom_step")

    for i in range(10):
        log_dict = {
            "custom_step": i**2,
            "validation_loss": 1 / (i + 1),
        }
        run.log(log_dict)
```

Within the W&B UI, you can see that the `validation_loss` is plotted against `custom_step`. Note how the x-axis is set to the square of the for loop index `i`.

{{< img src="/images/experiments/custom_x_axes.png" alt="Line plot panel that uses a custom x axis. Values are logged to W&B as the square of the loop number (i**2)." >}}

You can set a custom x-axis for multiple metrics using `globs` with string prefixes. As an example, the following code snippet plots logged metrics with the prefix `"train/*"` to the x-axis `"train/step"`:

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