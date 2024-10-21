---
title: "How do I plot multiple lines on a plot with a legend?"
tags:
   - experiments
---

Multi-line custom chart can be created by using `wandb.plot.line_series()`. You'll need to navigate to the [project page](../../app/pages/project-page.md) to see the line chart. To add a legend to the plot, pass the keys argument within `wandb.plot.line_series()`. For example:

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

You can find more information about Multi-line plots [here](../../track/log/plots.md#basic-charts) under the Multi-line tab.