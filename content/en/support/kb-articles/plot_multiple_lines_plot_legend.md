---
url: /support/:filename
title: "How do I plot multiple lines on a plot with a legend?"
toc_hide: true
type: docs
support:
   - experiments
---

Create a multi-line custom chart with `wandb.plot.line_series()`. Navigate to the [project page]({{< relref "/guides/models/track/project-page.md" >}}) to view the line chart. To add a legend, include the `keys` argument in `wandb.plot.line_series()`. For example:

```python

with wandb.init(project="my_project") as run:

    run.log(
        {
            "my_plot": wandb.plot.line_series(
                xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
            )
        }
    )
```

Refer to additional details about multi-line plots [here]({{< relref "/guides/models/track/log/plots.md#basic-charts" >}}) under the **Multi-line** tab.