---
menu:
  support:
    identifier: ja-support-kb-articles-plot_multiple_lines_plot_legend
support:
- experiments
title: How do I plot multiple lines on a plot with a legend?
toc_hide: true
type: docs
url: /support/:filename
---

Create a multi-line custom chart with `wandb.plot.line_series()`. Navigate to the [project page]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) to view the line chart. To add a legend, include the `keys` argument in `wandb.plot.line_series()`. For example:

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

Refer to additional details about multi-line plots [here]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ja" >}}) under the **Multi-line** tab.