---
title: MetricChangeFilter
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/_filters/run_metrics.py#L142-L188 >}}

Defines a filter that compares a change in a run metric against a user-defined threshold.

The change is calculated over "tumbling" windows, i.e. the difference
between the current window and the non-overlapping prior window.

| Attributes |  |
| :--- | :--- |
|  `prior_window` |  Size of the prior window over which the metric is aggregated (ignored if `agg is None`). If omitted, defaults to the size of the current window. |
|  `name` |  Name of the observed metric. |
|  `agg` |  Aggregate operation, if any, to apply over the window size. |
|  `window` |  Size of the window over which the metric is aggregated (ignored if `agg is None`). |
|  `threshold` |  Threshold value to compare against. |
