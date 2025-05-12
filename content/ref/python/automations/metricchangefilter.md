---
title: MetricChangeFilter
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.19.11/wandb/automations/_filters/run_metrics.py#L144-L190 >}}

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
