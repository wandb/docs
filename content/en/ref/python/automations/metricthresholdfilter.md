---
title: MetricThresholdFilter
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/_filters/run_metrics.py#L119-L139 >}}

Defines a filter that compares a run metric against a user-defined threshold value.

| Attributes |  |
| :--- | :--- |
|  `cmp` |  Comparison operator used to compare the metric value (left) vs. the threshold value (right). |
|  `name` |  Name of the observed metric. |
|  `agg` |  Aggregate operation, if any, to apply over the window size. |
|  `window` |  Size of the window over which the metric is aggregated (ignored if `agg is None`). |
|  `threshold` |  Threshold value to compare against. |
