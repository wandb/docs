---
title: "Optimizing multiple metrics"
displayed_sidebar: support
tags:
   - sweeps
---
To optimize multiple metrics in a single run, use a weighted sum of the individual metrics.

```python
metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
wandb.log({"metric_combined": metric_combined})
```

Log the new combined metric and set it as the optimization objective:

```yaml
metric:
  name: metric_combined
  goal: minimize
```