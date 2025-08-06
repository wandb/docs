---
menu:
  support:
    identifier: ko-support-kb-articles-optimizing_multiple_metrics
support:
- sweeps
- metrics
title: Optimizing multiple metrics
toc_hide: true
type: docs
url: /support/:filename
---

To optimize multiple metrics in a single run, use a weighted sum of the individual metrics.

```python
with wandb.init() as run:
  # Log individual metrics
  metric_a = run.summary.get("metric_a", 0.5)
  metric_b = run.summary.get("metric_b", 0.7)
  # ... log other metrics as needed
  metric_n = run.summary.get("metric_n", 0.9)

  # Combine metrics with weights
  # Adjust weights according to your optimization goals
  # For example, if you want to give more importance to metric_a and metric_n:  
  metric_combined = 0.3 * metric_a + 0.2 * metric_b + ... + 1.5 * metric_n
  run.log({"metric_combined": metric_combined})
```

Log the new combined metric and set it as the optimization objective:

```yaml
metric:
  name: metric_combined
  goal: minimize
```