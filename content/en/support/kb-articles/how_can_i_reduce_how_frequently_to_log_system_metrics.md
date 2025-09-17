---
url: /support/:filename
title: How can I change how frequently to log system metrics?
toc_hide: true
type: docs
support:
  - metrics
  - runs
---

To configure the frequency to log [system metrics]({{< relref "/ref/system-metrics.md" >}}), set `x_stats_sampling_interval` to a number of seconds, expressed as a float.

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=15))
```
