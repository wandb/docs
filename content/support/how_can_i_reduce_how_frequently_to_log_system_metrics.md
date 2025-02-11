---
title: How can I reduce how frequently to log system metrics?
toc_hide: true
type: docs
tags:
  - metrics
  - runs
---

To configure the frequency to log [system metrics]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}), set `_stats_sampling_interval` to a number of seconds (default: `10.0`), expressed as a float:

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```