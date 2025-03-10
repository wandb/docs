---
menu:
  support:
    identifier: ja-support-how_can_i_reduce_how_frequently_to_log_system_metrics
tags:
- metrics
- runs
title: How can I change how frequently to log system metrics?
toc_hide: true
type: docs
---

To configure the frequency to log [system metrics]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}), set `_stats_sampling_interval` to a number of seconds, expressed as a float. Default: `10.0`.

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```