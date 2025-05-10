---
url: /support/:filename
title: How can I change how frequently to log system metrics?
toc_hide: true
type: docs
support:
  - metrics
  - runs
translationKey: how_can_i_reduce_how_frequently_to_log_system_metrics
---
To configure the frequency to log [system metrics]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}), set `_stats_sampling_interval` to a number of seconds, expressed as a float. Default: `10.0`.

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```