---
title: How can I change how frequently to log system metrics?
menu:
  support:
    identifier: ja-support-how_can_i_reduce_how_frequently_to_log_system_metrics
tags:
- metrics
- runs
toc_hide: true
type: docs
---

[システム メトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}) をログする頻度を設定するには、 `_stats_sampling_interval` を秒数（ float 型）で指定します。デフォルト: `10.0`.

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```