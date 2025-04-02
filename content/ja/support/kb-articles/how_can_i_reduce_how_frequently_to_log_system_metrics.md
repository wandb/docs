---
title: How can I change how frequently to log system metrics?
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_reduce_how_frequently_to_log_system_metrics
support:
- metrics
- runs
toc_hide: true
type: docs
url: /support/:filename
---

[システムメトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}) のログを記録する頻度を設定するには、`_stats_sampling_interval` を秒数（浮動小数点数で表現）に設定します。デフォルト: `10.0`。

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```
