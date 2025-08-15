---
title: システムメトリクスのログ頻度を変更するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_reduce_how_frequently_to_log_system_metrics
support:
- メトリクス
- run
toc_hide: true
type: docs
url: /support/:filename
---

[システム メトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}) をログする頻度を設定するには、`_stats_sampling_interval` に秒数（float 値）を指定してください。デフォルト値は `10.0` です。

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```