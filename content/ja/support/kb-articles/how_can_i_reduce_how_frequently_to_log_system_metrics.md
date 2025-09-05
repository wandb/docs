---
title: システム メトリクスをログする頻度をどのように変更できますか？
menu:
  support:
    identifier: >-
      ja-support-kb-articles-how_can_i_reduce_how_frequently_to_log_system_metrics
support:
  - metrics
  - runs
toc_hide: true
type: docs
url: /ja/support/:filename
---
頻度を設定して[システムメトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}})をログするには、`_stats_sampling_interval` を秒数で設定します。これは浮動小数点として表現されます。デフォルトは `10.0` です。

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```