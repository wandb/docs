---
title: システムメトリクスの W&B へのログを無効にするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_disable_logging_of_system_metrics_to_wb
support:
- metrics
- runs
toc_hide: true
type: docs
url: /support/:filename
---

[システムメトリクス]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}})のログを無効にするには、`_disable_stats` を `True` に設定します:

```python
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```