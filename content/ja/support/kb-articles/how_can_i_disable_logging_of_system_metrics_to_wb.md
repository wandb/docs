---
title: W&B へのシステム メトリクスのログを無効化するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_disable_logging_of_system_metrics_to_wb
support:
- メトリクス
- runs
toc_hide: true
type: docs
url: /support/:filename
---

[システム メトリクス]({{< relref path="/ref/system-metrics.md" lang="ja" >}}) のログを無効にするには、 `_disable_stats` を `True` に設定します:

```python
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```