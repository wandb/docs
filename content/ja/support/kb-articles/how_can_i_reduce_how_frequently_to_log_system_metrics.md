---
title: システム メトリクス の ログ 頻度を変更するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_reduce_how_frequently_to_log_system_metrics
support:
- メトリクス
- runs
toc_hide: true
type: docs
url: /support/:filename
---

システム メトリクスを ログ する頻度を 設定 するには、`_stats_sampling_interval` に秒数（浮動小数点数）を指定します。デフォルト: `10.0`。

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```