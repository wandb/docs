---
title: バッチで一部のメトリクスを ログ し、他のメトリクスは エポック 単位でのみ ログ したい場合はどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_metrics_batches_some_metrics_epochs
support:
- 実験
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

各バッチで特定のメトリクスをログし、プロットを標準化するには、メトリクスと一緒に目的の x 軸の値をログします。カスタム プロットでは、Edit をクリックしてカスタム x 軸を選択します。

```python
import wandb

with wandb.init() as run:
    run.log({"batch": batch_idx, "loss": 0.3})
    run.log({"epoch": epoch, "val_acc": 0.94})
```