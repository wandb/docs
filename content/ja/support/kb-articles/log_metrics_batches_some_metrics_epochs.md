---
title: バッチごとに記録したいメトリクスと、エポックごとにのみ記録したいメトリクスがある場合はどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
- メトリクス
---

特定のメトリクスを各バッチごとにログし、プロットを標準化するには、メトリクスと一緒に目的の x 軸の値もログしてください。カスタムプロットで「編集」をクリックし、カスタム x 軸を選択できます。

```python
import wandb

with wandb.init() as run:
    # バッチごとの値とメトリクスをログ
    run.log({"batch": batch_idx, "loss": 0.3})
    # エポックごとの値とメトリクスをログ
    run.log({"epoch": epoch, "val_acc": 0.94})
```