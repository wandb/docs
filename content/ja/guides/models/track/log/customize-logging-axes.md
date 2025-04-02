---
title: Customize log axes
menu:
  default:
    identifier: ja-guides-models-track-log-customize-logging-axes
    parent: log-objects-and-media
---

`define_metric` を使用して、**カスタムのX軸**を設定します。カスタムのX軸は、トレーニング中に過去の異なるタイムステップに非同期で ログ を記録する必要がある場合に役立ちます。たとえば、エピソードごとの報酬とステップごとの報酬を追跡する強化学習で役立ちます。

[Google Colab で `define_metric` を実際に試してみる →](http://wandb.me/define-metric-colab)

### 軸のカスタマイズ

デフォルトでは、すべての メトリクス は同じX軸（W&Bの内部 `step`）に対して ログ が記録されます。場合によっては、前のステップに ログ を記録したり、別のX軸を使用したりすることがあります。

カスタムのX軸 メトリクス を設定する例を次に示します（デフォルトのステップの代わりに）。

```python
import wandb

wandb.init()
# カスタムのX軸メトリクスを定義
wandb.define_metric("custom_step")
# どのメトリクスをそれに対してプロットするかを定義
wandb.define_metric("validation_loss", step_metric="custom_step")

for i in range(10):
    log_dict = {
        "train_loss": 1 / (i + 1),
        "custom_step": i**2,
        "validation_loss": 1 / (i + 1),
    }
    wandb.log(log_dict)
```

X軸は、グロブを使用して設定することもできます。現在、文字列のプレフィックスを持つグロブのみが使用可能です。次の例では、プレフィックス `"train/"` を持つ ログ に記録されたすべての メトリクス をX軸 `"train/step"` にプロットします。

```python
import wandb

wandb.init()
# カスタムのX軸メトリクスを定義
wandb.define_metric("train/step")
# 他のすべての train/ メトリクス がこのステップを使用するように設定
wandb.define_metric("train/*", step_metric="train/step")

for i in range(10):
    log_dict = {
        "train/step": 2**i,  # 内部W&Bステップによる指数関数的な増加
        "train/loss": 1 / (i + 1),  # X軸は train/step
        "train/accuracy": 1 - (1 / (1 + i)),  # X軸は train/step
        "val/loss": 1 / (1 + i),  # X軸は内部 wandb step
    }
    wandb.log(log_dict)
```
