---
title: Customize log axes
menu:
  default:
    identifier: ja-guides-models-track-log-customize-logging-axes
    parent: log-objects-and-media
---

`define_metric` を使用して、**カスタムの X 軸** を設定します。カスタムの X 軸は、トレーニング 中に過去の異なるタイムステップに非同期で ログ を記録する必要がある場合に役立ちます。たとえば、これは、エピソードごとの報酬とステップごとの報酬を追跡する RL で役立ちます。

[Google Colab で `define_metric` を実際に試してみる →](http://wandb.me/define-metric-colab)

### 軸のカスタマイズ

デフォルトでは、すべての メトリクス は同じ X 軸に対して ログ 記録されます。これは W&B の内部 `step` です。場合によっては、前のステップに ログ したり、別の X 軸を使用したりする必要があるかもしれません。

以下は、デフォルトのステップの代わりに、カスタムの X 軸 メトリクス を設定する例です。

```python
import wandb

wandb.init()
# カスタムの X 軸 メトリクス を定義する
wandb.define_metric("custom_step")
# どの メトリクス をそれに対してプロットするかを定義する
wandb.define_metric("validation_loss", step_metric="custom_step")

for i in range(10):
    log_dict = {
        "train_loss": 1 / (i + 1),
        "custom_step": i**2,
        "validation_loss": 1 / (i + 1),
    }
    wandb.log(log_dict)
```

X 軸は、グロブを使用して設定することもできます。現在、文字列プレフィックスを持つグロブのみが利用可能です。次の例では、プレフィックス `"train/"` を持つ ログ 記録されたすべての メトリクス を X 軸 `"train/step"` にプロットします。

```python
import wandb

wandb.init()
# カスタムの X 軸 メトリクス を定義する
wandb.define_metric("train/step")
# 他のすべての train/ メトリクス がこのステップを使用するように設定する
wandb.define_metric("train/*", step_metric="train/step")

for i in range(10):
    log_dict = {
        "train/step": 2**i,  # 内部 W&B ステップによる指数関数的な増加
        "train/loss": 1 / (i + 1),  # X 軸は train/step
        "train/accuracy": 1 - (1 / (1 + i)),  # X 軸は train/step
        "val/loss": 1 / (1 + i),  # X 軸は内部 wandb ステップ
    }
    wandb.log(log_dict)
```