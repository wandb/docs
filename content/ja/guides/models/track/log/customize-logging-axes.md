---
title: ログ軸をカスタマイズする
menu:
  default:
    identifier: ja-guides-models-track-log-customize-logging-axes
    parent: log-objects-and-media
---

`define_metric` を使用して**カスタム x 軸**を設定します。 カスタム x 軸は、トレーニング中に過去の異なるタイムステップに非同期でログを記録する必要がある場合に便利です。たとえば、RL ではエピソードごとの報酬やステップごとの報酬を追跡する場合に役立ちます。

[Google Colab で `define_metric` を試す →](http://wandb.me/define-metric-colab)

### 軸をカスタマイズする

デフォルトでは、すべてのメトリクスは同じ x 軸に対してログが記録されます。これは、 W&B 内部の `step` です。時には、以前のステップにログを記録したい場合や、別の x 軸を使用したい場合があります。

以下は、デフォルトのステップの代わりにカスタムの x 軸メトリクスを設定する例です。

```python
import wandb

wandb.init()
# カスタム x 軸メトリクスを定義
wandb.define_metric("custom_step")
# どのメトリクスがそれに対してプロットされるかを定義
wandb.define_metric("validation_loss", step_metric="custom_step")

for i in range(10):
    log_dict = {
        "train_loss": 1 / (i + 1),
        "custom_step": i**2,
        "validation_loss": 1 / (i + 1),
    }
    wandb.log(log_dict)
```

x 軸はグロブを使用して設定することもできます。現在、文字列のプレフィックスを持つグロブのみが使用可能です。次の例では、プレフィックス `"train/"` を持つすべてのログされたメトリクスを、x 軸 `"train/step"` にプロットします:

```python
import wandb

wandb.init()
# カスタム x 軸メトリクスを定義
wandb.define_metric("train/step")
# 他のすべての train/ メトリクスをこのステップに使用するように設定
wandb.define_metric("train/*", step_metric="train/step")

for i in range(10):
    log_dict = {
        "train/step": 2**i,  # W&B 内部ステップと指数的な成長
        "train/loss": 1 / (i + 1),  # x 軸は train/step
        "train/accuracy": 1 - (1 / (1 + i)),  # x 軸は train/step
        "val/loss": 1 / (1 + i),  # x 軸は内部 wandb ステップ
    }
    wandb.log(log_dict)
```
