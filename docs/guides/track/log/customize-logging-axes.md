---
displayed_sidebar: default
---


# ログの軸をカスタマイズする

`define_metric` を使用して **カスタム x 軸** を設定します。カスタム x 軸は、トレーニング中に過去の異なるタイムステップに非同期でログを取る必要がある場合に便利です。例えば、エピソードごとの報酬とステップごとの報酬を追跡するRLにおいて有用です。

[Google Colabで `define_metric` を試してみる →](http://wandb.me/define-metric-colab)

### 軸をカスタマイズする

デフォルトでは、すべてのメトリクスは同じx軸、すなわちW&B内部の `step` に対してログされます。時には、以前のステップに対してログを取ったり、異なるx軸を使用したいことがあるかもしれません。

デフォルトのステップの代わりにカスタムx軸メトリクスを設定する例を示します。

```python
import wandb

wandb.init()
# カスタムx軸メトリクスを定義
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

x軸はグロブを使用して設定することもできます。現在のところ、文字列プレフィックスを持つグロブのみが利用可能です。次の例では、`"train/"` というプレフィックスを持つすべてのログされたメトリクスが x軸 `"train/step"` にプロットされます。

```python
import wandb

wandb.init()
# カスタムx軸メトリクスを定義
wandb.define_metric("train/step")
# 他のすべての train/ メトリクスがこのステップを使用するように設定
wandb.define_metric("train/*", step_metric="train/step")

for i in range(10):
    log_dict = {
        "train/step": 2**i,  # W&B内部ステップによる指数成長
        "train/loss": 1 / (i + 1),  # x軸は train/step
        "train/accuracy": 1 - (1 / (1 + i)),  # x軸は train/step
        "val/loss": 1 / (1 + i),  # x軸は内部のwandbステップ
    }
    wandb.log(log_dict)
```
