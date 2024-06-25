---
description: W&BをSkorchと統合する方法
slug: /guides/integrations/skorch
displayed_sidebar: default
---


# Skorch

Weights & Biases と Skorch を組み合わせて使用すると、最高のパフォーマンスを持つモデルを自動的にログすることができます。これには、すべてのモデルパフォーマンスのメトリクス、モデルのトポロジー、各エポック後の計算リソースが含まれます。wandb_run.dir に保存されたすべてのファイルは自動的に W&B サーバーにログされます。

[example run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13) を参照してください。

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `wandb_run` |  wandb.wandb_run.Run | データをログするために使用される wandb run。|
|`save_model` | bool (default=True)| 最良のモデルのチェックポイントを保存し、W&B サーバーの Run にアップロードするかどうか。|
|`keys_ignored`| str または str のリスト (default=None) | tensorboard にログを記録しないキーまたはキーのリスト。ユーザーが提供するキーに加えて、`event_` で始まるキーや `_best` で終わるキーはデフォルトで無視されることに注意してください。|

## Example Code

インテグレーションの動作を確認できるいくつかの例を作成しました：

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): インテグレーションを試すためのシンプルなデモ
* [A step by step guide](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch モデルのパフォーマンスを追跡するためのステップバイステップガイド

```python
# Install wandb
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# Create a wandb Run
wandb_run = wandb.init()
# Alternative: Create a wandb Run without a W&B account
wandb_run = wandb.init(anonymous="allow")

# Log hyper-parameters (optional)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## Methods

| Method | Description |
| :--- | :--- |
| `initialize`\(\) | コールバックの初期状態を（再）設定します。 |
| `on_batch_begin`\(net\[, X, y, training\]\) | 各バッチの開始時に呼び出されます。 |
| `on_batch_end`\(net\[, X, y, training\]\) | 各バッチの終了時に呼び出されます。 |
| `on_epoch_begin`\(net\[, dataset\_train, …\]\) | 各エポックの開始時に呼び出されます。 |
| `on_epoch_end`\(net, \*\*kwargs\) | 最後の履歴ステップから値をログし、最良のモデルを保存します。 |
| `on_grad_computed`\(net, named\_parameters\[, X, …\]\) | 勾配が計算された後、更新ステップが実行される前に各バッチで一度呼び出されます。 |
| `on_train_begin`\(net, \*\*kwargs\) | モデルのトポロジーをログし、勾配のフックを追加します。 |
| `on_train_end`\(net\[, X, y\]\) | トレーニングの終了時に呼び出されます。 |