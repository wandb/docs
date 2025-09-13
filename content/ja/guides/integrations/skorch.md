---
title: Skorch
description: W&B を Skorch と連携する方法。
menu:
  default:
    identifier: ja-guides-integrations-skorch
    parent: integrations
weight: 400
---

W&B を Skorch と併用することで、各エポックの後にベストパフォーマンスのモデルを自動でログでき、あわせてモデルのパフォーマンス メトリクス全体、モデルのトポロジー、計算リソースも記録できます。`wandb_run.dir` に保存されたファイルはすべて、W&B に自動でログされます。

[サンプル Run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13) を参照してください。

## パラメータ

| パラメータ | 型 | 説明 |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | データをログするために使用する wandb の Run。 |
|`save_model` | bool（デフォルト=True）| ベストモデルのチェックポイントを保存し、W&B 上のあなたの Run にアップロードするかどうか。|
|`keys_ignored`| str または str のリスト（デフォルト=None） | TensorBoard にログしないキー、またはキーのリスト。ユーザーが指定したキーに加えて、`event_` で始まるものや `_best` で終わるものなどのキーは、デフォルトで無視されます。|

## サンプルコード

インテグレーションの動作を確認できるよう、いくつかの例を用意しました:

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): インテグレーションを試すためのシンプルなデモ
* [ステップバイステップ ガイド](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch モデルのパフォーマンスをトラッキングするため

```python
# wandb をインストール
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb の Run を作成
wandb_run = wandb.init()
# 代替案: W&B アカウントなしで wandb の Run を作成
wandb_run = wandb.init(anonymous="allow")

# ハイパーパラメータをログ（任意）
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## メソッド リファレンス

| メソッド | 説明 |
| :--- | :--- |
| `initialize`() | コールバックの初期状態を（再）設定します。 |
| `on_batch_begin`(net[, X, y, training]) | 各バッチの開始時に呼び出されます。 |
| `on_batch_end`(net[, X, y, training]) | 各バッチの終了時に呼び出されます。 |
| `on_epoch_begin`(net[, dataset_train, …]) | 各エポックの開始時に呼び出されます。 |
| `on_epoch_end`(net, **kwargs) | 最後の履歴ステップから値をログし、ベストモデルを保存します。 |
| `on_grad_computed`(net, named_parameters[, X, …]) | 勾配が計算された後、更新ステップの前に、バッチごとに 1 回呼び出されます。 |
| `on_train_begin`(net, **kwargs) | モデルのトポロジーをログし、勾配用のフックを追加します。 |
| `on_train_end`(net[, X, y]) | トレーニングの終了時に呼び出されます。 |