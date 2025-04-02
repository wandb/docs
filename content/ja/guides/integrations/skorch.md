---
title: Skorch
description: Skorch と W&B を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-skorch
    parent: integrations
weight: 400
---

Skorch で Weights & Biases を使用すると、各エポック後に、すべてのモデルパフォーマンスメトリクス、モデルトポロジ、およびコンピューティングリソースとともに、最高のパフォーマンスを持つモデルを自動的にログ記録できます。 `wandb_run.dir` に保存されたすべてのファイルは、W&B サーバーに自動的にログ記録されます。

[run の例](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13) を参照してください。

## パラメータ

| パラメータ | タイプ | 説明 |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | データのログ記録に使用される wandb run。 |
| `save_model` | bool (default=True) | 最適なモデルのチェックポイントを保存し、W&B サーバー上の Run にアップロードするかどうか。 |
| `keys_ignored` | str または str のリスト (default=None) | tensorboard にログ記録しないキーまたはキーのリスト。 ユーザーが提供するキーに加えて、`event_` で始まるキーや `_best` で終わるキーはデフォルトで無視されることに注意してください。 |

## コード例

このインテグレーションの動作を確認するための例をいくつか作成しました。

*   [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): インテグレーションを試すための簡単なデモ
*   [ステップごとのガイド](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch モデルのパフォーマンスを追跡する

```python
# wandb をインストール
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb Run を作成
wandb_run = wandb.init()
# 代替案: W&B アカウントなしで wandb Run を作成
wandb_run = wandb.init(anonymous="allow")

# ハイパーパラメータをログ記録 (オプション)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## メソッドリファレンス

| メソッド | 説明 |
| :--- | :--- |
| `initialize`\(\) | コールバックの初期状態を（再）設定します。 |
| `on_batch_begin`\(net\[, X, y, training\]\) | 各バッチの開始時に呼び出されます。 |
| `on_batch_end`\(net\[, X, y, training\]\) | 各バッチの終了時に呼び出されます。 |
| `on_epoch_begin`\(net\[, dataset_train, …\]\) | 各エポックの開始時に呼び出されます。 |
| `on_epoch_end`\(net, \*\*kwargs\) | 最後の履歴ステップから値をログ記録し、最適なモデルを保存します。 |
| `on_grad_computed`\(net, named_parameters\[, X, …\]\) | 勾配が計算された後、更新ステップが実行される前に、バッチごとに 1 回呼び出されます。 |
| `on_train_begin`\(net, \*\*kwargs\) | モデルトポロジをログ記録し、勾配の hook を追加します。 |
| `on_train_end`\(net\[, X, y\]\) | トレーニングの終了時に呼び出されます。 |
