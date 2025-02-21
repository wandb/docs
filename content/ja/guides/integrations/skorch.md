---
title: Skorch
description: W&B を Skorch と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-skorch
    parent: integrations
weight: 400
---

Weights & Biases を Skorch と一緒に使用することで、各エポック後に最もパフォーマンスの良いモデルとすべてのモデルパフォーマンスメトリクス、モデルトポロジー、計算リソースを自動的にログすることができます。 `wandb_run.dir` に保存されたすべてのファイルは、自動的に W&B サーバーにログされます。

[例の run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13) を参照してください。

## パラメータ

| パラメータ | タイプ | 説明 |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | データのログに使用される wandb run。 |
| `save_model` | bool (default=True) | 最良のモデルのチェックポイントを保存し、それを W&B サーバーのあなたの Run にアップロードするかどうか。 |
| `keys_ignored` | str または list of str (default=None) | Tensorboard にログされないキーまたはキーのリスト。ユーザーによって提供されたキーに加え、`event_` で始まるキーや `_best` で終わるキーなどはデフォルトで無視されます。|

## 例のコード

インテグレーションがどのように機能するかを見るために、いくつかの例を作成しました：

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): インテグレーションを試すための簡単なデモ
* [ステップバイステップのガイド](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch モデルパフォーマンスをトラッキングするための

```python
# wandb をインストール
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb Run を作成
wandb_run = wandb.init()
# 別の方法: W&B アカウントなしで wandb Run を作成
wandb_run = wandb.init(anonymous="allow")

# ハイパーパラメータをログ (オプション)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## メソッドリファレンス

| メソッド | 説明 |
| :--- | :--- |
| `initialize`\(\) | コールバックの初期状態を\(再-\)セットします。 |
| `on_batch_begin`\(net\[, X, y, training\]\) | 各バッチの開始時に呼び出されます。 |
| `on_batch_end`\(net\[, X, y, training\]\) | 各バッチの終了時に呼び出されます。 |
| `on_epoch_begin`\(net\[, dataset_train, …\]\) | 各エポックの開始時に呼び出されます。 |
| `on_epoch_end`\(net, \*\*kwargs\) | 最後の履歴ステップから値をログし、最良のモデルを保存します。 |
| `on_grad_computed`\(net, named_parameters\[, X, …\]\) | 勾配が計算された後、更新ステップが行われる前にバッチごとに一度呼び出されます。 |
| `on_train_begin`\(net, \*\*kwargs\) | モデルトポロジーをログし、勾配のためのフックを追加します。 |
| `on_train_end`\(net\[, X, y\]\) | トレーニング終了時に呼び出されます。 |