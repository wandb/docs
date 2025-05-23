---
title: Skorch
description: W&B を Skorch と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-skorch
    parent: integrations
weight: 400
---

Weights & Biases を Skorch と一緒に使うことで、各エポックの後に最もパフォーマンスの良いモデルを自動的にログし、すべてのモデルパフォーマンスメトリクス、モデルトポロジー、計算リソースを記録することができます。`wandb_run.dir` に保存されたすべてのファイルは、自動的に W&B サーバーにログされます。

[example run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13) を参照してください。

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | データをログするために使用される wandb run。 |
|`save_model` | bool (default=True)| 最良のモデルのチェックポイントを保存し、W&B サーバー上の Run にアップロードするかどうか。|
|`keys_ignored`| str or list of str (default=None) | tensorboard にログされるべきでないキーまたはキーのリスト。ユーザーが提供するキーに加え、`event_` で始まるか `_best` で終わるキーはデフォルトで無視されます。|

## Example Code

インテグレーションがどのように機能するかを見るためのいくつかの例を作成しました:

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): インテグレーションを試すためのシンプルなデモ
* [A step by step guide](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch モデルのパフォーマンスをトラッキングするためのガイド

```python
# wandb をインストールする
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb Run を作成
wandb_run = wandb.init()
# 代わりの方法: W&B アカウントなしで wandb Run を作成
wandb_run = wandb.init(anonymous="allow")

# ハイパーパラメータをログ (オプション)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## Method reference

| Method | Description |
| :--- | :--- |
| `initialize`\(\) | コールバックの初期状態を（再）設定する。 |
| `on_batch_begin`\(net\[, X, y, training\]\) | 各バッチの開始時に呼び出される。 |
| `on_batch_end`\(net\[, X, y, training\]\) | 各バッチの終了時に呼び出される。 |
| `on_epoch_begin`\(net\[, dataset_train, …\]\) | 各エポックの開始時に呼び出される。 |
| `on_epoch_end`\(net, \*\*kwargs\) | 最後の履歴ステップの値をログし、最良のモデルを保存する。 |
| `on_grad_computed`\(net, named_parameters\[, X, …\]\) | 勾配が計算された後、更新ステップが行われる前に、各バッチごとに一度呼び出される。 |
| `on_train_begin`\(net, \*\*kwargs\) | モデルトポロジーをログし、勾配に対するフックを追加する。 |
| `on_train_end`\(net\[, X, y\]\) | トレーニングの終了時に呼び出される。 |