---
title: スコーチ
description: W&B を Skorch と統合する方法
menu:
  default:
    identifier: ja-guides-integrations-skorch
    parent: integrations
weight: 400
---

W&B と Skorch を組み合わせて使用することで、最良のパフォーマンスを持つモデルを自動的にログし、すべてのモデルパフォーマンスメトリクス、モデルのトポロジー、および各エポック後の計算リソースも記録できます。`wandb_run.dir` に保存されたファイルはすべて自動的に W&B にログされます。

[例の Run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13) をご覧ください。

## パラメータ

| パラメータ | 型 | 説明 |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | データのログに使用する wandb Run。 |
|`save_model` | bool (default=True)| 最良のモデルのチェックポイントを保存し、それを W&B の Run にアップロードするかどうか。|
|`keys_ignored`| str または str のリスト (default=None) | tensorboard へログしないキー、またはそのリスト。ユーザーが指定したキーに加え、デフォルトで `event_` で始まるキーや `_best` で終わるキーが無視されます。|

## コード例

インテグレーションの仕組みを体験できるいくつかの例を用意しました：

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): インテグレーションのシンプルなデモ
* [ステップバイステップガイド](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch モデルのパフォーマンスをトラッキングする方法

```python
# wandb をインストール
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb Run の作成
wandb_run = wandb.init()
# 別の方法: W&B アカウントなしで wandb Run を作成
wandb_run = wandb.init(anonymous="allow")

# ハイパーパラメータのログ（任意）
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
| `on_epoch_end`\(net, \*\*kwargs\) | 最後の履歴ステップから値をログし、ベストモデルを保存します。 |
| `on_grad_computed`\(net, named_parameters\[, X, …\]\) | 各バッチの勾配計算後、更新ステップの前に一度だけ呼び出されます。 |
| `on_train_begin`\(net, \*\*kwargs\) | モデルのトポロジーをログし、勾配にフックを追加します。 |
| `on_train_end`\(net\[, X, y\]\) | トレーニング終了時に呼び出されます。 |