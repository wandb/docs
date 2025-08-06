---
title: Skorch
description: W&B を Skorch と統合する方法
menu:
  default:
    identifier: skorch
    parent: integrations
weight: 400
---

W&B を Skorch と組み合わせて使用すると、各エポック終了後に最もパフォーマンスの良いモデルや、すべてのモデルパフォーマンスメトリクス、モデル構造、計算リソースなどを自動的にログできます。`wandb_run.dir` に保存されたすべてのファイルは自動的に W&B に記録されます。

[example run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13) をご覧ください。

## パラメータ

| パラメータ | 型 | 説明 |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | データのログに使用する wandb run。|
| `save_model` | bool (デフォルト=True) | 最良のモデルのチェックポイントを保存し、W&B 上の Run にアップロードするかどうか。|
| `keys_ignored` | str または str のリスト (デフォルト=None) | tensorboard へログしないキーまたはキーのリスト。ユーザーが指定したキーの他、`event_` で始まるものや `_best` で終わるキーもデフォルトで無視されます。|

## サンプルコード

インテグレーションの動作を確認できる例をいくつかご用意しています。

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): インテグレーションを試せるシンプルなデモ
* [ステップバイステップガイド](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch モデルのパフォーマンスをトラッキングする方法

```python
# wandb をインストール
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb Run を作成
wandb_run = wandb.init()
# 別の方法: W&B アカウントなしで Run を作成
wandb_run = wandb.init(anonymous="allow")

# ハイパーパラメータをログ（オプション）
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## メソッドリファレンス

| メソッド | 説明 |
| :--- | :--- |
| `initialize`\(\) | コールバックの初期状態を（再）設定します。 |
| `on_batch_begin`\(net\[, X, y, training\]\) | 各バッチの処理開始時に呼び出されます。 |
| `on_batch_end`\(net\[, X, y, training\]\) | 各バッチの処理終了時に呼び出されます。 |
| `on_epoch_begin`\(net\[, dataset_train, …\]\) | 各エポックの開始時に呼び出されます。 |
| `on_epoch_end`\(net, \*\*kwargs\) | 最後のヒストリーステップから値をログし、ベストモデルを保存します。 |
| `on_grad_computed`\(net, named_parameters\[, X, …\]\) | 勾配計算後、アップデートが行われる前に各バッチごとに一度呼び出されます。 |
| `on_train_begin`\(net, \*\*kwargs\) | モデル構造をログし、勾配用のフックを追加します。 |
| `on_train_end`\(net\[, X, y\]\) | トレーニング完了時に呼び出されます。 |