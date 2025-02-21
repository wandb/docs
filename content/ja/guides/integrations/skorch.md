---
title: Skorch
description: W&B と Skorch を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-skorch
    parent: integrations
weight: 400
---

Skorch で Weights & Biases を使用すると、すべてのモデルパフォーマンスの メトリクス 、モデルトポロジ、および各 エポック 後のコンピューティングリソースとともに、最高のパフォーマンスを持つモデルを自動的に ログ できます。 `wandb_run.dir` に保存されたすべてのファイルは、W&B サーバー に自動的に ログ されます。

[実行例](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13)をご覧ください。

## パラメータ

| パラメータ | タイプ | 説明 |
| :--- | :--- | :--- |
| `wandb_run` | `wandb.wandb_run`. Run | データの ログ に使用される wandb run。 |
| `save_model` | bool (default=True) | 最高のモデルの チェックポイント を保存し、W&B サーバー 上の Run にアップロードするかどうか。 |
| `keys_ignored` | str または str のリスト (default=None) | tensorboard に ログ してはならない キー または キー のリスト。 ユーザー が指定した キー に加えて、`event_` で始まる キー 、または `_best` で終わる キー は、デフォルトで無視されることに注意してください。 |

## コード 例

この インテグレーション の動作を確認するための例をいくつか作成しました。

*   [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing): インテグレーション を試すための簡単な デモ
*   [ステップごとの ガイド](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ): Skorch モデル のパフォーマンスを追跡する

```python
# wandb をインストール
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger

# wandb Run を作成
wandb_run = wandb.init()
# 代替案: W&B アカウントなしで wandb Run を作成
wandb_run = wandb.init(anonymous="allow")

# ハイパーパラメータ を ログ (オプション)
wandb_run.config.update({"learning rate": 1e-3, "batch size": 32})

net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])
net.fit(X, y)
```

## メソッド リファレンス

| メソッド | 説明 |
| :--- | :--- |
| `initialize`\(\) | コールバック の初期状態を (再) 設定します。 |
| `on_batch_begin`\(net\[, X, y, training\]\) | 各 バッチ の開始時に呼び出されます。 |
| `on_batch_end`\(net\[, X, y, training\]\) | 各 バッチ の終了時に呼び出されます。 |
| `on_epoch_begin`\(net\[, dataset_train, …\]\) | 各 エポック の開始時に呼び出されます。 |
| `on_epoch_end`\(net, \*\*kwargs\) | 最後の履歴ステップから 値 を ログ し、最高のモデルを保存します |
| `on_grad_computed`\(net, named_parameters\[, X, …\]\) | 勾配 が計算された後、更新ステップが実行される前に、 バッチ ごとに 1 回呼び出されます。 |
| `on_train_begin`\(net, \*\*kwargs\) | モデル トポロジを ログ し、 勾配 のための hook を追加します |
| `on_train_end`\(net\[, X, y\]\) | トレーニング の終了時に呼び出されます。 |
