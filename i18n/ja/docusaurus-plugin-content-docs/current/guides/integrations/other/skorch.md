---
slug: /guides/integrations/skorch
description: W&BとSkorchの統合方法。
---

# Skorch

SkorchとWeights & Biasesを組み合わせることで、ベストパフォーマンスのモデルを自動的にログに記録することができます。また、モデルパフォーマンスメトリクス、モデルトポロジー、エポックごとの計算リソースも一緒に記録されます。wandb_run.dirに保存されたすべてのファイルは、自動的にW&Bサーバーにログされます。

[example run](https://app.wandb.ai/borisd13/skorch/runs/s20or4ct?workspace=user-borisd13)を参照してください。

## パラメータ

| パラメータ | タイプ | 説明 |
| :--- | :--- | :--- |
| `wandb_run` |  wandb.wandb_run.Run | データをログに記録するために使用されるwandb run。|
|`save_model` | bool (デフォルト=True)| ベストなモデルのチェックポイントを保存し、それをW&Bサーバー上の実行にアップロードするかどうか。|
|`keys_ignored`| str または str のリスト (デフォルト=None) | TensorBoardにログしないキーまたはキーリスト。ユーザーが提供するキーに加えて、 `event_`で始まるキーや`_best`で終わるキーなどがデフォルトで無視されることに注意してください。|

## 例コード

統合方法を確認するためにいくつかの例を用意しました。

* [Colab](https://colab.research.google.com/drive/1Bo8SqN1wNPMKv5Bn9NjwGecBxzFlaNZn?usp=sharing) : 統合を試す簡単なデモ
* [ステップバイステップガイド](https://app.wandb.ai/cayush/uncategorized/reports/Automate-Kaggle-model-training-with-Skorch-and-W%26B--Vmlldzo4NTQ1NQ) : Skorchモデルのパフォーマンスを追跡する方法

```python
# wandbをインストール
... pip install wandb

import wandb
from skorch.callbacks import WandbLogger
```
# wandb Runを作成する

wandb_run = wandb.init()

# 代替案：W&Bアカウント無しでwandb Runを作成する

wandb_run = wandb.init(anonymous="allow")



# ハイパーパラメータをログに記録（オプション）

wandb_run.config.update({"学習率": 1e-3, "バッチサイズ": 32})



net = NeuralNet(..., callbacks=[WandbLogger(wandb_run)])

net.fit(X, y)

```



## メソッド



| メソッド | 説明 |

| :--- | :--- |

| `initialize`（) | コールバックの初期状態を（再）設定します。 |

| `on_batch_begin`（net [、X、y、training ]） | 各バッチの開始時に呼び出されます。 |

| `on_batch_end`（net [、X、y、training ]） | 各バッチの終了時に呼び出されます。 |

| `on_epoch_begin`（net [、 dataset\_train 、 … ]） | 各エポックの開始時に呼び出されます。 |

| `on_epoch_end`（net, **kwargs） | 直近の履歴ステップから値をログに記録し、最良のモデルを保存します。 |

| `on_grad_computed`（net, named\_parameters [、X , … ]） | 勾配が計算された後、更新ステップが実行される前に、バッチごとに一度呼び出されます。 |

| `on_train_begin`（net, **kwargs） | モデルのトポロジをログに記録し、勾配のフックを追加します。 |

| `on_train_end`（net [、X、y ]） | トレーニングの終了時に呼び出されます。 |