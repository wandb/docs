---
title: TensorFlow スイープ
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow_sweeps
    parent: integration-tutorials
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb" >}}
W&B を使用して、機械学習実験管理、データセットのバージョン管理、プロジェクトコラボレーションを行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

W&B Sweeps を使用してハイパーパラメーターの最適化を自動化し、インタラクティブな ダッシュボードでモデルの可能性を探りましょう:

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="" >}}

## なぜ sweeps を使うのか

* **クイックセットアップ**: W&B sweeps を数行のコードで実行。
* **透明性**: このプロジェクトは使用されるすべてのアルゴリズムを引用し、[コードはオープンソース](https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py)です。
* **強力**: Sweeps はカスタマイズオプションを提供し、複数のマシンやノートパソコンで簡単に実行できます。

詳しくは、[Sweep ドキュメント]({{< relref path="/guides/models/sweeps/" lang="ja" >}})を参照してください。

## このノートブックで扱う内容

* W&B Sweep と TensorFlow でのカスタム トレーニングループを開始する手順。
* 画像分類タスクのための最適なハイパーパラメーターを見つけること。

**注意**: _Step_ から始まるセクションには、ハイパーパラメータースイープを実行するために必要なコードが示されています。それ以外はシンプルな例を設定します。

## インストール、インポート、ログイン

### W&B のインストール

```bash
pip install wandb
```

### W&B のインポートとログイン

```python
import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
```

{{< alert >}}
W&B が初めてであるか、ログインしていない場合、`wandb.login()` を実行した後のリンクは、新規登録/ログインページへと案内します。
{{< /alert >}}

## データセットの準備

```python
# トレーニングデータセットの準備
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

## 分類器 MLP の構築

```python
def Model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)

    return keras.Model(inputs=inputs, outputs=outputs)


def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, logits)

    return loss_value


def test_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_value
```

## トレーニングループの記述

```python
def train(
    train_dataset,
    val_dataset,
    model,
    optimizer,
    loss_fn,
    train_acc_metric,
    val_acc_metric,
    epochs=10,
    log_step=200,
    val_log_step=50,
):

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # データセットのバッチを繰り返す
        for step, (x_batch_train, y_batch_train) in tqdm.tqdm(
            enumerate(train_dataset), total=len(train_dataset)
        ):
            loss_value = train_step(
                x_batch_train,
                y_batch_train,
                model,
                optimizer,
                loss_fn,
                train_acc_metric,
            )
            train_loss.append(float(loss_value))

        # 各エポックの終わりに検証ループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # 各エポック終了時にメトリクスを表示
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポック終了時にメトリクスをリセット
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # 3️⃣ wandb.log を使用してメトリクスをログ
        wandb.log(
            {
                "epochs": epoch,
                "loss": np.mean(train_loss),
                "acc": float(train_acc),
                "val_loss": np.mean(val_loss),
                "val_acc": float(val_acc),
            }
        )
```

## sweep を設定する

sweep を設定する手順:
* 最適化するハイパーパラメーターを定義する
* 最適化メソッドを選択する: `random`、`grid`、または `bayes`
* `bayes` の目標とメトリクスを設定する。例えば `val_loss` を最小化する
* `hyperband` を使用して、実行中のものを早期終了する

詳しくは [W&B Sweeps ドキュメント]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})を参照してください。

```python
sweep_config = {
    "method": "random",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "early_terminate": {"type": "hyperband", "min_iter": 5},
    "parameters": {
        "batch_size": {"values": [32, 64, 128, 256]},
        "learning_rate": {"values": [0.01, 0.005, 0.001, 0.0005, 0.0001]},
    },
}
```

## トレーニングループを包む

`wandb.config` を使用してハイパーパラメーターを設定してから `train` を呼び出すような関数 `sweep_train` を作成します。

```python
def sweep_train(config_defaults=None):
    # デフォルト値の設定
    config_defaults = {"batch_size": 64, "learning_rate": 0.01}
    # サンプルプロジェクト名で wandb を初期化
    wandb.init(config=config_defaults)  # これは Sweep で上書きされます

    # 他のハイパーパラメータを設定に指定、ある場合
    wandb.config.epochs = 2
    wandb.config.log_step = 20
    wandb.config.val_log_step = 50
    wandb.config.architecture_name = "MLP"
    wandb.config.dataset_name = "MNIST"

    # tf.data を使用して入力パイプラインを構築
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.shuffle(buffer_size=1024)
        .batch(wandb.config.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(wandb.config.batch_size).prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    # モデルを初期化
    model = Model()

    # モデルをトレーニングするためのオプティマイザーをインスタンス化
    optimizer = keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    # 損失関数をインスタンス化
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # メトリクスを準備
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    train(
        train_dataset,
        val_dataset,
        model,
        optimizer,
        loss_fn,
        train_acc_metric,
        val_acc_metric,
        epochs=wandb.config.epochs,
        log_step=wandb.config.log_step,
        val_log_step=wandb.config.val_log_step,
    )
```

## sweep を初期化し、パーソナルデジタルアシスタントを実行

```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count` パラメーターで実行の数を制限します。迅速な実行のために 10 に設定します。必要に応じて増やしてください。

```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

## 結果を視覚化

ライブ結果を見るには、先行する **Sweep URL** リンクをクリックしてください。

## サンプルギャラリー

W&B で追跡および視覚化されたプロジェクトを [Gallery](https://app.wandb.ai/gallery) で探索してください。

## ベストプラクティス

1. **Projects**: 複数の実行をプロジェクトに記録し、それらを比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数プロセスまたはクロスバリデーション折りたたみのために各プロセスを run として記録し、それらをグループ化します。`wandb.init(group='experiment-1')`
3. **Tags**: ベースラインまたはプロダクションモデルを追跡するためにタグを使用します。
4. **Notes**: テーブルのメモに run 間の変更を追跡するためのメモを入力します。
5. **Reports**: レポートを使用して進捗メモを作成し、同僚と共有し、MLプロジェクトのダッシュボードとスナップショットを作成します。

## 高度なセットアップ

1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): 管理されたクラスターでのトレーニングのために API キーを設定します。
2. [オフラインモード]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): W&B をプライベートクラウドまたはインフラストラクチャ内のエアギャップサーバーにインストールします。ローカルインストールは学術機関および企業チームに適しています。