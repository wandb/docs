---
title: TensorFlow Sweeps
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow_sweeps
    parent: integration-tutorials
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb" >}}
W&B を使って、機械学習の実験管理、データセットのバージョン管理、プロジェクトでの共同作業を行いましょう。
{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使う利点" >}}

W&B Sweeps を使ってハイパーパラメーター最適化を自動化し、インタラクティブなダッシュボードでモデルの可能性を探索しましょう:
{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="TensorFlow のハイパーパラメーター探索の結果" >}}

## なぜ Sweeps を使うのか

* **すぐに始められる**: 数行のコードで W&B Sweeps を実行できます。
* **透明性**: 使用したアルゴリズムをすべてプロジェクトで明記し、[コードはオープンソース](https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py)です。
* **強力**: Sweeps は柔軟にカスタマイズでき、複数マシンやノート PC でも簡単に動かせます。

詳しくは [Sweeps 概要]({{< relref path="/guides/models/sweeps/" lang="ja" >}})を参照してください。

## このノートブックで扱う内容

* W&B Sweep と TensorFlow のカスタムトレーニングループを始める手順。
* 画像分類タスクの最適なハイパーパラメーターの見つけ方。

**注**: _Step_ で始まるセクションは、ハイパーパラメーター探索を実行するために必要なコードを示します。残りの部分はシンプルな例のセットアップです。

## インストール、インポート、ログイン

### W&B をインストール

```bash
pip install wandb
```

### W&B をインポートしてログイン

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
W&B が初めて、または未ログインの場合は、`wandb.login()` 実行後のリンクからサインアップ／ログインページへ移動します。
{{< /alert >}}

## データセットを準備する

```python
# トレーニングデータセットを準備する
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

## 分類器 MLP を構築する

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

## トレーニングループを書く

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
    run = wandb.init(
        project="sweeps-tensorflow",
        job_type="train",
        config={
            "epochs": epochs,
            "log_step": log_step,
            "val_log_step": val_log_step,
            "architecture_name": "MLP",
            "dataset_name": "MNIST",
        },
    )
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # データセットのバッチを繰り返し処理する
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

        # 各エポックの終わりに検証ループを実行する
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # 各エポックの終わりにメトリクスを表示する
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポックの終わりにメトリクスをリセットする
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # 3. run.log() を使ってメトリクスをログする
        run.log(
            {
                "epochs": epoch,
                "loss": np.mean(train_loss),
                "acc": float(train_acc),
                "val_loss": np.mean(val_loss),
                "val_acc": float(val_acc),
            }
        )
    run.finish()
```

## Sweep を設定する

Sweep を設定する手順:
* 最適化するハイパーパラメーターを定義する
* 最適化メソッドを選ぶ: `random`、`grid`、または `bayes`
* `bayes` の場合は `val_loss` を最小化するなどのゴールとメトリクスを設定する
* 実行中の run の早期終了に `hyperband` を使う

詳しくは [sweep configuration ガイド]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})を参照してください。

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

## トレーニングループをラップする

`run.config()` を使って `train` を呼び出す前にハイパーパラメーターを設定する関数（例: `sweep_train`）を作成します。

```python
def sweep_train(config_defaults=None):
    # 既定値を設定する
    config_defaults = {"batch_size": 64, "learning_rate": 0.01}
    # サンプルのプロジェクト名で wandb を初期化する
    run = wandb.init(config=config_defaults)  # これは Sweep で上書きされる

    # 必要に応じて他のハイパーパラメーターを設定に指定する
    run.config.epochs = 2
    run.config.log_step = 20
    run.config.val_log_step = 50
    run.config.architecture_name = "MLP"
    run.config.dataset_name = "MNIST"

    # tf.data を使って入力パイプラインを構築する
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.shuffle(buffer_size=1024)
        .batch(run.config.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(run.config.batch_size).prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    # モデルを初期化する
    model = Model()

    # モデルを学習させるためのオプティマイザーを作成する
    optimizer = keras.optimizers.SGD(learning_rate=run.config.learning_rate)
    # 損失関数を作成する
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # メトリクスを用意する
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
        epochs=run.config.epochs,
        log_step=run.config.log_step,
        val_log_step=run.config.val_log_step,
    )
    run.finish()
```

## Sweep を初期化してエージェントを実行する

```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count` パラメータで run の数を制限します。手早く実行するには 10 に設定し、必要に応じて増やしてください。

```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

## 結果を可視化する

直前に表示される **Sweep URL** をクリックすると、ライブの結果を確認できます。

## 例のギャラリー

W&B でトラッキングと可視化を行っているプロジェクトを [Gallery](https://app.wandb.ai/gallery) で探索しましょう。

## ベストプラクティス
1. **Projects**: 複数の run を 1 つの Project にログして比較する。`wandb.init(project="project-name")`
2. **グループ**: 複数プロセスやクロスバリデーションの分割ごとに、それぞれを run としてログし、グループ化する。`wandb.init(group='experiment-1')`
3. **タグ**: ベースラインやプロダクションのモデルを追跡するためにタグを使う。
4. **ノート**: テーブルにノートを入力して、run 間の変更を追跡する。
5. **Reports**: Reports を進捗メモ、同僚との共有、ML プロジェクトのダッシュボードやスナップショットの作成に活用する。

## 高度なセットアップ
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): マネージドなクラスターでのトレーニング用に API キーを設定する。
2. [オフラインモード]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): 自社のインフラで、プライベートクラウドやエアギャップ環境のサーバーに W&B をインストールする。ローカルインストールは研究者やエンタープライズチームに適している。