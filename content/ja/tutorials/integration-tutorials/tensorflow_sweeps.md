---
title: TensorFlow Sweeps
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow_sweeps
    parent: integration-tutorials
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb" >}}  
Weights & Biases を使用して、機械学習の実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

Weights & Biases Sweeps を使用して、ハイパーパラメーター最適化を自動化し、対話型のダッシュボードを活用して可能なモデルの領域を探索しましょう:

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="" >}}

## なぜ Sweeps を使用するべきですか？

* **簡単なセットアップ**: わずか数行のコードで W&B Sweeps を実行できます。
* **透明性**: プロジェクトでは使用される全てのアルゴリズムが引用されており、[コードはオープンソースです](https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py)。
* **強力**: Sweeps は完全にカスタマイズ可能で設定可能です。多数のマシンで sweep を開始するのも、あなたのラップトップで sweep を開始するのも同じくらい簡単です。

**[公式ドキュメントをチェック]({{< relref path="/guides/models/sweeps/" lang="ja" >}})**

## このノートブックがカバーする内容

* カスタムトレーニングループを用いた W&B Sweep の簡単な開始手順。
* 画像分類タスクにおける最適なハイパーパラメーターの発見。

**注**: _Step_ から始まるセクションは既存のコードでハイパーパラメーター探索を行うために必要なもの全てです。他のコードはシンプルな例を設定するためにあります。

## インストール、インポート、およびログイン

### W&B のインストール

```bash
%%capture
!pip install wandb
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

> 補足: 初めて W&B を使用する場合やログインしていない場合、`wandb.login()` の実行後に表示されるリンクがサインアップ/ログインページに誘導します。サインアップはほんの数クリックで完了します。

## データセットの準備

```python
# 訓練データセットの準備
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

## シンプルな分類器 MLP の構築

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

## トレーニングループの作成

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

        # データセットのバッチを繰り返し処理
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

        # 各エポックの最後に検証ループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # 各エポックの最後にメトリクスを表示
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポックの最後にメトリクスをリセット
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

## Sweep の設定

ここでは以下を行います:
* 探索するハイパーパラメーターを定義する
* ハイパーパラメーター最適化メソッドを指定する。`random`、`grid`、`bayes` メソッドがあります。
* `bayes` を使用する場合、目的と `metric` を提供します。例として `val_loss` を `minimize` します。
* パフォーマンスが低い run を早期終了するために `hyperband` を使用します。

#### [Sweep 設定について詳しくはこちらをチェック]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})

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

`train` が呼び出される前に `wandb.config` を使用してハイパーパラメーターをセットする `sweep_train` のような関数が必要です。

```python
def sweep_train(config_defaults=None):
    # デフォルト値を設定
    config_defaults = {"batch_size": 64, "learning_rate": 0.01}
    # サンプルプロジェクト名で wandb を初期化
    wandb.init(config=config_defaults)  # Sweeps で上書きされます

    # 他のハイパーパラメーターを設定に指定する場合
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

    # モデルの初期化
    model = Model()

    # モデルを訓練するためのオプティマイザーのインスタンス化
    optimizer = keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    # ロス関数のインスタンス化
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

## Sweep を初期化しエージェントを実行する

```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count` パラメーターを使用して総 run 数を制限できます。このスクリプトを素早く実行するために、実行回数を 10 に制限しますが、実行回数を増やしてみて何が起こるか確認してください。

```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

## 結果を視覚化する

上記の **Sweep URL** リンクをクリックして、リアルタイムの結果を確認してください。

## 事例ギャラリー

Weights & Biases を使用して追跡および視覚化されたプロジェクトの事例を [ギャラリー →](https://app.wandb.ai/gallery) でご覧ください。

## ベストプラクティス

1. **Projects**: 複数の run をプロジェクトにログして比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスや交差検証フォールドの場合、各プロセスを run としてログし、それらをグループ化します。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインやプロダクションモデルを追跡するためにタグを追加します。
4. **Notes**: テーブル内でノートを入力し、run 間の変更を追跡します。
5. **Reports**: 同僚と進捗を共有し、ダッシュボードを作成し、ML プロジェクトのスナップショットを作成するための簡単なノートを記入します。

## 高度なセットアップ

1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): API キーを環境変数に設定し、管理されたクラスターでトレーニングを実行できるようにします。
2. [オフラインモード]({{< relref path="/support/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): Weights & Biases をプライベートクラウドやエアギャップされたサーバーにインストールします。学術機関から企業チームまでがローカルインストールを利用しています。