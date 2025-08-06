---
title: TensorFlow
menu:
  tutorials:
    identifier: tensorflow
    parent: integration-tutorials
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb" >}}

## このノートブックで学べること

* TensorFlowパイプラインに W&B を簡単に統合して 実験管理 を行う方法
* `keras.metrics` を使ったメトリクスの計算方法
* 独自のトレーニングループで `wandb.log` を利用してメトリクスを記録する方法

{{< img src="/images/tutorials/tensorflow/dashboard.png" alt="dashboard" >}}

**注意**: _Step_ から始まるセクションが、W&B を既存のコードに統合するときに必要な部分です。それ以外は一般的なMNISTの例です。

```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
```

## インストール、インポート、ログイン

### W&B のインストール

```jupyter
%%capture
!pip install wandb
```

### W&B をインポートしてログイン

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
```

> 補足 : W&B を初めて使う場合や、ログインしていない場合は、`wandb.login()` 実行後に表示されるリンクからサインアップ／ログインページにアクセスできます。サインアップはワンクリックで完了します。

### データセットの準備

```python
# トレーニングデータセットを準備
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.data を使って入力パイプラインを作成
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)
```

## モデルとトレーニングループの定義

```python
def make_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)

    return keras.Model(inputs=inputs, outputs=outputs)
```

```python
def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    # 1つのトレーニングステップを実行
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, logits)

    return loss_value
```

```python
def test_step(x, y, model, loss_fn, val_acc_metric):
    # 1つの検証ステップを実行
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_value
```

## トレーニングループに `wandb.log` を追加する

```python
def train(
    train_dataset,
    val_dataset,
    model,
    optimizer,
    train_acc_metric,
    val_acc_metric,
    epochs=10,
    log_step=200,
    val_log_step=50,
):
    run = wandb.init(
        project="my-tf-integration",
        config={
            "epochs": epochs,
            "log_step": log_step,
            "val_log_step": val_log_step,
            "architecture": "MLP",
            "dataset": "MNIST",
        },
    )
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # データセットのバッチを繰り返し処理
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(
                x_batch_train,
                y_batch_train,
                model,
                optimizer,
                loss_fn,
                train_acc_metric,
            )
            train_loss.append(float(loss_value))

        # 各エポックの最後にバリデーションループを走査
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

        # run.log()を使ってメトリクスを記録
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

## トレーニングの実行

### `wandb.init()` を呼び出して run を開始

これによって 実験 を開始したことが認識され、一意のIDとダッシュボードが生成されます。

[公式ドキュメントを参照]({{< relref "/ref/python/sdk/functions/init" >}})

```python
# プロジェクト名とオプション設定でwandbを初期化。
# configの値を調整して、wandbダッシュボードで結果を確認してみて下さい。
config = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 64,
    "log_step": 200,
    "val_log_step": 50,
    "architecture": "CNN",
    "dataset": "CIFAR-10",
}

run = wandb.init(project='my-tf-integration', config=config)
config = run.config

# モデルを初期化。
model = make_model()

# モデルをトレーニングするためのオプティマイザーを作成。
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 損失関数を作成。
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# メトリクスを準備。
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

train(
    train_dataset,
    val_dataset, 
    model,
    optimizer,
    train_acc_metric,
    val_acc_metric,
    epochs=config.epochs, 
    log_step=config.log_step, 
    val_log_step=config.val_log_step,
)

run.finish()  # Jupyter/Colabで実行完了を通知する場合に利用！
```

### 結果を可視化する

[runページ]({{< relref "/guides/models/track/runs/#view-logged-runs" >}}) をクリックして、リアルタイムの結果を確認できます。

## Sweep 101

W&B Sweeps を使えば、ハイパーパラメーターの自動最適化や様々なモデルの探索が手軽にできます。

[W&B Sweeps でのハイパーパラメータ最適化を実演したColabノートブックはこちら](https://wandb.me/tf-sweeps-colab)

### W&B Sweeps を利用する利点

* **すぐに使える** : 数行のコードで W&B Sweeps を使えます。
* **透明性が高い** : 使用しているすべてのアルゴリズムを明示し、[コードはオープンソース](https://github.com/wandb/sweeps) です。
* **高機能** : Sweep は完全にカスタマイズ・設定可能。数十台のマシンにまたがって sweep を実行するのも、ノートPCで始めるのも簡単です。

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="Sweep result" >}}

## Example Gallery

W&B を使ってトラッキングと可視化を行った Projects の様々な事例を [Fully Connected →](https://wandb.me/fc) でご覧いただけます。

## ベストプラクティス
1. **Projects**: 1つの Project に複数の run を記録して比較しましょう。`wandb.init(project="project-name")`
2. **Groups**: 複数プロセスや交差検証の各分割には、run ごとに記録してグループ化しましょう。`wandb.init(group="experiment-1")`
3. **Tags**: 現在のベースラインやプロダクションモデルを追跡するためにタグを付加しましょう。
4. **Notes**: 表に自由記述を残して run 間の変更点を記録しましょう。
5. **Reports**: 進捗メモを素早く記録し、仲間と共有できます。ML Project のダッシュボードやスナップショットも作成可能です。

### 上級者向けセットアップ
1. [環境変数]({{< relref "/guides/hosting/env-vars/" >}}): APIキー を環境変数に設定すると、マネージドクラスタでトレーニングを実行できます。
2. [オフラインモード]({{< relref "/support/kb-articles/run_wandb_offline.md" >}})
3. [オンプレミス]({{< relref "/guides/hosting/hosting-options/self-managed" >}}): W&B をプライベートクラウドや、閉域サーバ、独自インフラにインストール可能です。大学研究からエンタープライズまで、ローカルインストールに対応しています。
4. [Artifacts]({{< relref "/guides/core/artifacts/" >}}): モデルやデータセットの追跡・バージョニング管理を効率良く。パイプラインステップも自動で記録されます。