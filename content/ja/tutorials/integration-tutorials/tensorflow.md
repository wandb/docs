---
title: TensorFlow
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow
    parent: integration-tutorials
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb" >}}

## このノートブックで学べること

* W&B を TensorFlow パイプラインに簡単に組み込んで実験管理を行う方法
* `keras.metrics` を使ったメトリクスの計算
* `wandb.log` を利用してカスタムトレーニングループでメトリクスをログする方法

{{< img src="/images/tutorials/tensorflow/dashboard.png" alt="dashboard" >}}

**注意:** _Step_ で始まるセクションだけ読めば、既存コードに W&B を統合できます。それ以外の部分は標準的な MNIST の例です。

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

> 補足: 初めて W&B を使う場合や未ログインの場合、`wandb.login()` 実行時に表示されるリンクからサインアップ／ログインページにアクセスできます。サインアップはワンクリックで簡単です。

### データセットの準備

```python
# トレーニングデータセットを準備
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.data で入力パイプラインを構築
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
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)

    return loss_value
```

## トレーニングループに `wandb.log` を追加

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

        # データセットのバッチごとにイテレート
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

        # run.log() を使ってメトリクスを記録
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

これにより、実験の開始を W&B に通知でき、ユニークな ID とダッシュボードが作成されます。

[公式ドキュメントはこちら]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}})

```python
# プロジェクト名と（必要なら）設定情報を指定して wandb を初期化
# config の値を変えて、ダッシュボードで結果を確認してみましょう
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

# モデルを初期化
model = make_model()

# モデルを学習させるためのオプティマイザーをインスタンス化
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
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
    train_acc_metric,
    val_acc_metric,
    epochs=config.epochs, 
    log_step=config.log_step, 
    val_log_step=config.val_log_step,
)

run.finish()  # Jupyter／Colab で完了したことを知らせましょう！
```

### 結果を可視化

上記の [run ページ]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) リンクをクリックして、リアルタイムの結果を確認できます。

## Sweep 101

W&B Sweeps を使うと、ハイパーパラメータ最適化を自動化してモデルの探索が可能になります。

[W&B Sweeps でハイパーパラメータ最適化する Colab ノートブックの例を見る](https://wandb.me/tf-sweeps-colab)

### W&B Sweeps を利用する利点

* **すぐに使える:** 数行のコードで W&B Sweeps を実行可能
* **透明性:** 使用しているアルゴリズムはすべて明記しており、[コードはオープンソース](https://github.com/wandb/sweeps) です
* **パワフル:** 完全カスタマイズ・設定が可能。何十台ものマシンで sweep をスタートするのも自分のノートパソコンで始めるのも同じくらい簡単です

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="Sweep result" >}}

## 例ギャラリー

W&B で管理・可視化された Projects の事例を [Fully Connected →](https://wandb.me/fc) でチェックできます。

## ベストプラクティス
1. **Projects**: 複数の run をひとつの Project に記録して比較しましょう。`wandb.init(project="project-name")`
2. **Groups**: 複数プロセスや交差検証など、各プロセスを run として記録し group でまとめて管理しましょう。`wandb.init(group="experiment-1")`
3. **Tags**: 現在のベースラインやプロダクションモデルの追跡にはタグを追加しましょう。
4. **Notes**: テーブル内の notes で run ごとの変更内容を記録できます。
5. **Reports**: 進捗のメモやダッシュボード、スナップショットも Reports で手軽に共有できます。

### 発展的なセットアップ
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): 環境変数で APIキー を設定し、管理クラスター等でのトレーニングを可能にします。
2. [オフラインモード]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): プライベートクラウドや社内インフラのエアギャップサーバーに W&B をインストール。学術からエンタープライズまで幅広くローカル構築できます。
4. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): モデルやデータセットのバージョン管理も簡単。パイプラインの各ステップを自動認識し、モデルの学習に活用できます。