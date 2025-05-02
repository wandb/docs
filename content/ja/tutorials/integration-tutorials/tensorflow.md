---
title: TensorFlow
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow
    parent: integration-tutorials
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb" >}}

## このノートブックでカバーする内容

* Weights & Biases と TensorFlow パイプラインの簡単なインテグレーションによる実験管理。
* `keras.metrics` を使用したメトリクスの計算
* `wandb.log` を使用して、カスタムトレーニングループでこれらのメトリクスをログに記録する方法。

{{< img src="/images/tutorials/tensorflow/dashboard.png" alt="ダッシュボード" >}}

**注**: _ステップ_ から始まるセクションは、既存のコードに W&B を統合するために必要なすべてです。それ以外は通常の MNIST の例です。

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

### W&B のインポートとログイン

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
```

> サイドノート: もしこれが初めて W&B を使う場合や、まだログインしていない場合、`wandb.login()` 実行後に表示されるリンクはサインアップ/ログインページに移動します。サインアップはワンクリックで簡単です。

### データセットの準備

```python
# トレーニング用データセットの準備
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.data を使用して入力パイプラインを構築
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
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []
        val_loss = []

        # データセットのバッチに対して繰り返し処理
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

        # 各エポックの終了時に検証ループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # 各エポックの終了時にメトリクスを表示
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポックの終了時にメトリクスをリセット
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # ⭐: wandb.log を使用してメトリクスをログに記録
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

## トレーニングを実行

### `wandb.init` を呼び出して run を開始

これにより、実験を起動したことがわかり、ユニークな ID とダッシュボードを提供できます。

[公式ドキュメントをチェック]({{< relref path="/ref/python/init" lang="ja" >}})

```python
# プロジェクト名で wandb を初期化し、設定をオプションで指定します。
# 設定の値を変えて、wandb ダッシュボードでの結果を確認してください。
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

# モデルの初期化
model = make_model()

# モデルをトレーニングするためのオプティマイザーをインスタンス化
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

run.finish()  # Jupyter/Colab では、完了したことを知らせます！
```

### 結果を可視化

上記の [**run ページ**]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) リンクをクリックして、ライブ結果を確認してください。

## Sweep 101

Weights & Biases Sweeps を使用してハイパーパラメータの最適化を自動化し、可能なモデルのスペースを探索しましょう。

## [Weights & Biases Sweeps を使用した TensorFlow におけるハイパーパラメータ最適化をチェック](http://wandb.me/tf-sweeps-colab)

### W&B Sweeps を使用するメリット

* **簡単なセットアップ**: 数行のコードで W&B sweeps を実行できます。
* **透明性**: 使用するアルゴリズムをすべて引用しており、[コードはオープンソース](https://github.com/wandb/sweeps)です。
* **強力**: スイープは完全にカスタマイズ可能で設定可能です。何十台ものマシンでスイープを起動することができ、それはラップトップでスイープを開始するのと同じくらい簡単です。

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="スイープ結果" >}}

## サンプルギャラリー

W&B を使って記録・可視化されたプロジェクトの例を見ることができます。[Fully Connected →](https://wandb.me/fc)

## ベストプラクティス
1. **Projects**: 複数の runs をプロジェクトにログして、それらを比較します。 `wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスや交差検証フォールドの場合は、それぞれのプロセスを個別の run としてログし、まとめてグループ化します。 `wandb.init(group="experiment-1")`
3. **Tags**: 現在のベースラインやプロダクションモデルを追跡するためにタグを追加します。
4. **Notes**: テーブルにメモを入力して、runs 間の変更を追跡します。
5. **Reports**: 進捗について同僚と共有するために迅速にメモを取り、ML プロジェクトのダッシュボードとスナップショットを作成します。

### 高度なセットアップ
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): APIキーを環境変数に設定して、管理されたクラスターでトレーニングを実行できるようにします。
2. [オフラインモード]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): W&B をプライベートクラウドやエアギャップサーバーのあなたのインフラストラクチャ上にインストールします。学術的なユーザーから企業間のチームまで、みんなのためにローカルインストールを提供しています。
4. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): モデルとデータセットを一元化された方法で追跡し、バージョン管理することで、モデルをトレーニングする際にパイプラインステップを自動的にキャプチャします。