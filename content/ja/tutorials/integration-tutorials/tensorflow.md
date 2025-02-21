---
title: TensorFlow
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow
    parent: integration-tutorials
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb" >}}

Weights & Biases を使用して 機械学習 の 実験管理 、データセットのバージョン管理、およびプロジェクト コラボレーションを行います。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

## このノートブックでカバーする内容

* TensorFlow パイプライン への Weights and Biases の簡単なインテグレーションによる 実験管理 。
* `keras.metrics` を使用してメトリクスを計算します
* `wandb.log` を使用して、カスタム トレーニング ループにそれらのメトリクスをログします。


{{< img src="/images/tutorials/tensorflow/dashboard.png" alt="dashboard" >}}

**注意**: _Step_ から始まるセクションは、W&B を既存のコードに統合するために必要なすべての内容です。それ以外は標準的な MNIST の例です。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## インストール、インポート、ログイン

### W&B のインストール


```python
%%capture
!pip install wandb
```

### W&B をインポートしてログイン


```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
```

> 補足: 初めて W&B を使用するか、ログインしていない場合は、`wandb.login()` を実行した後に表示されるリンクでサインアップ/ログインページにアクセスできます。サインアップはワンクリックで簡単です。

### データセットの準備


```python
# トレーニング データセットを準備
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

## モデルと トレーニング ループを定義


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

## `wandb.log` をあなたの トレーニング ループに追加


```python
def train(train_dataset, val_dataset,  model, optimizer,
          train_acc_metric, val_acc_metric,
          epochs=10,  log_step=200, val_log_step=50):
  
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # データセットのバッチを繰り返し処理
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # 各エポックの最後に検証ループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 各エポックの最後にメトリクスを表示
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポックの最後にメトリクスをリセット
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # ⭐: wandb.log を使用してメトリクスをログ
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
```

## トレーニングの実行

### `wandb.init` を呼び出して run を開始

これにより、あなたが実験を開始したことを知らせることができ、ユニークなIDとダッシュボードを提供します。

[公式ドキュメントを確認]({{< relref path="/ref/python/init" lang="ja" >}})

```python
# プロジェクト名と、オプションで設定を指定して wandb を初期化
# 設定値をいじって、wandb ダッシュボードで結果を確認してください
config = {
              "learning_rate": 0.001,
              "epochs": 10,
              "batch_size": 64,
              "log_step": 200,
              "val_log_step": 50,
              "architecture": "CNN",
              "dataset": "CIFAR-10"
           }

run = wandb.init(project='my-tf-integration', config=config)
config = wandb.config

# モデルを初期化
model = make_model()

# モデルを学習させるためのオプティマイザーをインスタンス化
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 損失関数をインスタンス化
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# メトリクスを準備
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

train(train_dataset,
      val_dataset, 
      model,
      optimizer,
      train_acc_metric,
      val_acc_metric,
      epochs=config.epochs, 
      log_step=config.log_step, 
      val_log_step=config.val_log_step)

run.finish()  # Jupyter/Colab では、終了したことを知らせてください！
```

### 結果の可視化

上記の [**run ページ**]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) リンクをクリックして、ライブの結果を確認してください。

## Sweep 101

Weights & Biases Sweeps を使用して、 ハイパーパラメーター のオプティマイゼーションを自動化し、可能性のあるモデルの空間を探索します。

## [TensorFlowでのW&Bスイープを使用したハイパーパラメーターのオプティマイゼーションをチェック](http://wandb.me/tf-sweeps-colab)

### W&B Sweeps を使用する利点

* **クイックセットアップ**: わずか数行の コード で W&B スイープを実行できます。
* **透明性**: 使用しているすべてのアルゴリズムを引用し、[コードはオープンソース](https://github.com/wandb/sweeps) です。
* **強力**: スイープは完全にカスタマイズ可能で設定可能です。数十台のマシンでスイープを開始することができ、ノートパソコンでスイープを開始するのと同じくらい簡単です。

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="Sweep result" >}}

## ギャラリー例

プロジェクトが W&B でトラッキングされ、可視化されている例は例のギャラリーで確認できます [Fully Connected →](https://wandb.me/fc)

# 📏 ベストプラクティス
1. **Projects**: 複数の run をプロジェクトにログして比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスまたは交差検証フォールドの場合は、各プロセスを run としてログし、それらをグループ化します。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインまたはプロダクションモデルをトラックするためにタグを追加します。
4. **Notes**: テーブル内でメモを入力して、run 間の変更をトラックします。
5. **Reports**: 進捗について同僚と共有するためにクイックメモを取ったり、MLプロジェクトのダッシュボードとスナップショットを作成したりします。

## 高度な設定
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): マネージドクラスターでトレーニングを実行できるように、環境変数に APIキー を設定します。
2. [オフラインモード]({{< relref path="/support/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): W&B をプライベートクラウドやエアギャップのサーバーにインストールします。学術関係者からエンタープライズチームまで、ローカルインストールを提供しています。
4. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): パイプラインステップをトレーニング中に自動的に取り込むように設計された、モデルとデータセットのトラッキングとバージョン管理を行います。