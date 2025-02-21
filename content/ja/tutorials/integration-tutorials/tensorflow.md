---
title: TensorFlow
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow
    parent: integration-tutorials
weight: 4
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb" >}}

Weights & Biases を使用して、機械学習の 実験管理 、データセット の バージョン管理 、および プロジェクト の コラボレーションを行います。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

## この ノートブック の内容

*   Weights & Biases と TensorFlow パイプライン を簡単に 統合して、 実験管理 を行います。
*   `keras.metrics` で メトリクス を計算します。
*   `wandb.log` を使用して、カスタム トレーニング ループ でこれらの メトリクス を ログ に記録します。

{{< img src="/images/tutorials/tensorflow/dashboard.png" alt="dashboard" >}}

**注意**: _Step_ で始まるセクションは、W&B を既存の コード に 統合するために必要なすべてです。残りは、標準的な MNIST の例です。

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

### W&B のインポート と ログイン

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
```

> 補足: W&B を初めて使用する場合、または ログイン していない場合は、`wandb.login()` を実行した後に表示される リンク からサインアップ/ログイン ページに移動します。サインアップ は ワンクリック で簡単にできます。

### データセット の準備

```python
# トレーニングデータセットを準備します。
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.data を使用して 入力 パイプライン を構築します。
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)
```

## モデル と トレーニング ループ の定義

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

## トレーニング ループ に `wandb.log` を追加

```python
def train(train_dataset, val_dataset,  model, optimizer,
          train_acc_metric, val_acc_metric,
          epochs=10,  log_step=200, val_log_step=50):
  
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # データセット の バッチ を反復処理します。
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # 各 epoch の最後に 検証 ループ を実行します。
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 各 epoch の最後に メトリクス を表示します。
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各 epoch の最後に メトリクス を リセット します。
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # ⭐: wandb.log を使用して メトリクス を ログ に記録します。
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
```

## トレーニング の実行

### `wandb.init` を呼び出して run を開始します。

これにより、 実験 を開始したことが通知され、一意の ID と ダッシュボード が提供されます。

[公式 ドキュメント を確認してください]({{< relref path="/ref/python/init" lang="ja" >}})

```python
# プロジェクト 名とオプションで 構成 で wandb を 初期化 します。
# 構成 の 値 を色々試して、wandb ダッシュボード で 結果 を確認してください。
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

# モデル を 初期化 します。
model = make_model()

# モデル を トレーニング するための オプティマイザー を インスタンス化 します。
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 損失関数 を インスタンス化 します。
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# メトリクス を準備します。
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

run.finish()  # Jupyter/Colab で、完了したことをお知らせください!
```

### 結果 の 可視化

上記の [**run ページ**]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) リンク をクリックして、ライブ の 結果 を確認してください。

## Sweep 101

Weights & Biases Sweeps を使用して、 ハイパーパラメーター の 最適化 を自動化し、可能な モデル の 空間 を探索します。

## [W&B Sweeps を使用した TensorFlow での ハイパーパラメーター の 最適化 を確認してください](http://wandb.me/tf-sweeps-colab)

### W&B Sweeps を使用する 利点

*   **簡単な セットアップ**: 数行の コード だけで W&B sweeps を実行できます。
*   **透過的**: 使用しているすべての アルゴリズム を引用し、[コード は オープンソース です](https://github.com/wandb/sweeps)。
*   **強力**: sweeps は完全にカスタマイズ可能で、構成可能です。数十台の マシン で sweep を 起動でき、 ラップトップ で sweep を開始するのと同じくらい簡単です。

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="Sweep result" >}}

## サンプル ギャラリー

W&B で トラッキング および 可視化 された プロジェクト の 例については、 サンプル の ギャラリー を参照してください。[完全に 接続 →](https://wandb.me/fc)

# 📏 ベストプラクティス

1.  **Projects**: 複数の run を プロジェクト に ログ して、それらを比較します。`wandb.init(project="project-name")`
2.  **Groups**: 複数の プロセス または 交差検証 の folds については、各 プロセス を run として ログ し、それらを グループ化 します。`wandb.init(group='experiment-1')`
3.  **Tags**: 現在の ベースライン または プロダクション モデル を追跡するために タグ を追加します。
4.  **Notes**: テーブル に ノート を入力して、run 間の変更を追跡します。
5.  **Reports**: 同僚と共有するために 進捗状況 について簡単な メモ を取り、ML プロジェクト の ダッシュボード と スナップショット を作成します。

## 高度な セットアップ

1.  [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): マネージド クラスター で トレーニング を実行できるように、環境変数 に APIキー を設定します。
2.  [オフライン モード]({{< relref path="/support/run_wandb_offline.md" lang="ja" >}})
3.  [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): 独自の インフラストラクチャー の プライベートクラウド または エアギャップ サーバー に W&B を インストール します。 学術関係者から エンタープライズ チーム まで、あらゆる人に対応できる ローカル インストール があります。
4.  [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): モデル を トレーニング する際に パイプライン ステップ を自動的に 取得する 合理化された方法で、 モデル と データセット を追跡および バージョン管理 します。
