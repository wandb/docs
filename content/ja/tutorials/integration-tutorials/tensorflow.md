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

* W&B と TensorFlow パイプラインで 実験管理 を簡単に行う方法。
* `keras.metrics` を使って メトリクス を計算する方法
* カスタム トレーニング ループで `wandb.log` を使ってメトリクスをログする方法

{{< img src="/images/tutorials/tensorflow/dashboard.png" alt="ダッシュボード" >}}

**注意**: _Step_ で始まるセクションだけ読めば、既存の コード に W&B を組み込めます。残りは標準的な MNIST の例です。

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

> 参考: 初めて W&B を使う場合や未ログインの場合、`wandb.login()` 実行後に表示されるリンクからサインアップ／ログイン ページへ移動します。サインアップはワンクリックで完了します。

### データセットの準備

```python
# トレーニング データセットを準備
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.data を使って入力パイプラインを構築
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)
```

## モデルとトレーニング ループを定義

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

## トレーニング ループに `wandb.log` を追加


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

        # データセットのバッチを反復処理
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

        # 各 エポック の最後に検証ループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # 各 エポック の最後にメトリクスを表示
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各 エポック の最後にメトリクスをリセット
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # run.log() でメトリクスをログ
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

## トレーニング を実行

### Run を開始するために `wandb.init()` を呼び出す

これにより、 実験 を開始したことが W&B に伝わり、一意の ID と ダッシュボード を割り当てます。

[公式ドキュメントをチェック]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}})

```python
# プロジェクト名と必要に応じて 設定 を指定して wandb を初期化します。
# config の値をいじって、wandb の ダッシュボード 上で結果を確認してみましょう。
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

# モデルを学習させるための オプティマイザー を用意。
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 損失関数を用意。
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# メトリクスを用意。
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

run.finish()  # Jupyter／Colab では、終了したことを W&B に知らせましょう！
```

### 結果を可視化

上の [Run ページ]({{< relref path="/guides/models/track/runs/#view-logged-runs" lang="ja" >}}) リンクをクリックすると、ライブの結果が見られます。

## Sweep 101

W&B Sweeps を使って ハイパーパラメーター 最適化を自動化し、取り得る モデル の探索空間を調べましょう。

W&B Sweeps を使ったハイパーパラメーター最適化を紹介する [Colab ノートブック](https://wandb.me/tf-sweeps-colab) をご覧ください。

### W&B Sweeps を使う利点

* **すぐに始められる**: 数行の コード で W&B Sweeps を実行できます。
* **透明性**: 使用しているアルゴリズムはすべて明記しており、[コードはオープンソースです](https://github.com/wandb/sweeps)。
* **強力**: Sweep は完全にカスタマイズ可能・設定可能です。数十台のマシンにまたがる Sweep も、ノート PC で始めるのと同じくらい簡単に起動できます。

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="Sweep の結果" >}}

## サンプル ギャラリー

W&B でトラッキングして可視化したプロジェクトの例を、ギャラリーでご覧ください。[Fully Connected →](https://wandb.me/fc).

## ベストプラクティス
1. **Projects**: 複数の Runs を 1 つの Project にログして比較しましょう。 `wandb.init(project="project-name")`
2. **グループ**: 複数の プロセス や 交差検証 の各フォールドについて、各プロセスを個別の Run としてログし、まとめてグループ化します。 `wandb.init(group="experiment-1")`
3. **タグ**: 現在の ベースライン や プロダクション モデル をトラッキングするためにタグを追加します。
4. **ノート**: テーブルにノートを入力して、Runs 間の変更点を追跡しましょう。
5. **Reports**: 進捗を手早くメモして同僚と共有し、ML プロジェクトの ダッシュボード と スナップショット を作成しましょう。

### 高度なセットアップ
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): マネージドな クラスター で トレーニング を実行できるよう、環境変数に APIキー を設定します。
2. [オフライン モード]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): 自社の プライベートクラウド やエアギャップ環境の サーバー に W&B をインストールします。学術研究からエンタープライズまで、幅広いローカル導入に対応しています。
4. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): モデル と データセット をシームレスに追跡・バージョン管理できます。学習中に パイプライン のステップを自動的に検出します。