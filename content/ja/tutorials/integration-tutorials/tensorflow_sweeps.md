---
title: TensorFlow Sweeps
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow_sweeps
    parent: integration-tutorials
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb" >}}
Weights & Biases を使用して、機械学習 の 実験管理 、 データセット の バージョン管理 、および プロジェクト のコラボレーションを行います。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

Weights & Biases の Sweeps を使用して、 ハイパーパラメーター の 最適化を自動化し、インタラクティブな ダッシュボード を備えた、可能な モデル の空間を探索します。

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="" >}}

## Sweeps を使用する理由

* **簡単なセットアップ**: 数行の コード だけで、W&B の sweeps を実行できます。
* **透明性**: プロジェクト は、使用されているすべての アルゴリズム を引用し、[コード は オープンソース](https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py)です。
* **強力**: Sweeps は完全にカスタマイズおよび構成可能です。数十台のマシンで sweep を 起動 できます。ラップトップで sweep を開始するのと同じくらい簡単です。

**[公式ドキュメント を ご覧ください]({{< relref path="/guides/models/sweeps/" lang="ja" >}})**

## この ノートブック の内容

* TensorFlow の カスタム トレーニング ループ で W&B Sweep を開始するための簡単なステップ。
* 画像分類タスク に最適な ハイパーパラメーター を見つけます。

**注意**: _Step_ で始まるセクションは、既存の コード で ハイパーパラメーター sweep を実行するために必要なすべてです。
残りの コード は、簡単な例を設定するためにあります。

## インストール、インポート、および ログイン

### W&B のインストール

```bash
%%capture
!pip install wandb
```

### W&B のインポート と ログイン

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

> 補足: W&B を初めて使用する場合、または ログイン していない場合は、`wandb.login()` を実行した後に表示される リンク からサインアップ / ログイン ページに移動します。サインアップ は数回クリックするだけで簡単です。

## データセット の準備

```python
# トレーニングデータセット の準備
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

## 簡単な 分類器 MLP を構築する

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

## トレーニング ループ の作成

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

        # データセット の バッチ を 反復処理します
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

        # 各 エポック の最後に 検証 ループ を実行します
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # 各 エポック の最後に メトリクス を表示します
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各 エポック の最後に メトリクス を リセット します
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # 3️⃣ wandb.log を使用して メトリクス を ログ に記録する
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

## Sweep の構成

ここでは、次のことを行います。
* sweep する ハイパーパラメーター を定義します。
* ハイパーパラメーター 最適化 メソッド を提供します。`random`、`grid`、`bayes` の メソッド があります。
* `bayes` を使用している場合は、目的 と `metric` を指定します。たとえば、`val_loss` を `minimize` します。
* パフォーマンス の低い run を早期に終了させるには、`hyperband` を使用します。

#### [Sweep Configs の詳細はこちら]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})

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

## トレーニング ループ をラップする

`sweep_train` のような 関数 が必要になります。
`wandb.config` を使用して ハイパーパラメーター を設定します。
`train` が呼び出される前に。

```python
def sweep_train(config_defaults=None):
    # デフォルト値 を 設定します
    config_defaults = {"batch_size": 64, "learning_rate": 0.01}
    # サンプル プロジェクト 名 で wandb を 初期化 します
    wandb.init(config=config_defaults)  # これは Sweep で 上書き されます

    # その他の ハイパーパラメーター を 構成 に指定します (存在する場合)
    wandb.config.epochs = 2
    wandb.config.log_step = 20
    wandb.config.val_log_step = 50
    wandb.config.architecture_name = "MLP"
    wandb.config.dataset_name = "MNIST"

    # tf.data を 使用して 入力 パイプライン を構築します
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

    # モデル を 初期化 します
    model = Model()

    # モデル を トレーニング するための オプティマイザー を インスタンス化 します。
    optimizer = keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    # 損失関数 を インスタンス化 します。
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # メトリクス を準備します。
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

## Sweep の初期化 と エージェント の実行

```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count` パラメーター で 合計 run 数を制限できます。スクリプト の実行を高速化するために 10 に制限します。run 数を増やして何が起こるかを確認してください。

```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

## 結果 の 可視化

ライブ の 結果 を表示するには、上記の **Sweep URL** リンク をクリックしてください。

## ギャラリー の例

W&B で トラッキング および 可視化 された プロジェクト の例については、[ギャラリー →](https://app.wandb.ai/gallery) を参照してください。

## ベストプラクティス
1. **Projects**: 複数の run を プロジェクト に ログ して、それらを比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数の プロセス または 交差検証 folds の場合、各 プロセス を run として ログ に記録し、それらを グループ化 します。`wandb.init(group='experiment-1')`
3. **Tags**: 現在の ベースライン または プロダクション モデル を トラッキング するために タグ を追加します。
4. **Notes**: テーブル に メモ を入力して、run 間の変更を トラッキング します。
5. **Reports**: 同僚と共有するために 進捗状況 に関する 簡単な メモ を作成し、ML プロジェクト の ダッシュボード と スナップショット を作成します。

## 高度な設定
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): マネージド クラスター で トレーニング を実行できるように、環境変数 に APIキー を設定します。
2. [オフライン モード]({{< relref path="/support/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): プライベートクラウド または 独自の インフラストラクチャー 内の エアギャップ サーバー に W&B をインストールします。 学術関係者から エンタープライズ チーム まで、誰もが ローカル インストール を使用しています。
