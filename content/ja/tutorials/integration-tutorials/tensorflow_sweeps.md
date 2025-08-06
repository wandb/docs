---
title: TensorFlow スイープ
menu:
  tutorials:
    identifier: tensorflow_sweeps
    parent: integration-tutorials
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb" >}}
W&B を使って機械学習の実験管理、データセットのバージョン管理、プロジェクトでの共同作業を行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="Benefits of using W&B" >}}

W&B Sweeps を使えば、ハイパーパラメーター最適化を自動化し、インタラクティブなダッシュボードでモデルの可能性を探求できます。

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="TensorFlow hyperparameter sweep results" >}}

## sweep を使う理由

* **素早いセットアップ**：数行のコードで W&B sweep を開始できます。
* **透明性**：プロジェクトページで使用アルゴリズムがすべて明示され、[コードはオープンソース](https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py)です。
* **高機能**：sweep はカスタマイズに柔軟で、複数台のマシンやノート PC でも簡単に実行できます。

詳しくは [Sweeps overview]({{< relref "/guides/models/sweeps/" >}}) をご覧ください。

## このノートブックで学べること

* W&B Sweep と TensorFlow のカスタムトレーニングループの始め方
* 画像分類タスクでベストなハイパーパラメーターの見つけ方

**注意**：_Step_ で始まるセクションはハイパーパラメーター sweep を実行するために必要なコードを示します。それ以外はシンプルな例のセットアップです。

## インストール・インポート・ログイン

### W&B のインストール

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
W&B を初めて使う場合やログインしていない場合、`wandb.login()` 実行後のリンクからサインアップ/ログインページへアクセスできます。
{{< /alert >}}

## データセットの準備

```python
# トレーニングデータセットを準備
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

## 分類器 MLP を構築

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

        # データセットのバッチごとにイテレート
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

        # 3. run.log() を使ってメトリクスを記録
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

## sweep の設定

sweep 設定の手順：
* 最適化したいハイパーパラメーターを定義
* 最適化手法を選択：`random`、`grid`、または `bayes`
* `bayes` の場合、`val_loss` の最小化などのゴールとメトリクスを設定
* `hyperband` を使うと、高パフォーマンスの run を残して早期終了

詳細は [sweep configuration guide]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}}) を参照してください。

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

`sweep_train` のような関数を作り、
`sweep` 開始前に `run.config()` でハイパーパラメーターを設定し、`train` を呼び出します。

```python
def sweep_train(config_defaults=None):
    # デフォルト値を設定
    config_defaults = {"batch_size": 64, "learning_rate": 0.01}
    # サンプルのプロジェクト名で wandb を初期化
    run = wandb.init(config=config_defaults)  # Sweep で上書きされます

    # 他のハイパーパラメーターも config へ追加
    run.config.epochs = 2
    run.config.log_step = 20
    run.config.val_log_step = 50
    run.config.architecture_name = "MLP"
    run.config.dataset_name = "MNIST"

    # tf.data で入力パイプラインを作成
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

    # モデルの初期化
    model = Model()

    # オプティマイザーをインスタンス化
    optimizer = keras.optimizers.SGD(learning_rate=run.config.learning_rate)
    # ロス関数をインスタンス化
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # メトリクスを用意
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

## sweep の初期化とエージェント実行

```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count` パラメータで run の数を制限できます。素早く確認したい場合は 10 を指定。必要に応じて増やしてください。

```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

## 結果の可視化

**Sweep URL** のリンクをクリックして、結果をリアルタイムで確認できます。


## 事例ギャラリー

W&B でトラッキング＆可視化されたプロジェクトの例は [Gallery](https://app.wandb.ai/gallery) をご覧ください。

## ベストプラクティス
1. **Projects**: 複数の run を 1 つの project に記録して比較できます。 `wandb.init(project="project-name")`
2. **Groups**: 複数プロセスや交差検証用フォールドごとに run を作成してグループ化できます。 `wandb.init(group='experiment-1')`
3. **Tags**: ベースラインやプロダクション用モデルをタグで管理できます。
4. **Notes**: run 間の変更を追跡するため、表にメモを記入しましょう。
5. **Reports**: 進捗ノート・同僚との共有・MLプロジェクトのダッシュボードやスナップショット作成に Reports を活用しましょう。

## 応用セットアップ
1. [環境変数]({{< relref "/guides/hosting/env-vars/" >}})：マネージドクラスターなどでトレーニングするための API キー設定に。
2. [オフラインモード]({{< relref "/support/kb-articles/run_wandb_offline.md" >}})
3. [オンプレミス]({{< relref "/guides/hosting/hosting-options/self-managed" >}})：プライベートクラウドやエアギャップ環境、自組織のインフラで W&B を導入できます。ローカルインストールはアカデミックやエンタープライズチームにもおすすめです。