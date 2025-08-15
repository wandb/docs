---
title: TensorFlow Sweeps
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-tensorflow_sweeps
    parent: integration-tutorials
weight: 5
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb" >}}
W&B で機械学習の実験管理、データセットのバージョン管理、プロジェクトの共同作業を効率化しましょう。
{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使う利点" >}}

W&B Sweeps を使えば、ハイパーパラメーター探索を自動化し、インタラクティブなダッシュボードでモデルの可能性を探ることができます。

{{< img src="/images/tutorials/tensorflow/sweeps.png" alt="TensorFlow ハイパーパラメータースイープの結果" >}}

## sweeps を使う理由

* **すぐに使える**: W&B sweeps は数行のコードで実行可能です。
* **透明性**: プロジェクト内で利用されたアルゴリズムすべてを明示し、[コードはオープンソース](https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py)です。
* **強力**: Sweeps はカスタマイズ性が高く、複数マシンやノートパソコンでも簡単に実行できます。

詳細は [Sweeps overview]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) をご覧ください。

## このノートブックで学べること

* W&B Sweep の始め方と TensorFlow でのカスタムトレーニングループ構築の手順
* 画像分類タスクにおける最適なハイパーパラメーターの見つけ方

**注意**: _Step_ で始まるセクションにはハイパーパラメーター探索に必要なコードが記載されています。それ以外の部分はシンプルな実例の準備を行っています。

## インストール、インポート、ログイン

### W&B のインストール

```bash
pip install wandb
```

### W&B のインポートとログイン

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
W&B を初めて使う場合やログインしていない場合は、`wandb.login()` を実行後に表示されるリンクからサインアップ／ログインしてください。
{{< /alert >}}

## データセットの準備

```python
# トレーニングデータセットの準備
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

## 分類器 MLP の構築

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

        # 各エポック終了時にバリデーションループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(
                x_batch_val, y_batch_val, model, loss_fn, val_acc_metric
            )
            val_loss.append(float(val_loss_value))

        # エポック終了時にメトリクスを表示
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # エポック終了時にメトリクスをリセット
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # 3. run.log() でメトリクスを記録
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

sweep の設定手順:
* 最適化するハイパーパラメーターを定義
* 最適化メソッドを選択: `random`, `grid`, または `bayes`
* `bayes` にはゴールとメトリクスを設定（例: `val_loss` の最小化）
* 途中終了には `hyperband` を利用して効率的な run を実行

さらに詳しくは [sweep configuration ガイド]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) をご参照ください。

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

## トレーニングループのラップ

例えば `sweep_train` のような関数を作り、内で `run.config()` でハイパーパラメーターをセットして `train` を実行します。

```python
def sweep_train(config_defaults=None):
    # デフォルト値を設定
    config_defaults = {"batch_size": 64, "learning_rate": 0.01}
    # 仮のプロジェクト名で wandb を初期化
    run = wandb.init(config=config_defaults)  # Sweeps で上書きされます

    # その他のハイパーパラメーターを設定
    run.config.epochs = 2
    run.config.log_step = 20
    run.config.val_log_step = 50
    run.config.architecture_name = "MLP"
    run.config.dataset_name = "MNIST"

    # tf.data を用いて入力パイプラインを作成
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

    # オプティマイザーの設定
    optimizer = keras.optimizers.SGD(learning_rate=run.config.learning_rate)
    # 損失関数の設定
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # メトリクスの準備
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

## sweep を初期化してエージェントを実行

```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count` パラメータで run 数を制限できます。素早く試すために 10 に設定しています。必要に応じて増減してください。

```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

## 結果の可視化

**Sweep URL** のリンクをクリックすると、ライブの結果が確認できます。

## ギャラリー事例

[Gallery](https://app.wandb.ai/gallery) で、W&B でトラッキング・可視化されたプロジェクトの例を見てみましょう。

## ベストプラクティス
1. **Projects**: 複数の run をひとつのプロジェクトに記録して比較しましょう。`wandb.init(project="project-name")`
2. **Groups**: 複数プロセスやクロスバリデーションのfoldごとに run を分けて記録し、グループ化しましょう。`wandb.init(group='experiment-1')`
3. **Tags**: ベースラインやプロダクションモデルを追跡するためにタグを活用しましょう。
4. **Notes**: テーブルの notes 欄に変更点を記録して、run ごとの差分を明確にしましょう。
5. **Reports**: レポートで進捗をまとめたり、同僚との共有、MLプロジェクトのダッシュボードやスナップショット作成にも活用できます。

## 高度な設定
1. [環境変数]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}): マネージドクラスターでトレーニングする際に APIキー を設定しましょう。
2. [オフラインモード]({{< relref path="/support/kb-articles/run_wandb_offline.md" lang="ja" >}})
3. [オンプレミス]({{< relref path="/guides/hosting/hosting-options/self-managed" lang="ja" >}}): プライベートクラウドや、自前インフラのエアギャップサーバーへ W&B をインストール。ローカルインストールはアカデミックや企業チームに最適です。