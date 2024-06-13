
# TensorFlow Sweeps

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb)

Weights & Biasesを使用して、機械学習の実験の追跡、データセットのバージョン管理、およびプロジェクトのコラボレーションを行いましょう。

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

Weights & Biases Sweepsを使用してハイパーパラメーターの最適化を自動化し、可能なモデルの空間を探索します。以下のようなインタラクティブなダッシュボード付きです。

![](https://i.imgur.com/AN0qnpC.png)

## 🤔 Sweepsを使う理由

* **簡単なセットアップ**: わずか数行のコードでW&B Sweepsを実行できます。
* **透明性**: 当社が使用しているすべてのアルゴリズムを引用しており、[コードはオープンソースです](https://github.com/wandb/client/tree/master/wandb/sweeps)。
* **強力**: 当社のSweepsは完全にカスタマイズ可能かつ設定可能です。多数のマシンでスイープを実行することができ、ラップトップでスイープを開始するのと同じくらい簡単です。

**[公式ドキュメントをチェックする $\rightarrow$](https://docs.wandb.com/sweeps)**

## このノートブックでカバーする内容

* TensorFlowでのカスタムトレーニングループを使用して、W&B Sweepを始めるための簡単な手順。
* 画像分類タスクに最適なハイパーパラメーターを見つけること。

**注**: _Step_ で始まるセクションは、既存のコードでハイパーパラメーター探索を実行するために必要なすべてです。それ以外のコードは単純な例を設定するためのものです。

# 🚀 インストール、インポート、ログイン

### Step 0️⃣: W&Bのインストール

```python
%%capture
!pip install wandb
```

### Step 1️⃣: W&Bのインポートとログイン

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
from wandb.keras import WandbCallback

wandb.login()
```

> サイドノート: 初めてW&Bを使用する場合、またはログインしていない場合、`wandb.login()` を実行した後に表示されるリンクからサインアップ/ログインページにアクセスできます。サインアップは数回のクリックで完了します。

# 👩‍🍳 データセットの準備

```python
# トレーニングデータセットの準備
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

# 🧠 モデルとトレーニングループの定義

## 🏗️ シンプルな分類器MLPの構築

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

## 🔁 トレーニングループの作成

### Step 3️⃣: `wandb.log`を使ってメトリクスをログする

```python
def train(train_dataset,
          val_dataset, 
          model,
          optimizer,
          loss_fn,
          train_acc_metric,
          val_acc_metric,
          epochs=10, 
          log_step=200, 
          val_log_step=50):
  
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # データセットのバッチを繰り返し処理する
        for step, (x_batch_train, y_batch_train) in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # 各エポックの終わりに検証ループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, x_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 各エポックの終わりにメトリクスを表示
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポックの終わりにメトリクスをリセット
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # 3️⃣ wandb.logを使ってメトリクスをログする
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc': float(val_acc)})
```

# Step 4️⃣: Sweepの設定

ここでは以下を行います。
* スイープするハイパーパラメーターを定義
* ハイパーパラメータ最適化メソッドを提供。`random`、`grid`、`bayes`メソッドがあります。
* 目的とする`metric`および`bayes`を使用する場合、例えば`val_loss`を`minimize`する。
* パフォーマンスの低いrunの早期終了には`hyperband`を使用

#### [Sweep Configsの詳細はこちら→](https://docs.wandb.com/sweeps/configuration)

```python
sweep_config = {
  'method': 'random', 
  'metric': {
      'name': 'val_loss',
      'goal': 'minimize'
  },
  'early_terminate':{
      'type': 'hyperband',
      'min_iter': 5
  },
  'parameters': {
      'batch_size': {
          'values': [32, 64, 128, 256]
      },
      'learning_rate':{
          'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]
      }
  }
}
```

# Step 5️⃣: トレーニングループをラップする

以下のように、`train`が呼び出される前にハイパーパラメーターを設定するために`wandb.config`を使用する関数が必要です。

```python
def sweep_train(config_defaults=None):
    # デフォルト値を設定
    config_defaults = {
        "batch_size": 64,
        "learning_rate": 0.01
    }
    # サンプルプロジェクト名でwandbを初期化
    wandb.init(config=config_defaults)  # これはSweepで上書きされます

    # 他のハイパーパラメーターを設定に指定
    wandb.config.epochs = 2
    wandb.config.log_step = 20
    wandb.config.val_log_step = 50
    wandb.config.architecture_name = "MLP"
    wandb.config.dataset_name = "MNIST"

    # tf.dataを使用して入力パイプラインを構築
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset.shuffle(buffer_size=1024)
                                  .batch(wandb.config.batch_size)
                                  .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = (val_dataset.batch(wandb.config.batch_size)
                              .prefetch(buffer_size=tf.data.AUTOTUNE))

    # モデルを初期化
    model = Model()

    # モデルをトレーニングするためのオプティマイザーをインスタンス化
    optimizer = keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    # ロス関数をインスタンス化
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # メトリクスを準備
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    train(train_dataset,
          val_dataset, 
          model,
          optimizer,
          loss_fn,
          train_acc_metric,
          val_acc_metric,
          epochs=wandb.config.epochs, 
          log_step=wandb.config.log_step, 
          val_log_step=wandb.config.val_log_step)
```

# Step 6️⃣: Sweepの初期化とエージェントの実行

```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count`パラメーターで総run数を制限できます。ここではスクリプトの実行を早くするために10に制限します。run数を増やして何が起こるか見てみましょう。

```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

# 👀 結果の可視化

上記の**Sweep URL**リンクをクリックしてライブ結果を確認します。

# 🎨 ギャラリー例

W&Bで追跡および可視化されたプロジェクトの例は[Gallery →](https://app.wandb.ai/gallery)をご覧ください。

# 📏 ベストプラクティス
1. **Projects**: 複数のrunをログして比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数プロセスや交差検証フォールドの場合、各プロセスをrunsとしてログし、グループ化します。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインまたはプロダクションモデルを追跡するためにタグを追加します。
4. **Notes**: テーブルにメモを入力してrun間の変更を追跡します。
5. **Reports**: 同僚と進捗を共有するためのクイックメモを作成し、MLプロジェクトのダッシュボードやスナップショット化します。

# 🤓 高度な設定
1. [Environment variables](https://docs.wandb.com/library/environment-variables): APIキーを環境変数に設定して、管理されたクラスターでトレーニングを実行します。
2. [Offline mode](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun`モードを使用してオフラインでトレーニングし、後で結果を同期します。
3. [On-prem](https://docs.wandb.com/self-hosted): プライベートクラウドや自社インフラストラクチャのエアギャップされたサーバーにW&Bをインストールします。大学から企業チームまで、ローカルインストールを提供しています。