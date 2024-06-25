
# TensorFlow

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb)

Weights & Biases を使用して、機械学習実験の追跡、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

<div><img /></div>

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## このノートブックでカバーする内容

* Weights and Biases と TensorFlow の簡単なインテグレーションによる実験管理
* `keras.metrics` を用いてメトリクスを計算する方法
* `wandb.log` を使ってカスタムトレーニングループでメトリクスをログする方法

## インタラクティブな W&B ダッシュボードはこのようになります：

![dashboard](/images/tutorials/tensorflow/dashboard.png)

**注意**: _Step_ で始まるセクションは W&B を既存のコードに組み込むために必要な部分です。その他の部分は標準的な MNIST の例に過ぎません。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

# 🚀 インストール、インポート、ログイン

## Step 0️⃣: W&B をインストールする

```python
%%capture
!pip install wandb
```

## Step 1️⃣: W&B をインポートしてログインする

```python
import wandb
from wandb.keras import WandbCallback

wandb.login()
```

> 補足: もし W&B を初めて利用する場合やログインしていない場合は、`wandb.login()` 実行後に表示されるリンクからサインアップ/ログインページに移動できます。サインアップはワンクリックで簡単に完了します。

# 👩‍🍳 データセットの準備

```python
# トレーニングデータセットの準備
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

# 🧠 モデルとトレーニングループの定義

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

## Step 2️⃣: トレーニングループに `wandb.log` を追加する

```python
def train(train_dataset, val_dataset, model, optimizer,
          train_acc_metric, val_acc_metric,
          epochs=10, log_step=200, val_log_step=50):
  
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

        # 各エポックの終わりに検証ループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
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

        # ⭐: wandb.log を使ってメトリクスをログ
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc': float(val_acc)})
```

# 👟 トレーニングの実行

## Step 3️⃣: run を開始するために `wandb.init` を呼び出す

これにより、実験を開始し、一意のIDとダッシュボードを提供します。

[公式ドキュメントはこちら $\rightarrow$](https://docs.wandb.com/library/init)

```python
# プロジェクト名とオプションで設定を指定して wandb を初期化
# 設定値を変更し、wandb ダッシュボードで結果を確認
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

# モデルの初期化
model = make_model()

# モデルをトレーニングするためにオプティマイザーをインスタンス化
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 損失関数をインスタンス化
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# メトリクスの準備
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

run.finish()  # Jupyter/Colab では、終了を知らせるために！
```

# 👀 結果を視覚化

ライブ結果を見るには、上記の [**run ページ**](https://docs.wandb.ai/ref/app/pages/run-page) のリンクをクリックしてください。

# 🧹 Sweep 101

Weights & Biases Sweeps を使用してハイパーパラメーターの最適化を自動化し、考えられるモデルの空間を探索しましょう。

## [W&B Sweeps を使用した TensorFlow のハイパーパラメーター最適化の詳細はこちら $\rightarrow$](http://wandb.me/tf-sweeps-colab)

### W&B Sweeps 使用の利点

* **迅速な設定**: わずか数行のコードで W&B sweeps を実行できます。
* **透明性**: 使用しているすべてのアルゴリズムを引用しており、[コードはオープンソースです](https://github.com/wandb/client/tree/master/wandb/sweeps)。
* **強力**: sweeps は完全にカスタマイズおよび設定可能です。数十台のマシンにわたって sweep を起動するのも、ラップトップで sweep を開始するのと同じくらい簡単です。

<img src="https://i.imgur.com/6eWHZhg.png" alt="Sweep Result" />

# 🎨 例のギャラリー

W&B で追跡・可視化されたプロジェクトの例をギャラリーでご覧ください。[Fully Connected →](https://wandb.me/fc)

# 📏 ベストプラクティス
1. **Projects**: 複数の run をプロジェクトにログして比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスや交差検証フォールドのために、各プロセスを runs としてログし、一括りにします。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインやプロダクションモデルを追跡するためにタグを追加します。
4. **Notes**: テーブルにメモを書き込み、run 間の変更を追跡します。
5. **Reports**: 同僚と進捗を共有するためにレポートを書き込み、ML プロジェクトのダッシュボードやスナップショットを作成します。

## 🤓 高度な設定
1. [環境変数](https://docs.wandb.com/library/environment-variables): 環境変数に API キーを設定し、管理されたクラスターでトレーニングを実行できます。
2. [オフラインモード](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun` モードを使用してオフラインでトレーニングを実行し、後で結果を同期します。
3. [オンプレミス](https://docs.wandb.com/self-hosted): プライベートクラウドやエアギャップサーバーのインフラに W&B をインストールします。アカデミアから企業チームまで、ローカルインストールに対応しています。
4. [Artifacts](http://wandb.me/artifacts-colab): モデルやデータセットを追跡及びバージョン管理し、パイプラインのステップをトレーニングしながら自動的に取得します。