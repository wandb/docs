
# TensorFlow

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb)

Weights & Biasesを使用して機械学習実験のトラッキング、データセットのバージョン管理、プロジェクトのコラボレーションを行います。

<div><img /></div>

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## このノートブックでカバーする内容

* TensorFlowパイプラインにWeights and Biasesを簡単に統合して実験をトラッキングします。
* `keras.metrics`でメトリクスを計算します。
* 独自のトレーニングループで`wandb.log`を使用してそれらのメトリクスをログに記録します。

## インタラクティブなW&Bダッシュボードはこのようになります：

![dashboard](/images/tutorials/tensorflow/dashboard.png)

**注意**: _Step_ で始まるセクションは、W&Bを既存のコードに統合するために必要なすべてです。それ以外は標準的なMNISTの例です。

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

## Step 0️⃣: W&Bのインストール

```python
%%capture
!pip install wandb
```

## Step 1️⃣: W&Bのインポートとログイン

```python
import wandb
from wandb.keras import WandbCallback

wandb.login()
```

> サイドノート: これが初めてのW&B使用か、ログインしていない場合、`wandb.login()`を実行した後に表示されるリンクからサインアップ/ログインページに進むことができます。サインアップはワンクリックで簡単です。

# 👩‍🍳 データセットの準備

```python
# トレーニングデータセットを準備します
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.dataを使用して入力パイプラインを構築します
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

## Step 2️⃣: トレーニングループに`wandb.log`を追加

```python
def train(train_dataset, val_dataset, model, optimizer,
          train_acc_metric, val_acc_metric,
          epochs=10, log_step=200, val_log_step=50):
  
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # データセットのバッチを繰り返します
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # 各エポックの終わりに検証ループを実行します
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 各エポックの終わりにメトリクスを表示します
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポックの終わりにメトリクスをリセットします
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # ⭐: `wandb.log`を使用してメトリクスをログに記録します
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc': float(val_acc)})
```

# 👟 トレーニングの実行

## Step 3️⃣: `wandb.init`を呼び出してRunを開始

これにより、実験を開始していることがわかり、ユニークIDとダッシュボードが提供されます。

[公式ドキュメントはこちらをチェック $\rightarrow$](https://docs.wandb.com/library/init)

```python
# プロジェクト名とオプションで設定値を指定してwandbを初期化します。
# 設定値を変更して、wandbダッシュボードで結果を確認しましょう。
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

# モデルを初期化します。
model = make_model()

# オプティマイザーをインスタンス化してモデルをトレーニングします。
optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
# 損失関数をインスタンス化します。
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# メトリクスを準備します。
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

run.finish()  # Jupyter/Colabでは、runの終了を知らせます！
```

# 👀 結果の可視化

ライブ結果を見るには、上記の[**runページ**](https://docs.wandb.ai/ref/app/pages/run-page)リンクをクリックしてください。

# 🧹 Sweep 101

Weights & Biases Sweepsを使用してハイパーパラメーターの最適化を自動化し、可能なモデルの空間を探索します。

## [W&B Sweepsを使用したTensorFlowのハイパーパラメーター最適化をチェック $\rightarrow$](http://wandb.me/tf-sweeps-colab)

### W&B Sweepsを使用する利点

* **クイックセットアップ**: 数行のコードでW&B Sweepsを実行できます。
* **透明性**: 使用しているアルゴリズムをすべて引用し、[コードはオープンソース](https://github.com/wandb/client/tree/master/wandb/sweeps)です。
* **強力**: Sweepsは完全にカスタマイズ可能で構成可能です。数十台のマシンでSweepを開始することも、ラップトップで開始するのと同じくらい簡単です。

<img src="https://i.imgur.com/6eWHZhg.png" alt="Sweep Result" />

# 🎨 例のギャラリー

W&Bを使ってトラッキングおよび可視化したプロジェクトの例をギャラリーでご覧ください [Fully Connected →](https://wandb.me/fc)

# 📏 ベストプラクティス
1. **Projects**: 複数のRunをプロジェクトにログして比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスや交差検証フォールドごとに、各プロセスをRunとしてログし、それらをグループ化します。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインやプロダクションモデルを追跡するためにタグを追加します。
4. **Notes**: テーブルにメモを入力して、Run間の変更を追跡します。
5. **Reports**: 進捗状況について同僚と共有するために簡単なメモを取り、MLプロジェクトのダッシュボードやスナップショットを作成します。

## 🤓 上級設定
1. [環境変数](https://docs.wandb.com/library/environment-variables): 環境変数にAPIキーを設定して、管理されたクラスターでトレーニングを実行します。
2. [オフラインモード](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun` モードを使用してオフラインでトレーニングし、後で結果を同期します。
3. [オンプレ](https://docs.wandb.com/self-hosted): プライベートクラウドやエアギャップされたサーバーにW&Bをインストールできます。学術機関からエンタープライズチームまで、ローカルインストールに対応しています。
4. [Artifacts](http://wandb.me/artifacts-colab): モデルやデータセットをトラッキングし、バージョン管理されるような効率的な方法で、トレーニング中に開発フローのステップを自動的に取得します。