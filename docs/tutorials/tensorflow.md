


# TensorFlow

[**Colabノートブックで試してみる →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Simple_TensorFlow_Integration.ipynb)

Weights & Biasesを使用して、機械学習の実験トラッキング、データセットバージョン管理、プロジェクトコラボレーションを行いましょう。

<div><img /></div>

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

<div><img /></div>

## このノートブックで扱う内容

* Weights & Biases を TensorFlow pipeline に簡単に統合して実験をトラッキングする方法。
* `keras.metrics` を使ってメトリクスを計算する方法。
* カスタムトレーニングループで `wandb.log` を使用してメトリクスをログに記録する方法。

## インタラクティブなW&Bダッシュボードはこのように見えます:

![dashboard](/images/tutorials/tensorflow/dashboard.png)

**注意**: _Step_ で始まるセクションは、W&Bを既存のコードに統合するために必要なものだけです。それ以外は標準的なMNISTの例です。

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

## Step 0️⃣: W&Bをインストール

```python
%%capture
!pip install wandb
```

## Step 1️⃣: W&Bをインポートしてログイン

```python
import wandb
from wandb.keras import WandbCallback

wandb.login()
```

> サイドノート: これが初めてのW&Bの使用、またはログインしていない場合は、`wandb.login()` 実行後に表示されるリンクでサインアップ/ログインページにアクセスできます。サインアップはワンクリックで簡単です。

# 👩‍🍳 データセットを準備

```python
# トレーニングデータセットを準備
BATCH_SIZE = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# tf.dataを使用して入力パイプラインを構築
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)
```

# 🧠 モデルとトレーニングループを定義

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

## Step 2️⃣: `wandb.log`をトレーニングループに追加

```python
def train(train_dataset, val_dataset,  model, optimizer,
          train_acc_metric, val_acc_metric,
          epochs=10,  log_step=200, val_log_step=50):
  
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_loss = []   
        val_loss = []

        # データセットのバッチごとに反復処理
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
            train_loss.append(float(loss_value))

        # 各エポックの終了時に検証ループを実行
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 各エポックの終了時にメトリクスを表示
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポックの終了時にメトリクスをリセット
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # ⭐: wandb.logを使用してメトリクスをログに記録
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
```

# 👟 トレーニングを実行

## Step 3️⃣: `wandb.init`を呼び出してrunを開始

これにより、実験を開始していることが通知され、ユニークIDとダッシュボードが提供されます。

[公式ドキュメントはこちら $\rightarrow$](https://docs.wandb.com/library/init)

```python
# プロジェクト名とオプションで設定値を指定してwandbを初期化
# 設定値を変更して、wandbダッシュボードで結果を確認してください
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

# モデルをトレーニングするためのオプティマイザーをインスタンス化
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

run.finish()  # Jupyter/Colabで実行を終了したことを通知
```

# 👀 結果を可視化

上の[**runページ**](https://docs.wandb.ai/ref/app/pages/run-page)リンクをクリックして、ライブ結果を確認してください。

# 🧹 Sweep 101

Weights & Biases Sweepsを使用してハイパーパラメーターの最適化を自動化し、可能なモデルの空間を探索しましょう。

## [W&B Sweepsを使用したTensorFlowのハイパーパラメーター最適化の詳細はこちら →](http://wandb.me/tf-sweeps-colab)

### W&B Sweepsを使用する利点

* **簡単なセットアップ**: 数行のコードでW&B sweepsを実行できます。
* **透明性**: 使用しているアルゴリズムをすべて引用し、[コードはオープンソース](https://github.com/wandb/client/tree/master/wandb/sweeps)です。
* **強力**: Sweepsは完全にカスタマイズおよび構成可能です。数十台のマシンでスイープを開始するのも、ノートパソコンで開始するのも同じくらい簡単です。

<img src="https://i.imgur.com/6eWHZhg.png" alt="Sweep Result" />

# 🎨 Example Gallery

W&Bでトラッキングおよび可視化されたプロジェクトの例をギャラリーでご覧ください。[Fully Connected →](https://wandb.me/fc)

# 📏 ベストプラクティス

1. **Projects**: 複数のrunsをプロジェクトにログして比較。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスや交差検証フォールドの場合、それぞれのプロセスをrunsとしてログし、グループ化。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインやプロダクションモデルをトラッキングするためにタグを追加。
4. **Notes**: テーブルにメモを記入して、runs間の変更をトラッキング。
5. **Reports**: 進捗に関するメモを同僚と共有し、MLプロジェクトのダッシュボードとスナップショットを作成。

## 🤓 高度なセットアップ
1. [環境変数](https://docs.wandb.com/library/environment-variables): 環境変数にAPIキーを設定して、管理されたクラスターでトレーニングを実行。
2. [オフラインモード](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): オフラインでトレーニングし、後で結果を同期するために `dryrun` モードを使用。
3. [オンプレミス](https://docs.wandb.com/self-hosted): プライベートクラウドやエアギャップされたサーバーにW&Bをインストール。学術機関から企業チームまで、すべての人にローカルインストールを提供。
4. [Artifacts](http://wandb.me/artifacts-colab): モデルやデータセットをトラックおよびバージョン管理し、開発フローのステップを自動的に取得してトレーニングモデルに反映。