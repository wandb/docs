


# TensorFlow Sweeps

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb)

Weights & Biases を使用して、機械学習実験の追跡、データセットのバージョン管理、プロジェクトのコラボレーションを行います。

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

Weights & BiasesのSweepsを使用して、ハイパーパラメーターの最適化を自動化し、可能なモデルの空間を探索できます。以下のようなインタラクティブなダッシュボードが付属しています。

![](https://i.imgur.com/AN0qnpC.png)


## 🤔 なぜSweepsを使うべきか？

* **簡単なセットアップ**: 数行のコードでW&B sweepsを実行できます。
* **透明性**: 使用しているアルゴリズムをすべて引用しており、[コードはオープンソースです](https://github.com/wandb/client/tree/master/wandb/sweeps)。
* **強力**: Sweepsは完全にカスタマイズおよび設定可能です。数十台のマシンでスイープを実行することも、ノートパソコンでスイープを開始することも同じくらい簡単です。

**[公式ドキュメントをチェックする $\rightarrow$](https://docs.wandb.com/sweeps)**


## このノートブックがカバーする内容

* TensorFlowでのカスタムトレーニングループを使用したW&B Sweepの簡単な開始手順。
* 画像分類タスクに最適なハイパーパラメーターを見つけます。

**注意**: _Step_ で始まるセクションは既存のコードでハイパーパラメータースイープを行うために必要なものです。残りのコードはシンプルな例を設定するためのものです。


# 🚀 インストール、インポート、ログイン

### Step 0️⃣: W&Bをインストールする


```python
%%capture
!pip install wandb
```

### Step 1️⃣: W&Bをインポートしてログインする


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

> サイドノート: これが初めてのW&Bの使用であったり、まだログインしていない場合、`wandb.login()`を実行した後に表示されるリンクからサインアップ/ログインページに移動します。サインアップは数回のクリックだけで簡単にできます。

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

### Step 3️⃣: `wandb.log` でメトリクスをログする


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

        # 各エポックの終わりに検証ループを実行する
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            val_loss_value = test_step(x_batch_val, y_batch_val, 
                                       model, loss_fn, 
                                       val_acc_metric)
            val_loss.append(float(val_loss_value))
            
        # 各エポックの終わりにメトリクスを表示する
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        val_acc = val_acc_metric.result()
        print("Validation acc: %.4f" % (float(val_acc),))

        # 各エポックの終わりにメトリクスをリセットする
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # 3️⃣ `wandb.log` を使用してメトリクスをログする
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc':float(val_acc)})
```

# Step 4️⃣: Sweepの設定

ここでは次のことを行います:
* スイープするハイパーパラメーターの定義
* ハイパーパラメーター最適化のメソッドの指定。`random`、`grid`、`bayes`メソッドがあります。
* `bayes`を使用する場合は、目的と`metric`の指定、例えば`val_loss`を`minimize`する。
* パフォーマンスが低いrunを早期終了するために`hyperband`を使用

#### [Sweepの設定についてもっと見る $\rightarrow$](https://docs.wandb.com/sweeps/configuration)


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

以下のような関数`sweep_train`が必要です。これにより、`train`が呼び出される前に`wandb.config`を使用してハイパーパラメーターが設定されます。


```python
def sweep_train(config_defaults=None):
    # デフォルト値を設定
    config_defaults = {
        "batch_size": 64,
        "learning_rate": 0.01
    }
    # サンプルプロジェクト名でwandbを初期化
    wandb.init(config=config_defaults)  # これはSweepで上書きされます

    # その他のハイパーパラメーターを設定に指定
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

    # モデルをトレーニングするためにオプティマイザーをインスタンス化
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

# Step 6️⃣: Sweepを初期化し、エージェントを実行


```python
sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
```

`count`パラメーターでrunの合計数を制限できます。スクリプトをすばやく実行するために、ここでは10に制限していますが、runの数を増やして何が起こるか確認してください。


```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

# 👀 結果の可視化

上記の**Sweep URL**リンクをクリックして、ライブ結果を確認してください。


# 🎨 例ギャラリー

W&Bを使用して追跡および可視化されたプロジェクトの例を[ギャラリーで見る →](https://app.wandb.ai/gallery)

# 📏 ベストプラクティス
1. **Projects**: 複数のrunをプロジェクトにログして比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスや交差検証フォールドのために、各プロセスをrunとしてログし、それらをグループにまとめます。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインやプロダクションモデルを追跡するためにタグを追加します。
4. **Notes**: 走行間の変更を追跡するためにテーブルにメモを入力します。
5. **Reports**: 進捗状況に関するクイックメモを同僚と共有し、MLプロジェクトのダッシュボードやスナップショットを作成します。

# 🤓 高度なセットアップ
1. [環境変数](https://docs.wandb.com/library/environment-variables): 管理されたクラスター上でトレーニングを実行するために環境変数にAPIキーを設定します。
2. [オフラインモード](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun`モードを使用してオフラインでトレーニングを実行し、後で結果を同期します。
3. [オンプレミス](https://docs.wandb.com/self-hosted): プライベートクラウドまたはエアギャップサーバーにW&Bをインストールします。学術機関からエンタープライズチームまで、すべてのローカルインストールに対応しています。