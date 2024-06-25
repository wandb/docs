
# TensorFlow Sweeps

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb)

Weights & Biasesを使用して機械学習の実験管理、データセットのバージョン管理、およびプロジェクトの共同作業を行います。

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

Weights & BiasesのSweepsを使ってハイパーパラメーターの最適化を自動化し、インタラクティブなダッシュボードを使って可能なモデルの空間を探索します。

![](https://i.imgur.com/AN0qnpC.png)


## 🤔 なぜSweepsを使うべきか？

* **クイックセットアップ**: 数行のコードだけでW&Bのsweepsを実行できます。
* **透明性**: 使用している全てのアルゴリズムを明記しており、[コードはオープンソースです](https://github.com/wandb/client/tree/master/wandb/sweeps)。
* **強力**: Sweepsは完全にカスタマイズ・設定可能です。数十のマシンにまたがるsweepを起動するのも、ノートパソコンでsweepを開始するのと同じくらい簡単です。

**[公式ドキュメントを見る $\rightarrow$](https://docs.wandb.com/sweeps)**


## このノートブックでカバーする内容

* TensorFlowで独自のトレーニングループを使ってW&B Sweepを開始するシンプルな手順。
* 画像分類タスクの最適なハイパーパラメーターを見つけます。

**注意**: _Step_から始まるセクションは、既存のコードでハイパーパラメーターsweepを実行するために必要なものです。
他のコードは単純な例を設定するためのものです。

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

> サイドノート: これが初めてのW&Bの使用またはログインしていない場合、`wandb.login()`を実行した後に表示されるリンクでサインアップ/ログインページにアクセスできます。サインアップは数クリックで完了します。

# 👩‍🍳 データセットの準備


```python
# トレーニングデータセットを準備
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
```

# 🧠 モデルとトレーニングループの定義

## 🏗️ シンプルな分類器MLPを構築


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

## 🔁 トレーニングループを書く

### Step 3️⃣: `wandb.log`でメトリクスをログ


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

        # データセットのバッチを繰り返し処理
        for step, (x_batch_train, y_batch_train) in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
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

        # 3️⃣ wandb.logを使用してメトリクスをログ
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'acc': float(train_acc), 
                   'val_loss': np.mean(val_loss),
                   'val_acc': float(val_acc)})
```

# Step 4️⃣: Sweepを設定する

ここで行うことは:
* 探索するハイパーパラメーターを定義
* ハイパーパラメーターの最適化方法を提供します。 `random`, `grid`, `bayes` メソッドがあります。
* `bayes`を使用する場合、目的と`metric`を提供します。例えば、`val_loss`を`最小化`する。
* `hyperband`を使用してパフォーマンスの低いrunを早期終了

#### [Sweepの設定に関する詳細はこちら $\rightarrow$](https://docs.wandb.com/sweeps/configuration)


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

# Step 5️⃣: トレーニングループをラップ

`sweep_train`のような関数が必要です。この関数は`wandb.config`を使用してハイパーパラメーターを設定し、その後に `train` が呼び出されます。


```python
def sweep_train(config_defaults=None):
    # デフォルト値を設定
    config_defaults = {
        "batch_size": 64,
        "learning_rate": 0.01
    }
    # サンプルプロジェクト名でwandbを初期化
    wandb.init(config=config_defaults)  # これはSweepで上書きされます

    # その他のハイパーパラメーター設定を指定する場合
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
    # 損失関数をインスタンス化
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

`count`パラメーターを使用してrunの総数を制限できます。このスクリプトを早く実行するために10に制限します。runの数を増やして結果を確認してみてください。


```python
wandb.agent(sweep_id, function=sweep_train, count=10)
```

# 👀 結果を可視化

**Sweep URL**リンクをクリックして、ライブ結果を確認してください。

# 🎨 ギャラリーの例

W&Bで追跡・可視化されたプロジェクトの例を[ギャラリー →](https://app.wandb.ai/gallery)で確認してください。

# 📏 ベストプラクティス
1. **Projects**: 複数のrunをログして比較する。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスや交差検証フォールドの場合、各プロセスをRunsとしてログし、一つのグループにまとめる。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインやプロダクションモデルを追跡するためにタグを追加します。
4. **Notes**: テーブル内でメモを入力して、run間の変更を追跡します。
5. **Reports**: 進捗に関するメモを取って同僚と共有したり、MLプロジェクトのダッシュボードやスナップショットを作成します。

# 🤓 高度なセットアップ
1. [環境変数](https://docs.wandb.com/library/environment-variables): 環境変数にAPIキーを設定し、管理されたクラスターでトレーニングを実行します。
2. [オフラインモード](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): オフラインでトレーニングし、後で結果を同期するために`dryrun`モードを使用します。
3. [オンプレミス](https://docs.wandb.com/self-hosted): プライベートクラウドやエアギャップされたサーバーにW&Bをインストールします。学術機関から企業のTeamsまで、ローカルインストールを提供しています。