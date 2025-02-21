---
title: Keras tables
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-keras_tables
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}
Weights & Biases を使用して、機械学習の実験管理、データセットのバージョン管理、およびプロジェクトコラボレーションを行います。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

この Colab ノートブックでは、`WandbEvalCallback` を紹介します。これは、モデル予測の可視化やデータセットの可視化のための便利なコールバックを構築するために継承できる抽象コールバックです。

## セットアップとインストール

まず、最新バージョンの Weights and Biases をインストールしましょう。その後、この Colab インスタンスを W&B と認証して使用します。

```shell
pip install -qq -U wandb
```

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# Weights and Biases 関連のインポート
import wandb
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
from wandb.integration.keras import WandbEvalCallback
```

もしこれが初めて W&B を使用する場合、またはログインしていない場合、`wandb.login()` を実行した後に表示されるリンクは、サインアップ/ログインページに連れて行ってくれます。[無料アカウント](https://wandb.ai/signup)にサインアップすることは、数クリックで簡単です。

```python
wandb.login()
```

## ハイパーパラメーター

適切なコンフィグシステムを使用することは、再現性のある機械学習のために推奨されるベストプラクティスです。W&B を使用して、各実験のハイパーパラメーターをトラッキングできます。このコラボでは、シンプルな Python の `dict` をコンフィグシステムとして使用します。

```python
configs = dict(
    num_classes=10,
    shuffle_buffer=1024,
    batch_size=64,
    image_size=28,
    image_channels=1,
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=10,
)
```

## データセット

この Colab では、TensorFlow Dataset カタログから [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) データセットを使用します。TensorFlow/Keras を使用してシンプルな画像分類パイプラインを構築することを目指します。

```python
train_ds, valid_ds = tfds.load("fashion_mnist", split=["train", "test"])
```

```
AUTOTUNE = tf.data.AUTOTUNE

def parse_data(example):
    # 画像を取得
    image = example["image"]
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # ラベルを取得
    label = example["label"]
    label = tf.one_hot(label, depth=configs["num_classes"])

    return image, label

def get_dataloader(ds, configs, dataloader_type="train"):
    dataloader = ds.map(parse_data, num_parallel_calls=AUTOTUNE)

    if dataloader_type == "train":
        dataloader = dataloader.shuffle(configs["shuffle_buffer"])
      
    dataloader = (
        dataloader
        .batch(configs["batch_size"])
        .prefetch(AUTOTUNE)
    )

    return dataloader
```

```python
trainloader = get_dataloader(train_ds, configs)
validloader = get_dataloader(valid_ds, configs, dataloader_type="valid")
```

## モデル

```python
def get_model(configs):
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
        weights="imagenet", include_top=False
    )
    backbone.trainable = False

    inputs = layers.Input(
        shape=(configs["image_size"], configs["image_size"], configs["image_channels"])
    )
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3, 3), padding="same")(resize)
    preprocess_input = tf.keras.applications.mobilenet.preprocess_input(neck)
    x = backbone(preprocess_input)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(configs["num_classes"], activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)
```

```python
tf.keras.backend.clear_session()
model = get_model(configs)
model.summary()
```

## モデルのコンパイル

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top@5_accuracy"),
    ],
)
```

## `WandbEvalCallback`

`WandbEvalCallback` は、主にモデル予測の可視化、二次的にデータセットの可視化のための Keras コールバックを構築するための抽象基底クラスです。

これはデータセットおよびタスクに依存しない抽象コールバックです。これを使用するには、この基底コールバッククラスから継承し、`add_ground_truth` と `add_model_prediction` メソッドを実装します。

`WandbEvalCallback` は便利なメソッドを提供するユーティリティクラスで、次のことを可能にします：

- データと予測の `wandb.Table` インスタンスの作成、
- データと予測のテーブルを `wandb.Artifact` としてログ、
- データテーブルを `on_train_begin` にログ、
- 各エポック終了 (`on_epoch_end`) 時に予測テーブルをログ。

例として、画像分類タスクのために `WandbClfEvalCallback` を以下に実装しました。このコールバックは以下を行います：
- 検証データ (`data_table`) を W&B にログ、
- 推論を実行し、各エポックの終了時に予測 (`pred_table`) を W&B にログ。

## メモリのフットプリントがどのように削減されるか

`on_train_begin` メソッドが呼び出されたときに `data_table` を W&B にログします。W&B Artifact としてアップロードされた後、このテーブルへの参照を取得し、`data_table_ref` クラス変数を使用してアクセスできます。`data_table_ref` は 2D リストで、`self.data_table_ref[idx][n]` のようにインデックスできます。ここで `idx` は行番号、`n` は列番号です。以下の例でその使い方を見てみましょう。

```python
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validloader, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.val_data = validloader.unbatch().take(num_samples)

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(self.val_data):
            self.data_table.add_data(idx, wandb.Image(image), np.argmax(label, axis=-1))

    def add_model_predictions(self, epoch, logs=None):
        # 予測を取得
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            )

    def _inference(self):
        preds = []
        for image, label in self.val_data:
            pred = self.model(tf.expand_dims(image, axis=0))
            argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]
            preds.append(argmax_pred)

        return preds
```

## トレーニング

```python
# W&B の run を初期化
run = wandb.init(project="intro-keras", config=configs)

# モデルをトレーニング
model.fit(
    trainloader,
    epochs=configs["epochs"],
    validation_data=validloader,
    callbacks=[
        WandbMetricsLogger(log_freq=10),
        WandbClfEvalCallback(
            validloader,
            data_table_columns=["idx", "image", "ground_truth"],
            pred_table_columns=["epoch", "idx", "image", "ground_truth", "prediction"],
        ),  # ここで WandbEvalCallback を使用していることに注意
    ],
)

# W&B の run を終了
run.finish()
```