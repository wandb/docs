---
title: Keras tables
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-keras_tables
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}
Weights & Biases を使用して、 機械学習 の 実験管理 、 データセット の バージョン管理 、および プロジェクト コラボレーションを行います。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

この Colab ノートブック では、モデルの 予測 の 可視化 および データセット の 可視化に役立つ コールバック を構築するために継承できる抽象 コールバック である `WandbEvalCallback` を紹介します。

## セットアップとインストール

まず、Weights & Biases の最新 バージョン をインストールしましょう。次に、この colab インスタンスを 認証 して W&B を使用します。


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

# Weights and Biases related imports
import wandb
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
from wandb.integration.keras import WandbEvalCallback
```

W&B を初めて使用する場合、または ログイン していない場合は、`wandb.login()` の実行後に表示されるリンクからサインアップ/ログイン ページに移動します。[無料アカウント](https://wandb.ai/signup) へのサインアップは、数回クリックするだけで簡単に行えます。


```python
wandb.login()
```

## ハイパーパラメーター

再現性のある 機械学習 には、適切な config システムを使用することをお勧めします。W&B を使用して、すべての 実験 の ハイパーパラメーター を追跡できます。この colab では、単純な Python の `dict` を config システムとして使用します。


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

この colab では、TensorFlow Dataset カタログの [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) データセット を使用します。TensorFlow/Keras を使用して、単純な 画像分類 パイプライン を構築することを目指します。


```python
train_ds, valid_ds = tfds.load("fashion_mnist", split=["train", "test"])
```


```
AUTOTUNE = tf.data.AUTOTUNE


def parse_data(example):
    # Get image
    image = example["image"]
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Get label
    label = example["label"]
    label = tf.one_hot(label, depth=configs["num_classes"])

    return image, label


def get_dataloader(ds, configs, dataloader_type="train"):
    dataloader = ds.map(parse_data, num_parallel_calls=AUTOTUNE)

    if dataloader_type=="train":
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

`WandbEvalCallback` は、主にモデルの 予測 の 可視化、次に データセット の 可視化のための Keras コールバック を構築するための抽象基本クラスです。

これは、 データセット とタスクに依存しない抽象 コールバック です。これを使用するには、この基本 コールバック クラスから継承し、`add_ground_truth` および `add_model_prediction` メソッド を実装します。

`WandbEvalCallback` は、次のことに役立つ メソッド を提供する ユーティリティ クラスです。

- データ と 予測 の `wandb.Table` インスタンスを作成する
- データ と 予測 の テーブル を `wandb.Artifact` として ログ に記録する
- `on_train_begin` で データテーブル を ログ に記録する
- `on_epoch_end` で 予測 テーブル を ログ に記録する

例として、以下に 画像分類 タスク の `WandbClfEvalCallback` を実装しました。この コールバック の例:
- 検証 データ ( `data_table` ) を W&B に ログ 記録します。
- 推論 を実行し、すべての エポック の最後に 予測 ( `pred_table` ) を W&B に ログ 記録します。


## メモリフットプリントを削減する方法

`on_train_begin` メソッド が呼び出されると、`data_table` を W&B に ログ 記録します。W&B Artifact としてアップロードされると、`data_table_ref` クラス変数を使用して アクセス できるこの テーブル への参照を取得します。`data_table_ref` は、`self.data_table_ref[idx][n]` のように インデックス を付けることができる 2D リストです。ここで、`idx` は行番号、`n` は列番号です。以下の例で使用方法を見てみましょう。

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
        # Get predictions
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

## 訓練


```python
# Initialize a W&B run
run = wandb.init(project="intro-keras", config=configs)

# Train your model
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
        ),  # Notice the use of WandbEvalCallback here
    ],
)

# Close the W&B run
run.finish()
```