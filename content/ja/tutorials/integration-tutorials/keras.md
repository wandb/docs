---
title: Keras
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-keras
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}
Weights & Biases を使用して機械学習の実験管理、データセットのバージョン管理、およびプロジェクトのコラボレーションを行います。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

この Colab ノートブックでは、`WandbMetricsLogger` コールバックを紹介します。このコールバックを[実験管理]({{< relref path="/guides/models/track" lang="ja" >}})に使用します。これにより、トレーニングと検証のメトリクスをシステムメトリクスと共に Weights and Biases にログします。

## セットアップとインストール

まず、最新バージョンの Weights and Biases をインストールします。次に、この Colab インスタンスを W&B で認証します。

```shell
pip install -qq -U wandb
```

```python
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# Weights and Biases 関連のインポート
import wandb
from wandb.integration.keras import WandbMetricsLogger
```

もし W&B を初めて使用する場合やログインしていない場合は、`wandb.login()` を実行した後に表示されるリンクからサインアップ/ログインページに移動できます。[無料アカウント](https://wandb.ai/signup)の登録は簡単です。

```python
wandb.login()
```

## ハイパーパラメーター

適切な構成システムの使用は、再現可能な機械学習のための推奨されるベストプラクティスです。W&B を使用して、各実験のハイパーパラメーターを追跡できます。この colab では、シンプルな Python `dict` を構成システムとして使用します。

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

この colab では、TensorFlow Dataset カタログから [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) データセットを使用します。私たちは TensorFlow/Keras を使ってシンプルな画像分類パイプラインを構築することを目的としています。

```python
train_ds, valid_ds = tfds.load("fashion_mnist", split=["train", "test"])
```

```python
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

    dataloader = dataloader.batch(configs["batch_size"]).prefetch(AUTOTUNE)

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

## トレーニング

```python
# W&B ランを初期化
run = wandb.init(project="intro-keras", config=configs)

# モデルをトレーニング
model.fit(
    trainloader,
    epochs=configs["epochs"],
    validation_data=validloader,
    callbacks=[
        WandbMetricsLogger(log_freq=10)
    ],  # ここで WandbMetricsLogger を使用していることに注意
)

# W&B ランを終了
run.finish()
```