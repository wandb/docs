---
title: Keras
menu:
  tutorials:
    identifier: keras
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}
W&B を使って機械学習実験管理、データセットのバージョン管理、プロジェクトのコラボレーションをしましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使うメリット" >}}

この Colabノートブック では `WandbMetricsLogger` コールバックを紹介します。このコールバックを使えば、[Experiment Tracking]({{< relref "/guides/models/track" >}}) が簡単です。トレーニングやバリデーションのメトリクス、そしてシステムメトリクスもまとめて W&B に ログ できます。


## セットアップとインストール

まず、W&B の最新バージョンをインストールしましょう。その後、この colab インスタンスを認証して W&B を使えるようにします。

```shell
pip install -qq -U wandb
```

```python
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# W&B に関するインポート
import wandb
from wandb.integration.keras import WandbMetricsLogger
```

初めて W&B を使う場合やログインしていない場合は、`wandb.login()` 実行後に表示されるリンクからサインアップ／ログインできます。[無料アカウント](https://wandb.ai/signup)の作成も数クリックで簡単です。

```python
wandb.login()
```

## ハイパーパラメーター

再現性のある機械学習には適切な config システムの利用が推奨されています。W&B を使うことで、すべての実験ごとにハイパーパラメーターを管理できます。この colab では、シンプルな Python の `dict` を config システムとして利用します。

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

この colab では、TensorFlow Datasetカタログから [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) データセットを利用します。TensorFlow/Keras を使ったシンプルな画像分類パイプラインを構築するのがゴールです。

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
# W&B Run を初期化
run = wandb.init(project="intro-keras", config=configs)

# モデルのトレーニング
model.fit(
    trainloader,
    epochs=configs["epochs"],
    validation_data=validloader,
    callbacks=[
        WandbMetricsLogger(log_freq=10)
    ],  # ここで WandbMetricsLogger を使用しています
)

# W&B Run を終了
run.finish()
```