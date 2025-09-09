---
title: Keras
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-keras
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}
W&B を使って、機械学習の実験管理、データセットのバージョン管理、プロジェクトでの共同作業を行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使う利点" >}}

この Colab ノートブックでは、`WandbMetricsLogger` コールバックを紹介します。[実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) にこのコールバックを使うことで、トレーニングおよび検証のメトリクスに加えてシステムメトリクスも W&B にログできます。


## セットアップとインストール

まず、W&B の最新バージョンをインストールします。続いて、この Colab インスタンスを W&B で認証します。


```shell
pip install -qq -U wandb
```


```python
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# W&B に関連する import
import wandb
from wandb.integration.keras import WandbMetricsLogger
```

初めて W&B を使う場合や未ログインの場合、`wandb.login()` を実行した後に表示されるリンクからサインアップ/ログインページへ移動できます。[無料アカウント](https://wandb.ai/signup) の作成は数回のクリックで完了します。


```python
wandb.login()
```

## ハイパーパラメーター

再現性のある機械学習のためには、適切なコンフィグシステムを使うことが推奨されるベストプラクティスです。W&B を使えば、各実験のハイパーパラメーターを追跡できます。この Colab では、コンフィグシステムとしてシンプルな Python の `dict` を使います。


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

この Colab では、TensorFlow Datasets のカタログから [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) データセットを使用します。TensorFlow/Keras を使ってシンプルな画像分類パイプラインを構築します。


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
# W&B の Run を初期化
run = wandb.init(project="intro-keras", config=configs)

# モデルをトレーニングする
model.fit(
    trainloader,
    epochs=configs["epochs"],
    validation_data=validloader,
    callbacks=[
        WandbMetricsLogger(log_freq=10)
    ],  # ここで WandbMetricsLogger を使っている点に注目
)

# W&B の Run を終了
run.finish()
```