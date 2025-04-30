---
title: Keras
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-keras
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}
Weights & Biases を使用して、機械学習の実験管理、データセット バージョン管理、プロジェクトのコラボレーションを行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

この Colabノートブックでは、`WandbMetricsLogger` コールバックを紹介します。このコールバックは [実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) に使用できます。これにより、トレーニングと検証のメトリクスとシステムメトリクスを Weights & Biases に記録します。

## セットアップとインストール

まず、最新バージョンの Weights & Biases をインストールしましょう。次に、この Colabインスタンスを認証して W&B を使用できるようにします。

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

W&B を初めて使用する場合、またはログインしていない場合、`wandb.login()` を実行した後に表示されるリンクはサインアップ/ログインページに導きます。[無料アカウント](https://wandb.ai/signup) のサインアップは数クリックで完了します。

```python
wandb.login()
```

## ハイパーパラメーター

適切な設定システムの使用は、再現可能な機械学習の推奨ベストプラクティスです。W&B を使用して、実験ごとにハイパーパラメーターを追跡できます。この Colab では、シンプルな Python の `dict` を設定システムとして使用します。

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

この Colab では、TensorFlow データセットカタログから [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) データセットを使用します。私たちの目標は、TensorFlow/Keras を使用してシンプルな画像分類パイプラインを構築することです。

```python
train_ds, valid_ds = tfds.load("fashion_mnist", split=["train", "test"])
```

```python
AUTOTUNE = tf.data.AUTOTUNE

def parse_data(example):
    # 画像を取得する
    image = example["image"]
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # ラベルを取得する
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
# W&B の run を初期化する
run = wandb.init(project="intro-keras", config=configs)

# あなたのモデルをトレーニングする
model.fit(
    trainloader,
    epochs=configs["epochs"],
    validation_data=validloader,
    callbacks=[
        WandbMetricsLogger(log_freq=10)
    ],  # ここで WandbMetricsLogger を使用することに注意
)

# W&B の run を終了する
run.finish()
```