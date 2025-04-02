---
title: Keras models
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-keras_models
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
Weights & Biases を使用して、 機械学習 の 実験管理 、 データセット の バージョン管理 、および プロジェクト の コラボレーションを行います。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

この Colab ノートブック では、`WandbModelCheckpoint` コールバック について紹介します。この コールバック を使用して、モデル の チェックポイント を Weights & Biases [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) に ログ 記録します。

## セットアップとインストール

まず、Weights & Biases の最新バージョンをインストールしましょう。次に、この Colab インスタンスを認証して W&B を使用します。

```python
!pip install -qq -U wandb
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
from wandb.integration.keras import WandbModelCheckpoint
```

W&B を初めて使用する場合、または ログイン していない場合は、`wandb.login()` の実行後に表示されるリンクからサインアップ/ログイン ページに移動します。[無料アカウント](https://wandb.ai/signup) へのサインアップは、数回クリックするだけで簡単に行えます。

```python
wandb.login()
```

## ハイパーパラメーター

再現性のある 機械学習 のためには、適切な config システムを使用することが推奨されるベストプラクティスです。W&B を使用して、すべての 実験 の ハイパーパラメーター を追跡できます。この Colab では、単純な Python `dict` を config システムとして使用します。

```python
configs = dict(
    num_classes = 10,
    shuffle_buffer = 1024,
    batch_size = 64,
    image_size = 28,
    image_channels = 1,
    earlystopping_patience = 3,
    learning_rate = 1e-3,
    epochs = 10
)
```

## データセット

この Colab では、TensorFlow Dataset カタログの [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) データセット を使用します。TensorFlow/Keras を使用して、単純な 画像分類 パイプライン を構築することを目指します。

```python
train_ds, valid_ds = tfds.load('fashion_mnist', split=['train', 'test'])
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
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
    backbone.trainable = False

    inputs = layers.Input(shape=(configs["image_size"], configs["image_size"], configs["image_channels"]))
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3,3), padding="same")(resize)
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
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')]
)
```

## 学習

```python
# W&B run を初期化
run = wandb.init(
    project = "intro-keras",
    config = configs
)

# モデルを学習
model.fit(
    trainloader,
    epochs = configs["epochs"],
    validation_data = validloader,
    callbacks = [
        WandbMetricsLogger(log_freq=10),
        WandbModelCheckpoint(filepath="models/") # ここで WandbModelCheckpoint を使用していることに注意してください
    ]
)

# W&B run を閉じる
run.finish()
```