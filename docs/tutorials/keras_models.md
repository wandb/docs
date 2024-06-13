
# Keras Models

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb)

Weights & Biases を使って機械学習の実験トラッキング、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

このColabノートブックでは `WandbModelCheckpoint` コールバックを紹介しています。このコールバックを使用して、モデルのチェックポイントをWeights & Biases [Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning)にログします。

# 🌴 セットアップとインストール

まず、Weights & Biasesの最新バージョンをインストールしましょう。その後、このColabインスタンスをW&B用に認証します。

```python
!pip install -qq -U wandb
```

```python
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# Weights & Biases関連のインポート
import wandb
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
```

初めてW&Bを使用する場合やログインしていない場合は、`wandb.login()`を実行した後に表示されるリンクからサインアップ/ログインページにアクセスできます。[無料アカウント](https://wandb.ai/signup)を作成するのは数クリックで済みます。

```python
wandb.login()
```

# 🌳 ハイパーパラメーター

適切な設定システムの使用は、再現性のある機械学習のベストプラクティスとして推奨されます。W&Bを使用して、各実験のハイパーパラメーターを追跡できます。ここでは、シンプルなPythonの `dict` を設定システムとして使用します。

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

# 🍁 データセット

このColabノートブックでは、TensorFlowデータセットカタログから[CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100)データセットを使用します。目標は、TensorFlow/Kerasを使用してシンプルな画像分類開発フローを構築することです。

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

# 🎄 モデル

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

# 🌿 モデルのコンパイル

```python
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')]
)
```

# 🌻 トレーニング

```python
# W&B runを初期化
run = wandb.init(
    project = "intro-keras",
    config = configs
)

# モデルのトレーニング
model.fit(
    trainloader,
    epochs = configs["epochs"],
    validation_data = validloader,
    callbacks = [
        WandbMetricsLogger(log_freq=10),
        WandbModelCheckpoint(filepath="models/") # ここでWandbModelCheckpointを使用していることに注意
    ]
)

# W&B runを終了
run.finish()
```