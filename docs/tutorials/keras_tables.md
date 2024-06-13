
# Keras Tables

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

Weights & Biasesを使用して機械学習実験の追跡、データセットのバージョン管理、プロジェクトのコラボレーションを行います。

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

このColabノートブックでは、`WandbEvalCallback`を紹介します。これは抽象コールバックで、モデル予測の可視化やデータセットの可視化のために役立つコールバックを構築するために継承されます。詳細は[💫 `WandbEvalCallback`](https://colab.research.google.com/drive/107uB39vBulCflqmOWolu38noWLxAT6Be#scrollTo=u50GwKJ70WeJ&line=1&uniqifier=1)セクションを参照してください。

# 🌴 セットアップとインストール

まず、最新バージョンのWeights and Biasesをインストールしましょう。その後、このColabインスタンスを認証してW&Bを使用します。

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

# Weights and Biases関連のインポート
import wandb
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
from wandb.integration.keras import WandbEvalCallback
```

もし初めてW&Bを使用する場合やログインしていない場合、`wandb.login()`を実行した後に表示されるリンクでサインアップ/ログインページにアクセスできます。数回のクリックで[無料アカウント](https://wandb.ai/signup)を作成できます。

```python
wandb.login()
```

# 🌳 ハイパーパラメーター

再現性のある機械学習において適切なコンフィグシステムの使用は推奨されるベストプラクティスです。W&Bを使用して各実験のハイパーパラメーターを追跡できます。このColabでは、シンプルなPythonの`dict`をコンフィグシステムとして使用します。

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

# 🍁 データセット

このColabでは、TensorFlow Datasetカタログから[CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100)データセットを使用します。TensorFlow/Kerasを使用してシンプルな画像分類パイプラインを構築することを目指します。

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

# 🌿 モデルのコンパイル

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

# 💫 `WandbEvalCallback`

`WandbEvalCallback`は、主にモデル予測の可視化、二義的にはデータセットの可視化のためにKerasのコールバックを構築するための抽象基底クラスです。

これはデータセットおよびタスクに依存しない抽象コールバックです。これを使用するには、この基底コールバッククラスから継承し、`add_ground_truth`および`add_model_prediction`メソッドを実装します。

`WandbEvalCallback`は以下の便利なメソッドを提供するユーティリティクラスです：

- データと予測の`wandb.Table`インスタンスを作成する、
- `wandb.Artifact`としてデータと予測のTablesをログする、
- `on_train_begin`メソッドでデータテーブルをログする、
- `on_epoch_end`メソッドで予測テーブルをログする。

例として、以下に画像分類タスク用の`WandbClfEvalCallback`を実装しています。この例のコールバックは：
- 検証データ（`data_table`）をW&Bにログする、
- 推論を行い、エポック終了時に予測（`pred_table`）をW&Bにログする。

## メモリフットプリントがどのように削減されるのか？

`on_train_begin`メソッドが呼び出されると`data_table`をW&Bにログします。一度W&B Artifactとしてアップロードされると、`data_table_ref`クラス変数を使ってこのテーブルへの参照を取得できます。`data_table_ref`は2Dリストで、`self.data_table_ref[idx][n]`のようにインデックスを付けることができます。`idx`は行番号、`n`は列番号です。以下の例でその使い方を見てみましょう。

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

# 🌻 トレーニング

```python
# W&Bのrunを初期化
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
        ),  # ここでWandbEvalCallbackを使用します
    ],
)

# W&Bのrunを終了
run.finish()
```