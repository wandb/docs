


# Keras Tables

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

Weights & Biasesを使用して、機械学習の実験トラッキング、データセットのバージョン管理、プロジェクトのコラボレーションを行います。

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

このColabノートブックでは、`WandbEvalCallback`を紹介します。これは、モデルの予測可視化やデータセットの可視化に役立つコールバックを構築するために継承される抽象コールバックです。詳細は[💫 `WandbEvalCallback`](https://colab.research.google.com/drive/107uB39vBulCflqmOWolu38noWLxAT6Be#scrollTo=u50GwKJ70WeJ&line=1&uniqifier=1)のセクションを参照してください。

# 🌴 セットアップとインストール

まず、最新のWeights and Biasesをインストールしましょう。その後、このColabインスタンスを認証してW&Bを使用します。

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

初めてW&Bを使用する場合やログインしていない場合、`wandb.login()`を実行後に表示されるリンクからサインアップ/ログインページに移動できます。数クリックで[無料アカウント](https://wandb.ai/signup)を作成できます。

```python
wandb.login()
```

# 🌳 ハイパーパラメーター

適切な設定システムの使用は、再現可能な機械学習のための推奨されるベストプラクティスです。W&Bを使用してすべての実験にハイパーパラメーターをトラッキングできます。このColabでは、シンプルなPythonの`dict`を設定システムとして使用します。

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

このColabでは、TensorFlowデータセットカタログから[CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100)データセットを使用します。TensorFlow/Kerasを使用してシンプルな画像分類開発フローを構築することを目指します。

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

    if (dataloader_type == "train"):
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

`WandbEvalCallback`は主にモデルの予測可視化およびデータセットの可視化を行うためのKerasコールバックを構築するための抽象基底クラスです。

これはデータセットやタスクに依存しない抽象コールバックです。これを使用するには、この基底コールバッククラスを継承し、`add_ground_truth`メソッドと`add_model_prediction`メソッドを実装します。

`WandbEvalCallback`は以下の便利なメソッドを提供するユーティリティクラスです：

- データおよび予測の`wandb.Table`インスタンスを作成する
- データおよび予測テーブルを`wandb.Artifact`としてログ
- データテーブルを`on_train_begin`にログ
- 予測テーブルを`on_epoch_end`にログ

例として、画像分類タスクのための`WandbClfEvalCallback`を以下に実装しました。このコールバックの例では：
- 検証データ（`data_table`）をW&Bにログ
- 推論を行い、各エポック終了時に予測（`pred_table`）をW&Bにログ

## メモリ使用量はどのように削減されるのか？

`on_train_begin`メソッドが呼ばれたときに、`data_table`をW&Bにログします。一度W&B Artifactとしてアップロードされると、このテーブルへの参照が取得され、`data_table_ref`クラス変数を使用してアクセスできます。`data_table_ref`は2次元リストで、`self.data_table_ref[idx][n]`のようにインデックスすることができます。ここで`idx`は行番号、`n`は列番号です。以下の例でその使用方法を確認しましょう。

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
# W&B runを初期化
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
        ),  # ここでWandbEvalCallbackを使用
    ],
)

# W&B runを終了
run.finish()
```