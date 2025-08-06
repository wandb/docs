---
title: Keras テーブル
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-keras_tables
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}
W&B で機械学習の実験管理、データセットのバージョン管理、プロジェクトでのコラボレーションを行いましょう。
{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使うメリット" >}}

この Colabノートブック では、`WandbEvalCallback` を紹介します。これは抽象的なコールバックで、モデルの予測可視化やデータセットの可視化のための便利なカスタムコールバックを作成する際のベースになります。

## セットアップとインストール

まず W&B の最新バージョンをインストールします。その後、この colab インスタンスで W&B を認証します。

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

# W&B 関連のインポート
import wandb
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
from wandb.integration.keras import WandbEvalCallback
```

W&B を初めて使う場合やまだログインしていない場合は、`wandb.login()` 実行後に表示されるリンクからサインアップ／ログインページにアクセスできます。[無料アカウントの作成](https://wandb.ai/signup)も数クリックで完了します。

```python
wandb.login()
```

## ハイパーパラメーター

再現性ある機械学習のためには、適切な config システムの利用がおすすめです。W&B を使えば各実験のハイパーパラメーターも管理できます。この colab ではシンプルな Python の `dict` を config システムとして使います。

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

この colab では、TensorFlow Dataset カタログから [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) データセットを利用します。TensorFlow/Keras を使ったシンプルな画像分類パイプラインを構築します。

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

`WandbEvalCallback` は、主にモデル予測の可視化、さらにデータセット可視化のための Keras コールバックを構築するための抽象基底クラスです。

このクラスは、データセットやタスクに依存しない抽象的なコールバックです。利用する際は、このベースクラスを継承し、`add_ground_truth` や `add_model_prediction` メソッドを実装してください。

`WandbEvalCallback` は次のような便利なメソッドを提供しています：

- データや予測の `wandb.Table` インスタンスの作成
- データや予測の Tables を `wandb.Artifact` としてログ
- 学習開始時 (`on_train_begin`) にデータテーブルをログ
- 各エポック終了時 (`on_epoch_end`) に予測テーブルをログ

以下は画像分類タスク用に実装した `WandbClfEvalCallback` の例です。このコールバックでは：
- 検証用データ（`data_table`）を W&B に記録
- 推論を実行し、各エポック末に予測（`pred_table`）を W&B に記録

## メモリ使用量の削減方法

`on_train_begin` が呼ばれたときに `data_table` を W&B へ記録します。これが W&B Artifact としてアップロードされると、このテーブルへの参照が `data_table_ref` クラス変数に格納されます。`data_table_ref` は2次元リストであり、`self.data_table_ref[idx][n]` のようにインデックス指定が可能です（`idx` は行番号、`n` は列番号）。下記の例で使い方を確認しましょう。

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

## 学習

```python
# W&B Run の初期化
run = wandb.init(project="intro-keras", config=configs)

# モデルを学習
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
        ),  # ここで WandbEvalCallback を使用している点に注目
    ],
)

# W&B Run をクローズ
run.finish()
```