---
title: モデルをトラッキングする
description: W&B Python SDK を使って、モデル、その依存関係、そしてそのモデルに関連するその他の情報をトラッキングしましょう。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-log-model-to-experiment
    parent: model-registry
weight: 3
---

W&B の Python SDK を使って、モデル、その依存関係、その他のモデルに関連する情報をトラッキングしましょう。

裏側では、W&B が [model artifact]({{< relref path="./model-management-concepts.md#model-artifact" lang="ja" >}}) のリネージを作成します。これは W&B App で、または W&B の Python SDK からプログラム的に確認できます。詳しくは [モデルリネージマップの作成]({{< relref path="./model-lineage.md" lang="ja" >}}) をご覧ください。

## モデルをログする方法

`run.log_model` API を使ってモデルをログできます。`path` パラメータにはモデルファイルが保存されているパスを指定します。パスにはローカルファイル、ディレクトリ、または `s3://bucket/path` のような外部バケットを示す [リファレンス URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ja" >}}) も指定できます。

オプションで `name` パラメータに model artifact の名前を設定できます。`name` を指定しない場合、W&B は入力パスのベース名に run ID を付与して名前を自動作成します。

以下のコードスニペットをコピー＆ペーストし、`<>` で囲まれた値はご自身のものに置き換えてください。

```python
import wandb

# W&B run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# モデルをログ
run.log_model(path="<path-to-model>", name="<name>")
```

<details>

<summary>例：Keras モデルを W&B にログする</summary>

以下のコード例は、畳み込みニューラルネットワーク (CNN) モデルを W&B にログする方法を示しています。

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run を初期化
run = wandb.init(entity="charlie", project="mnist-project", config=config)

# トレーニング用アルゴリズム
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# モデルを保存
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# モデルをログ
run.log_model(path=full_path, name="MNIST")

# W&B で run を明示的に終了
run.finish()
```
</details>