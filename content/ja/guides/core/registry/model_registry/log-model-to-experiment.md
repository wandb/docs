---
title: モデルを追跡する
description: W&B Python SDK を使用して、モデル、モデルの依存関係、およびそのモデルに関連するその他の情報を追跡します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-log-model-to-experiment
    parent: model-registry
weight: 3
---

モデル、モデルの依存関係、およびそのモデルに関連するその他の情報を W&B Python SDK を使用して追跡します。

内部的には、W&B は [モデルアーティファクト]({{< relref path="./model-management-concepts.md#model-artifact" lang="ja" >}}) のリネージを作成し、W&B アプリ UI で表示したり、W&B Python SDK を使用してプログラム的に確認することができます。詳細は [モデルリネージマップの作成]({{< relref path="./model-lineage.md" lang="ja" >}}) を参照してください。

## モデルをログする方法

`run.log_model` API を使用してモデルをログします。モデルファイルが保存されているパスを `path` パラメータに提供してください。このパスはローカルファイル、ディレクトリー、または `s3://bucket/path` のような外部バケットへの[リファレンス URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ja" >}}) のいずれかにすることができます。

オプションでモデルアーティファクトの名前を `name` パラメータに指定できます。`name` が指定されていない場合、W&B は入力パスのベース名を実行 ID を前に付けたものとして使用します。

以下のコードスニペットをコピーして貼り付けてください。`<>` で囲まれた値をあなた自身のものに置き換えてください。

```python
import wandb

# W&B run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# モデルをログする
run.log_model(path="<path-to-model>", name="<name>")
```

<details>

<summary>例: Keras モデルを W&B にログする</summary>

以下のコード例は、畳み込みニューラルネットワーク (CNN) モデルを W&B にログする方法を示します。

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run を初期化
run = wandb.init(entity="charlie", project="mnist-project", config=config)

# トレーニングアルゴリズム
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

# モデルをログする
run.log_model(path=full_path, name="MNIST")

# W&B に対して明示的に run の終了を通知します。
run.finish()
```
</details>