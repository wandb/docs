---
title: モデルをトラッキングする
description: W&B Python SDK を使って、モデル、モデルの依存関係、およびそのモデルに関連するその他の情報をトラッキングできます。
menu:
  default:
    identifier: log-model-to-experiment
    parent: model-registry
weight: 3
---

W&B Python SDK を使って、モデル、モデルの依存関係、およびそのモデルに関連するその他の情報をトラッキングできます。

W&B の内部処理では、[model artifact]({{< relref "./model-management-concepts.md#model-artifact" >}}) のリネージが作成されます。これは W&B App で可視化したり、W&B Python SDK からプログラム的に操作したりできます。詳しくは [Create model lineage map]({{< relref "./model-lineage.md" >}}) をご参照ください。

## モデルのログ方法

`run.log_model` API を使用してモデルをログします。`path` パラメータには、モデルファイルが保存されているパスを指定してください。パスはローカルファイル、ディレクトリ、または `s3://bucket/path` のような外部バケットへの [reference URI]({{< relref "/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" >}}) を指定できます。

また、`name` パラメータでモデル artifact の名前をオプションで指定できます。`name` を指定しない場合は、W&B が入力パスのベース名に run ID を付加したものを利用します。

次のコードスニペットをコピー＆ペーストしてご活用ください。`<>` で囲われた部分はご自身の値に置き換えてください。

```python
import wandb

# W&B run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# モデルをログ
run.log_model(path="<path-to-model>", name="<name>")
```

<details>

<summary>例：Keras モデルを W&B にログする</summary>

このコード例では、畳み込みニューラルネットワーク (CNN) モデルを W&B にログする方法を示しています。

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run を初期化
run = wandb.init(entity="charlie", project="mnist-project", config=config)

# トレーニング用パラメータ取得
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

# W&B に run の終了を明示的に伝える
run.finish()
```
</details>