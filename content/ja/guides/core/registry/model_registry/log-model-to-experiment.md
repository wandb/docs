---
title: Models を追跡
description: W&B Python SDK を使って、モデルとその依存関係、モデルに関連するその他の情報を追跡します。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-log-model-to-experiment
    parent: model-registry
weight: 3
---

W&B Python SDK を使用して、モデル、その依存関係、およびそのモデルに関連するその他の情報をトラッキングします。
内部では、W&B は [モデル アーティファクト]({{< relref path="./model-management-concepts.md#model-artifact" lang="ja" >}}) の リネージ を作成し、これは W&B App または W&B Python SDK からプログラム的に表示できます。詳細については、[モデルリネージ マップの作成]({{< relref path="./model-lineage.md" lang="ja" >}}) を参照してください。

## モデルをログする方法

`run.log_model` API を使用してモデルをログします。モデルファイルが保存されているパスを `path` パラメータに指定します。そのパスは、ローカルファイル、ディレクトリ、または `s3://bucket/path` のような外部 バケット への [参照 URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ja" >}}) のいずれかです。
必要に応じて、`name` パラメータに モデル アーティファクトの名前を指定します。`name` が指定されていない場合、W&B は入力パスのベース名に run ID を先頭に付けて使用します。
次のコードスニペットをコピーして貼り付けます。`< >` で囲まれた値を独自の値に置き換えてください。

```python
import wandb

# W&B run を初期化
run = wandb.init(project="<project>", entity="<entity>")

# モデルをログ
run.log_model(path="<path-to-model>", name="<name>")
```

<details>
<summary>例: Keras モデルを W&B にログする</summary>
次のコード例は、畳み込みニューラルネットワーク（CNN）モデルを W&B にログする方法を示しています。

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run を初期化
run = wandb.init(entity="charlie", project="mnist-project", config=config)

# トレーニング アルゴリズム
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

# W&B に run の終了を明示的に指示します。
run.finish()
```
</details>