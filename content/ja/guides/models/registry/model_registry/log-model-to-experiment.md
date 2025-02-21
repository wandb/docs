---
title: Track a model
description: W&B Python SDK を使用して、モデル 、モデル の依存関係、およびその モデル に関連するその他の 情報 を追跡します。
menu:
  default:
    identifier: ja-guides-models-registry-model_registry-log-model-to-experiment
    parent: model-registry
weight: 3
---

W&B Python SDK を使用して、モデル、モデルの依存関係、およびそのモデルに関連するその他の情報を追跡します。

W&B は内部で [モデル artifact ]({{< relref path="./model-management-concepts.md#model-artifact" lang="ja" >}}) のリネージを作成します。これは、W&B App UI で表示したり、W&B Python SDK でプログラム的に表示したりできます。詳細については、[モデルリネージマップの作成]({{< relref path="./model-lineage.md" lang="ja" >}}) を参照してください。

## モデルをログに記録する方法

モデルをログに記録するには、`run.log_model` API を使用します。モデルファイルが保存されているパスを `path` パラメータに指定します。パスは、ローカルファイル、ディレクトリー、または `s3://bucket/path` などの外部バケットへの [参照 URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ja" >}}) にすることができます。

オプションで、`name` パラメータにモデル artifact の名前を指定します。`name` が指定されていない場合、W&B は run ID を先頭に付加した入力パスのベース名を使用します。

次のコードスニペットをコピーして貼り付けます。`<>` で囲まれた 値を必ず独自の値に置き換えてください。

```python
import wandb

# W&B の run を初期化します
run = wandb.init(project="<project>", entity="<entity>")

# モデルをログに記録します
run.log_model(path="<path-to-model>", name="<name>")
```

<details>

<summary> 例：Keras モデルを W&B に記録する </summary>

次のコード例は、畳み込みニューラルネットワーク (CNN) モデルを W&B に記録する方法を示しています。

```python showLineNumbers
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B の run を初期化します
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

# モデルをログに記録
# highlight-next-line
run.log_model(path=full_path, name="MNIST")

# W&B に run を終了するように明示的に指示します。
run.finish()
```
</details>
