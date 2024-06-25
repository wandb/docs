---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Track a model

W&B Python SDK を使用して、モデル、モデルの依存関係、およびそのモデルに関連するその他の情報をトラッキングします。

内部的には、W&B は [model artifact](./model-management-concepts.md#model-artifact) のリネージを作成し、W&B App UI で視覚化したり、W&B Python SDK でプログラム的に確認することができます。詳細については [Create model lineage map](./model-lineage.md) を参照してください。

## モデルをログする方法

`run.log_model` API を使用してモデルをログします。モデルファイルが保存されているパスを `path` パラメータに提供します。このパスはローカルファイル、ディレクトリー、または `s3://bucket/path` のような外部バケットへの [reference URI](../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references) であることができます。

オプションで、`name` パラメータにモデルアーティファクトの名前を提供します。`name` が指定されていない場合、W&B は入力パスのベース名に run ID を付加して使用します。

次のコードスニペットをコピーして貼り付けます。`<>` で囲まれた値を自分のものに置き換えてください。

```python
import wandb

# Initialize a W&B run
run = wandb.init(project="<project>", entity="<entity>")

# Log the model
run.log_model(path="<path-to-model>", name="<name>")
```

<details>

<summary>Example: Log a Keras model to W&B</summary>

次のコード例では、畳み込みニューラルネットワーク (CNN) モデルを W&B にログする方法を示しています。

```python showLineNumbers
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# Initialize a W&B run
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

# モデルをログ
# highlight-next-line
run.log_model(path=full_path, name="MNIST")

# W&B に明示的に run の終了を伝えます。
run.finish()
```
</details>