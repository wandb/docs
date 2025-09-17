---
title: モデルのリネージ マップを作成
description: ''
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-model-lineage
    parent: model-registry
weight: 7
---

このページでは、レガシー W&B Model Registry におけるリネージグラフの作成について説明します。W&B Registry におけるリネージグラフの詳細は「[リネージマップの作成と表示]({{< relref path="../lineage.md" lang="ja" >}})」を参照してください。

{{% alert %}}
W&B は、アセットをレガシー [W&B Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) から新しい [W&B Registry]({{< relref path="./" lang="ja" >}}) へ移行します。この移行は W&B によって完全に管理・トリガーされ、ユーザーの作業は不要です。既存のワークフローへの影響を最小限に抑え、可能な限りシームレスになるよう設計されています。詳細は「[レガシー Model Registry からの移行]({{< relref path="../model_registry_eol.md" lang="ja" >}})」を参照してください。
{{% /alert %}}

W&B にモデルのアーティファクトをログすると、リネージグラフを利用できます。リネージグラフは、run によってログされたアーティファクトと、特定の run が使用したアーティファクトを可視化します。

つまり、モデルのアーティファクトをログしておけば、少なくともそれを使用または生成した W&B の run が表示されます。[依存関係を追跡する]({{< relref path="#track-an-artifact-dependency" lang="ja" >}}) 場合は、モデルのアーティファクトが使用した入力も表示されます。

例えば、以下の画像は ML の実験全体で作成および使用されたアーティファクトを示しています。

{{< img src="/images/models/model_lineage_example.png" alt="モデルのリネージグラフ" >}}

左から右へ、画像は以下を示しています。
1. `jumping-monkey-1` W&B run は `mnist_dataset:v0` データセットアーティファクトを作成しました。
2. `vague-morning-5` W&B run は `mnist_dataset:v0` データセットアーティファクトを使用してモデルをトレーニングしました。この W&B run の出力は、`mnist_model:v0` というモデルアーティファクトでした。
3. `serene-haze-6` という run は、モデルアーティファクト (`mnist_model:v0`) を使用してモデルを評価しました。

## アーティファクトの依存関係を追跡する

依存関係を追跡するには、`use_artifact` API を使って、データセットのアーティファクトを W&B の run の入力として宣言します。

以下のコードスニペットは、`use_artifact` API の使用方法を示しています。

```python
# run を初期化します
run = wandb.init(project=project, entity=entity)

# アーティファクトを取得し、依存関係としてマークします
artifact = run.use_artifact(artifact_or_name="name", aliases="<alias>")
```

アーティファクトを取得したら、例えばそれを使ってモデルの性能を評価できます。

<details>

<summary>例: モデルをトレーニングし、データセットをモデルの入力として追跡する</summary>

```python
job_type = "train_model"

config = {
    "optimizer": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

run = wandb.init(project=project, job_type=job_type, config=config)

version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)

artifact = run.use_artifact(name)

train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")

# 設定辞書から値を変数に格納し、簡単にアクセスできるようにします
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# モデル アーキテクチャを作成します
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

# トレーニングデータのラベルを生成します
y_train = keras.utils.to_categorical(y_train, num_classes)

# トレーニングセットとテストセットを作成します
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)

# モデルをトレーニングします
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)

# モデルをローカルに保存します
path = "model.h5"
model.save(path)

path = "./model.h5"
registered_model_name = "MNIST-dev"
name = "mnist_model"

run.link_model(path=path, registered_model_name=registered_model_name, name=name)
run.finish()
```

</details>