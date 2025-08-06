---
title: モデルリネージマップを作成する
description: ''
menu:
  default:
    identifier: model-lineage
    parent: model-registry
weight: 7
---

このページでは、レガシーな W&B Model Registry でリネージグラフを作成する方法について説明します。W&B Registry でのリネージグラフについては、[リネージマップの作成と表示]({{< relref "../lineage.md" >}})をご覧ください。

{{% alert %}}
W&B は、レガシーな [W&B Model Registry]({{< relref "/guides/core/registry/model_registry/" >}}) のアセットを新しい [W&B Registry]({{< relref "./" >}}) に移行します。この移行は W&B により完全に管理・実施されるため、ユーザーによる操作は不要です。既存のワークフローへの影響を最小限に抑えつつ、できる限りシームレスに移行されるよう設計されています。詳細は [レガシー Model Registry からの移行]({{< relref "../model_registry_eol.md" >}})をご覧ください。
{{% /alert %}}

モデルアーティファクトを W&B にログすることで活用できる便利な機能の 1 つがリネージグラフです。リネージグラフでは、それぞれの run で記録・利用されたアーティファクトの関係を可視化できます。

つまり、モデルアーティファクトをログした場合、少なくともそのモデルアーティファクトを生成もしくは利用した W&B run を確認できます。[依存関係をトラッキングする]({{< relref "#track-an-artifact-dependency" >}})ことで、そのモデルアーティファクトが入力として利用したアーティファクトも確認できるようになります。

例えば、以下の画像は ML 実験内で作成・利用されたアーティファクトを表しています：

{{< img src="/images/models/model_lineage_example.png" alt="Model lineage graph" >}}

左から右にかけて、画像は以下の流れを示しています：
1. `jumping-monkey-1` という W&B run が `mnist_dataset:v0` という dataset artifact を作成しました。
2. `vague-morning-5` という W&B run が `mnist_dataset:v0` の dataset artifact を使ってモデルのトレーニングを行いました。この W&B run の出力は `mnist_model:v0` というモデルアーティファクトでした。
3. `serene-haze-6` という run が `mnist_model:v0` モデルアーティファクトを利用してモデル評価を行いました。

## アーティファクトの依存関係をトラッキングする

`use_artifact` API を使い、dataset artifact を W&B run の入力として宣言することで依存関係をトラッキングできます。

以下のコードスニペットは、`use_artifact` API の使い方を示しています。

```python
# run を初期化
run = wandb.init(project=project, entity=entity)

# アーティファクトを取得し、依存関係としてマーク
artifact = run.use_artifact(artifact_or_name="name", aliases="<alias>")
```

アーティファクトを取得できたら、そのアーティファクトを用いてモデルの性能評価などに利用できます。

<details>

<summary>例：モデルのトレーニングと、入力データセットの依存関係のトラッキング</summary>

```python
job_type = "train_model"

config = {
    "optimizer": "adam",  # オプティマイザーの設定
    "batch_size": 128,    # バッチサイズ
    "epochs": 5,          # エポック数
    "validation_split": 0.1, # 検証データの割合
}

run = wandb.init(project=project, job_type=job_type, config=config)

version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)

artifact = run.use_artifact(name)

train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")

# config 辞書の値を変数に展開（アクセスしやすくするため）
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# モデルのアーキテクチャーを作成
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

# トレーニングデータのラベルをカテゴリ変数へ変換
y_train = keras.utils.to_categorical(y_train, num_classes)

# トレーニング・テストセットの作成
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)

# モデルの学習
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)

# モデルをローカルに保存
path = "model.h5"
model.save(path)

path = "./model.h5"
registered_model_name = "MNIST-dev"
name = "mnist_model"

run.link_model(path=path, registered_model_name=registered_model_name, name=name)
run.finish()
```

</details>