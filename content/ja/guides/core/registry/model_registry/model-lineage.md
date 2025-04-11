---
title: モデルリネージ マップを作成する
description: ''
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-model-lineage
    parent: model-registry
weight: 7
---

このページでは、従来の W&B Model Registry でのリネージグラフの作成について説明します。W&B Registry でのリネージグラフについて学ぶには、[リネージマップの作成と表示]({{< relref path="../lineage.md" lang="ja" >}})を参照してください。

{{% alert %}}
W&B は、従来の [W&B Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) から新しい [W&B Registry]({{< relref path="./" lang="ja" >}}) へのアセット移行を管理および実行します。この移行は W&B によって完全に管理され、ユーザーによる介入は必要ありません。このプロセスは、既存のワークフローへの影響を最小限に抑えて、可能な限りシームレスに設計されています。[従来の Model Registry からの移行]({{< relref path="../model_registry_eol.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

モデルアーティファクトを W&B にログする際の便利な機能の一つにリネージグラフがあります。リネージグラフは、run によってログされたアーティファクトと特定の run で使用されたアーティファクトを表示します。

つまり、モデルアーティファクトをログする際には、少なくともモデルアーティファクトを使用または生成した W&B run を表示するためのアクセスが可能です。[依存関係を追跡する]({{< relref path="#track-an-artifact-dependency" lang="ja" >}})場合、モデルアーティファクトで使用された入力も見ることができます。

例えば、以下の画像では、ML 実験全体で作成および使用されたアーティファクトが示されています。

{{< img src="/images/models/model_lineage_example.png" alt="" >}}

画像は左から右に向かって次のように示しています。
1. `jumping-monkey-1` W&B run によって `mnist_dataset:v0` のデータセットアーティファクトが作成されました。
2. `vague-morning-5` W&B run は `mnist_dataset:v0` データセットアーティファクトを使用してモデルをトレーニングしました。この W&B run の出力は `mnist_model:v0` というモデルアーティファクトでした。
3. `serene-haze-6` という run は `mnist_model:v0` のモデルアーティファクトを使用してモデルを評価しました。

## アーティファクトの依存関係を追跡

データセットアーティファクトを W&B run の入力として宣言することで、`use_artifact` API を使用して依存関係を追跡できます。

以下のコードスニペットでは、`use_artifact` API の使用方法を示します。

```python
# Run を初期化
run = wandb.init(project=project, entity=entity)

# アーティファクトを取得し、依存関係としてマーク
artifact = run.use_artifact(artifact_or_name="name", aliases="<alias>")
```

アーティファクトを取得した後、そのアーティファクトを使用して（例えば）、モデルのパフォーマンスを評価できます。

<details>

<summary>例: モデルを訓練し、データセットをモデルの入力として追跡</summary>

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

# 設定辞書から変数に値を保存して簡単にアクセス
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# モデルアーキテクチャーの作成
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

# トレーニングデータのラベルを生成
y_train = keras.utils.to_categorical(y_train, num_classes)

# トレーニングセットとテストセットの作成
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)

# モデルのトレーニング
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