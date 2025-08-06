---
title: モデルリネージマップを作成する
description: ''
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-model-lineage
    parent: model-registry
weight: 7
---

このページでは、従来の W&B モデルレジストリにおけるリネージグラフの作成について説明します。W&B Registry でのリネージグラフについては、[リネージマップの作成と表示]({{< relref path="../lineage.md" lang="ja" >}})をご覧ください。

{{% alert %}}
W&B は従来の [W&B Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) から新しい [W&B Registry]({{< relref path="./" lang="ja" >}}) への資産移行を行います。この移行は W&B により完全に管理・実行され、ユーザー側での対応は必要ありません。このプロセスは既存のワークフローにほとんど影響しない形でシームレスに進行します。詳細は [従来モデルレジストリからの移行]({{< relref path="../model_registry_eol.md" lang="ja" >}})を参照してください。
{{% /alert %}}

W&B でモデルアーティファクトをログする際の便利な機能の 1 つがリネージグラフです。リネージグラフは、run によって記録されたアーティファクトや、特定の run で使用されたアーティファクトを可視化します。

つまり、モデルアーティファクトをログすると、そのモデルアーティファクトを使用または生成した W&B run を最低限確認できます。また、[依存関係をトラッキング]({{< relref path="#track-an-artifact-dependency" lang="ja" >}}) すると、モデルアーティファクトが利用した入力も参照できます。

例えば、以下の画像は ML 実験全体で作成・利用されたアーティファクトを示しています。

{{< img src="/images/models/model_lineage_example.png" alt="Model lineage graph" >}}

左から右にかけて、画像は次の流れを示します。
1. `jumping-monkey-1` という W&B run が `mnist_dataset:v0` という dataset artifact を作成しました。
2. `vague-morning-5` という W&B run が `mnist_dataset:v0` dataset artifact を使ってモデルのトレーニングを実施。この W&B run の出力が `mnist_model:v0` というモデルアーティファクトです。
3. `serene-haze-6` という run が（`mnist_model:v0`）モデルアーティファクトを用いてモデルを評価しました。

## アーティファクトの依存関係をトラッキングする

依存関係をトラッキングするには、`use_artifact` API を使い、W&B run に入力として dataset artifact を指定します。

以下のコードスニペットは、`use_artifact` API の使い方を示しています。

```python
# run を初期化
run = wandb.init(project=project, entity=entity)

# アーティファクトを取得し、依存関係としてマーク
artifact = run.use_artifact(artifact_or_name="name", aliases="<alias>")
```

アーティファクトを取得したら、そのアーティファクトを使って（例えば）モデルの性能評価などが行えます。

<details>

<summary>例：モデルのトレーニングと、入力データセットの依存関係記録</summary>

```python
job_type = "train_model"

config = {
    "optimizer": "adam",  # オプティマイザー
    "batch_size": 128,    # バッチサイズ
    "epochs": 5,          # エポック数
    "validation_split": 0.1, # 検証用分割率
}

run = wandb.init(project=project, job_type=job_type, config=config)

version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)

artifact = run.use_artifact(name)

train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")

# config 辞書から値を変数に格納し、使いやすくする
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

# トレーニングデータのラベルを生成
y_train = keras.utils.to_categorical(y_train, num_classes)

# トレーニングセットとテストセットを作成
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)

# モデルをトレーニング
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