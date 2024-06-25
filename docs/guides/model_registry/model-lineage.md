---
displayed_sidebar: default
---


# Create model lineage map
W&B にモデルアーティファクトをログする便利な機能のひとつがリネージグラフです。リネージグラフは、run によって記録されたアーティファクトや、特定の run で使用されたアーティファクトを表示します。

これは、モデルアーティファクトをログすることで、最低限モデルアーティファクトを使用または生成した W&B run を見ることができることを意味します。もし [依存関係を追跡](#track-an-artifact-dependency) する場合、モデルアーティファクトが使用した入力も確認できます。

例えば、次の画像は ML 実験全体で作成され使用されたアーティファクトを示しています。

![](/images/models/model_lineage_example.png)

左から右へ画像は次のように示されています：
1. `jumping-monkey-1` W&B run が `mnist_dataset:v0` データセットアーティファクトを作成しました。
2. `vague-morning-5` W&B run は `mnist_dataset:v0` データセットアーティファクトを使用してモデルをトレーニングしました。この W&B run の出力は `mnist_model:v0` というモデルアーティファクトです。
3. `serene-haze-6` という run がそのモデルアーティファクト (`mnist_model:v0`) を使用してモデルを評価しました。

## Track an artifact dependency

依存関係を追跡するためには、データセットアーティファクトを W&B run に対する入力として `use_artifact` API を使用して宣言します。

次のコードスニペットは `use_artifact` API の使い方を示しています：

```python
# run を初期化
run = wandb.init(project=project, entity=entity)

# アーティファクトを取得し、依存関係としてマーク
artifact = run.use_artifact(artifact_or_name="name", aliases="<alias>")
```

アーティファクトを取得すると、そのアーティファクトを使用してモデルの性能を評価することができます（例えば）。

<details>

<summary>例: モデルをトレーニングし、データセットをモデルの入力として追跡</summary>

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

# highlight-start
artifact = run.use_artifact(name)
# highlight-end

train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")

# 辞書から設定値を取り出し、変数に格納して簡単にアクセスできるようにする
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# モデルアーキテクチャーを作成
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

# highlight-start
run.link_model(path=path, registered_model_name=registered_model_name, name=name)
# highlight-end
run.finish()
```

</details>