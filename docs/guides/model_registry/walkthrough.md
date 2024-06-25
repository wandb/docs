---
description: W&Bを使ったモデル管理の方法を学ぶ
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Walkthrough

このウォークスルーでは、W&Bにモデルをログする方法を紹介します。このセッションが終わる頃には以下のことができるようになります：

* MNISTデータセットとKerasフレームワークを使用してモデルを作成しトレーニングする。
* トレーニングしたモデルをW&Bプロジェクトにログする。
* 作成したモデルに依存するデータセットをマークする。
* モデルをW&Bレジストリにリンクする。
* レジストリにリンクしたモデルのパフォーマンスを評価する。
* モデルバージョンをプロダクションに準備完了とマークする。

:::note
* このガイドに示された順序でコードスニペットをコピーしてください。
* Model Registryに固有でないコードは折りたたみセルに隠されています。
:::

## Setting up

始める前に、このウォークスルーに必要なPythonの依存関係をインポートします：

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
```

W&B entityを`entity`変数に提供してください：

```python
entity = "<entity>"
```

### Create a dataset artifact

まず、データセットを作成します。以下のコードスニペットは、MNISTデータセットをダウンロードする関数を作成します：

```python
def generate_raw_data(train_size=6000):
    eval_size = int(train_size / 6)
    (x_train, y_train), (x_eval, y_eval) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_eval = x_eval.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)

    print("Generated {} rows of training data.".format(train_size))
    print("Generated {} rows of eval data.".format(eval_size))

    return (x_train[:train_size], y_train[:train_size]), (
        x_eval[:eval_size],
        y_eval[:eval_size],
    )

# Create dataset
(x_train, y_train), (x_eval, y_eval) = generate_raw_data()
```

次に、データセットをW&Bにアップロードします。これを行うには、[artifact](../artifacts/intro.md)オブジェクトを作成し、そのアーティファクトにデータセットを追加します。

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# Initialize a W&B run
run = wandb.init(entity=entity, project=project, job_type=job_type)

# Create W&B Table for training data
train_table = wandb.Table(data=[], columns=[])
train_table.add_column("x_train", x_train)
train_table.add_column("y_train", y_train)
train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

# Create W&B Table for eval data
eval_table = wandb.Table(data=[], columns=[])
eval_table.add_column("x_eval", x_eval)
eval_table.add_column("y_eval", y_eval)
eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_eval"])})

# Create an artifact object
artifact_name = "{}_dataset".format(model_use_case_id)
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# Add wandb.WBValue obj to the artifact.
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# Persist any changes made to the artifact.
artifact.save()

# Tell W&B this run is finished.
run.finish()
```

:::tip
データセットのようなファイルをアーティファクトに保存することは、モデルの依存関係を追跡するために有用です。
:::

## Train a model
前のステップで作成したアーティファクトデータセットを使用してモデルをトレーニングします。

### Declare dataset artifact as an input to the run

前のステップで作成したデータセットアーティファクトをW&B runの入力として宣言します。これにより、特定のモデルをトレーニングするために使用されたデータセット（およびそのバージョン）を追跡することができます。W&Bは収集した情報を使用して[リネージマップ](./model-lineage.md)を作成します。

`use_artifact` APIを使用して、データセットアーティファクトをrunの入力として宣言し、アーティファクト自体を取得します。

```python
job_type = "train_model"
config = {
    "optimizer": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

# Initialize a W&B run
run = wandb.init(project=project, job_type=job_type, config=config)

# Retrieve the dataset artifact
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# Get specific content from the dataframe
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```

モデルの入力と出力を追跡する方法についての詳細は、[Create model lineage](./model-lineage.md) mapを参照してください。

### Define and train model

このウォークスルーでは、Kerasを使用して2D Convolutional Neural Network (CNN)を定義し、MNISTデータセットの画像を分類します。

<details>
<summary>MNISTデータに対するCNNのトレーニング</summary>

```python
# Store values from our config dictionary into variables for easy accessing
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# Create model architecture
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

# Generate labels for training data
y_train = keras.utils.to_categorical(y_train, num_classes)

# Create training and test set
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)
```
次にモデルをトレーニングします：

```python
# Train the model
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)
```

最後に、ローカルマシンにモデルを保存します：

```python
# Save model locally
path = "model.h5"
model.save(path)
```
</details>

## Log and link a model to the Model Registry
[`link_model`](../../ref/python/run.md#link_model) APIを使用して、モデルをファイルとしてW&B runにログし、W&B Model Registryにリンクします。

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

指定した`registered-model-name`が既に存在しない場合、W&Bは登録されたモデルを自動的に作成します。

オプションのパラメータに関する詳細は、APIリファレンスガイドの[`link_model`](../../ref/python/run.md#link_model)を参照してください。

## Evaluate the performance of a model
モデルのパフォーマンスを評価することは一般的なプラクティスです。

まず、前のステップでW&Bに保存した評価用データセットアーティファクトを取得します。

```python
job_type = "evaluate_model"

# Initialize a run
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# Get dataset artifact, mark it as a dependency
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# Get desired dataframe
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

評価したいモデルのバージョンをW&Bからダウンロードします。`use_model` APIを使用して、モデルにアクセスしてダウンロードします。

```python
alias = "latest"  # alias
name = "mnist_model"  # name of the model artifact

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Kerasモデルをロードし、ロスを計算します：

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

最後に、ロスのメトリクスをW&B runにログします：

```python
# # Log metrics, images, tables, or any data useful for evaluation.
run.log(data={"loss": (loss, _)})
```

## Promote a model version 
[*model alias*](./model-management-concepts.md#model-alias)を使用して、機械学習ワークフローの次のステージにモデルバージョンを準備完了とマークします。各登録モデルには1つ以上のモデルエイリアスが存在できます。モデルエイリアスは特定のモデルバージョンにのみ含まれます。

たとえば、モデルのパフォーマンスを評価した後、そのモデルがプロダクションの準備が整っていると確信した場合、その特定のモデルバージョンに`production`エイリアスを追加します。

:::tip
`production`エイリアスは、モデルをプロダクション対応とマークするためによく使用されるエイリアスの1つです。
:::

W&B App UIを使用してエイリアスを追加することも、Python SDKを使用してプログラム的に追加することもできます。以下のステップでは、W&B Model Registry Appを使用してエイリアスを追加する方法を示します。

1. Model Registry Appに移動します：[https://wandb.ai/registry/model](https://wandb.ai/registry/model)
2. 登録されたモデル名の横にある**View details**をクリックします。
3. **Versions**セクション内で、昇格させたいモデルバージョン名の横にある**View**ボタンをクリックします。
4. **Aliases**フィールドの横にあるプラスアイコン（**+**）をクリックします。
5. 表示されるフィールドに`production`と入力します。
6. キーボードのEnterキーを押します。

![](/images/models/promote_model_production.gif)