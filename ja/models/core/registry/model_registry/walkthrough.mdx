---
title: 'Tutorial: W&B を使ったモデル管理'
description: W&B を活用したモデル管理の使い方を学ぶ
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-walkthrough
    parent: model-registry
weight: 1
---

W&B にモデルをログする方法を示す次のウォークスルーに従ってください。このウォークスルーの終わりまでに次のことができるようになります：

* MNIST データセットと Keras フレームワークを使用してモデルを作成およびトレーニングします。
* トレーニングしたモデルを W&B プロジェクトにログします。
* 作成したモデルの依存関係として使用したデータセットをマークします。
* モデルを W&B Registry にリンクします。
* レジストリにリンクしたモデルのパフォーマンスを評価します。
* モデルバージョンをプロダクション用に準備完了としてマークします。

{{% alert %}}
* このガイドで提示された順にコードスニペットをコピーしてください。
* モデルレジストリに固有でないコードは折りたたみ可能なセルに隠されています。
{{% /alert %}}

## セットアップ

始める前に、このウォークスルーに必要な Python の依存関係をインポートします：

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split
```

`entity` 変数に W&B エンティティを指定します：

```python
entity = "<entity>"
```

### データセット アーティファクトを作成する

まず、データセットを作成します。次のコードスニペットは、MNIST データセットをダウンロードする関数を作成します：
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

# データセットを作成
(x_train, y_train), (x_eval, y_eval) = generate_raw_data()
```

次に、データセットを W&B にアップロードします。これを行うには、[artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) オブジェクトを作成し、そのアーティファクトにデータセットを追加します。

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# W&B run を初期化
run = wandb.init(entity=entity, project=project, job_type=job_type)

# トレーニングデータ用に W&B Table を作成
train_table = wandb.Table(data=[], columns=[])
train_table.add_column("x_train", x_train)
train_table.add_column("y_train", y_train)
train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

# 評価データ用に W&B Table を作成
eval_table = wandb.Table(data=[], columns=[])
eval_table.add_column("x_eval", x_eval)
eval_table.add_column("y_eval", y_eval)
eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_eval"])})

# アーティファクトオブジェクトを作成
artifact_name = "{}_dataset".format(model_use_case_id)
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# wandb.WBValue オブジェクトをアーティファクトに追加
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# アーティファクトに加えられた変更を永続化
artifact.save()

# W&B にこの run が完了したことを知らせます
run.finish()
```

{{% alert %}}
アーティファクトにファイル（データセットなど）を保存することは、モデルの依存関係を追跡できるため、モデルをログに記録するという文脈で便利です。
{{% /alert %}}

## モデルのトレーニング
前のステップで作成したアーティファクトデータセットを使用してモデルをトレーニングします。

### データセットアーティファクトを run の入力として宣言

前のステップで作成したデータセットアーティファクトを W&B run の入力として宣言します。これにより、特定のモデルをトレーニングするために使用されたデータセット（およびデータセットのバージョン）を追跡できるため、モデルをログに記録するという文脈で特に便利です。W&B は収集された情報を使用して、[lineage map]({{< relref path="./model-lineage.md" lang="ja" >}}) を作成します。

`use_artifact` API を使用して、データセットアーティファクトを run の入力として宣言し、アーティファクト自体を取得します。

```python
job_type = "train_model"
config = {
    "optimizer": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

# W&B run を初期化
run = wandb.init(project=project, job_type=job_type, config=config)

# データセットアーティファクトを取得
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# データフレームから特定のコンテンツを取得
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```

モデルの入力と出力を追跡する方法の詳細については、[Create model lineage]({{< relref path="./model-lineage.md" lang="ja" >}}) mapを参照してください。

### モデルの定義とトレーニング

このウォークスルーでは、Keras を使用して MNIST データセットから画像を分類するための 2D 畳み込みニューラルネットワーク (CNN) を定義します。

<details>
<summary>MNIST データに対する CNN のトレーニング</summary>

```python
# 設定辞書から値を取得して変数に格納（アクセスしやすくするため）
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# モデルアーキテクチャを作成
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
```
次に、モデルをトレーニングします：

```python
# モデルをトレーニング
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)
```

最後に、モデルをローカルマシンに保存します：

```python
# モデルをローカルに保存
path = "model.h5"
model.save(path)
```
</details>

## モデルを Model Registry にログし、リンクする
[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) API を使用して、一つまたは複数のファイルを W&B run にログし、それを [W&B Model Registry]({{< relref path="./" lang="ja" >}}) にリンクします。

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

指定した名前の `registered-model-name` がまだ存在しない場合、W&B は登録されたモデルを作成します。

オプションのパラメータに関する詳細は、API リファレンスガイドの [`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) を参照してください。

## モデルのパフォーマンスを評価する
複数のモデルのパフォーマンスを評価するのは一般的な手法です。

まず、前のステップで W&B に保存された評価データセットアーティファクトを取得します。

```python
job_type = "evaluate_model"

# 初期化
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# データセットアーティファクトを取得し、それを依存関係としてマーク
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# 必要なデータフレームを取得
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

評価したい W&B からの[モデルバージョン]({{< relref path="./model-management-concepts.md#model-version" lang="ja" >}}) をダウンロードします。`use_model` API を使用してモデルにアクセスし、ダウンロードします。

```python
alias = "latest"  # エイリアス
name = "mnist_model"  # モデルアーティファクトの名前

# モデルにアクセスしダウンロードします。ダウンロードされたアーティファクトへのパスを返します
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Keras モデルをロードし、損失を計算します：

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

最後に、損失のメトリクスを W&B run にログします：

```python
# メトリクス、画像、テーブル、または評価に役立つデータをログします。
run.log(data={"loss": (loss, _)})
```

## モデルバージョンを昇格する
[*モデルエイリアス*]({{< relref path="./model-management-concepts.md#model-alias" lang="ja" >}}) を使用して、機械学習ワークフローの次のステージに準備が整ったモデルバージョンをマークします。各登録済みモデルは 1 つまたは複数のモデルエイリアスを持つことができます。モデルエイリアスは、1 度に 1 つのモデルバージョンにのみ所属できます。

例えば、モデルのパフォーマンスを評価した後、そのモデルがプロダクションの準備が整ったと確信したとします。モデルバージョンを昇格させるために、特定のモデルバージョンに `production` エイリアスを追加します。

{{% alert %}}
`production` エイリアスは、モデルをプロダクション対応としてマークするために使用される最も一般的なエイリアスの 1 つです。
{{% /alert %}}

W&B アプリ UI を使用してインタラクティブに、または Python SDK を使用してプログラムでモデルバージョンにエイリアスを追加できます。次のステップは、W&B Model Registry App を使用してエイリアスを追加する方法を示しています：

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) の Model Registry App に移動します。
2. 登録されているモデルの名前の横にある **View details** をクリックします。
3. **Versions** セクション内で、プロモーションしたいモデルバージョンの名前の横にある **View** ボタンをクリックします。
4. **Aliases** フィールドの隣にあるプラスアイコン (**+**) をクリックします。
5. 表示されるフィールドに `production` と入力します。
6. キーボードの Enter キーを押します。

{{< img src="/images/models/promote_model_production.gif" alt="" >}}