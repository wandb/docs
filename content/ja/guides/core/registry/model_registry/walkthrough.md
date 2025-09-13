---
title: 'チュートリアル: W&B を使ったモデル管理'
description: W&B で Model Management を行う方法を学ぶ
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-walkthrough
    parent: model-registry
weight: 1
---

以下のウォークスルーでは、W&B にモデルをログする方法を説明します。このウォークスルーを完了すると、次のことができるようになります。
* MNIST データセットと Keras フレームワークを使用してモデルを作成し、トレーニングする。
* トレーニングしたモデルを W&B の Projects にログする。
* 使用したデータセットを、作成したモデルの依存関係としてマークする。
* モデルを W&B Model Registry にリンクする。
* レジストリにリンクしたモデルのパフォーマンスを評価する。
* プロダクションに対応するモデルバージョンとしてマークする。
{{% alert %}}
* このガイドで提示されている順序でコードスニペットをコピーしてください。
* Model Registry に固有ではないコードは、折りたたみ可能なセルに隠されています。
{{% /alert %}}

## 設定
始める前に、このウォークスルーに必要な Python の依存関係をインポートします。
```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split
```
`entity` 変数に W&B の Entities を指定します。
```python
entity = "<entity>"
```

### データセット Artifact を作成する
まず、データセットを作成します。以下のコードスニペットは、MNIST データセットをダウンロードする関数を作成します。
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
次に、データセットを W&B にアップロードします。これを行うには、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) オブジェクトを作成し、その Artifact にデータセットを追加します。
```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# W&B の run を初期化
run = wandb.init(entity=entity, project=project, job_type=job_type)

# トレーニングデータ用の W&B Table を作成
train_table = wandb.Table(data=[], columns=[])
train_table.add_column("x_train", x_train)
train_table.add_column("y_train", y_train)
train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

# 評価データ用の W&B Table を作成
eval_table = wandb.Table(data=[], columns=[])
eval_table.add_column("x_eval", x_eval)
eval_table.add_column("y_eval", y_eval)
eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_eval"])})

# Artifact オブジェクトを作成
artifact_name = "{}_dataset".format(model_use_case_id)
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# wandb.WBValue オブジェクトを Artifact に追加
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# Artifact への変更を永続化
artifact.save()

# この run が完了したことを W&B に知らせる
run.finish()
```
{{% alert %}}
ファイル（データセットなど）を Artifacts に保存することは、モデルの依存関係を追跡できるため、モデルのログ記録のコンテキストで役立ちます。
{{% /alert %}}

## モデルをトレーニングする
前のステップで作成した Artifacts のデータセットでモデルをトレーニングします。

### データセット Artifact を run への入力として宣言する
前のステップで作成したデータセット Artifact を W&B の run への入力として宣言します。これは、特定のモデルのトレーニングに使用されたデータセット（およびデータセットのバージョン）を追跡できるため、モデルのログ記録のコンテキストで特に役立ちます。W&B は収集された情報を使用して、[リネージマップ]({{< relref path="./model-lineage.md" lang="ja" >}}) を作成します。
`use_artifact` API を使用して、データセット Artifact を run の入力として宣言し、Artifact 自体を取得します。
```python
job_type = "train_model"
config = {
    "optimizer": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

# W&B の run を初期化
run = wandb.init(project=project, job_type=job_type, config=config)

# データセット Artifact を取得
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# データフレームから必要な内容を取得
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```
モデルの入力と出力を追跡する方法の詳細については、[モデルリネージマップの作成]({{< relref path="./model-lineage.md" lang="ja" >}}) を参照してください。

### モデルを定義し、トレーニングする
このウォークスルーでは、Keras を使用して 2D 畳み込みニューラルネットワーク (CNN) を定義し、MNIST データセットの画像を分類します。
<details>
<summary>MNIST データで CNN をトレーニングする</summary>

```python
# config 辞書の値を変数に取り出して使いやすくする
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
```
次に、モデルをトレーニングします。
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
最後に、モデルをローカルマシンに保存します。
```python
# モデルをローカルに保存
path = "model.h5"
model.save(path)
```
</details>

## モデルをログして Model Registry にリンクする
[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) API を使用して、1 つ以上のファイルを W&B の run にログし、[W&B Model Registry]({{< relref path="./" lang="ja" >}}) にリンクします。
```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```
`registered-model-name` に指定した名前がまだ存在しない場合、W&B は自動的に registered model を作成します。
オプションのパラメータについては、API リファレンスガイドの [`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) を参照してください。

## モデルのパフォーマンスを評価する
1 つ以上のモデルのパフォーマンスを評価することは一般的な慣行です。
まず、以前のステップで W&B に保存された評価用データセット Artifact を取得します。
```python
job_type = "evaluate_model"

# run を初期化
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# データセット Artifact を取得し、依存関係としてマーク
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# 目的のデータフレームを取得
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```
評価したい [モデルバージョン]({{< relref path="./model-management-concepts.md#model-version" lang="ja" >}}) を W&B からダウンロードします。`use_model` API を使用してモデルにアクセスし、ダウンロードします。
```python
alias = "latest"  # エイリアス
name = "mnist_model"  # model Artifact の名前

# モデルにアクセスしてダウンロードする。ダウンロードされた Artifact へのパスを返す
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```
Keras モデルをロードし、損失を計算します。
```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```
最後に、損失メトリクスを W&B の run にログします。
```python
# # 評価に役立つメトリクス、画像、テーブル、その他のデータをログする
run.log(data={"loss": (loss, _)})
```

## モデルバージョンを昇格させる
機械学習ワークフローの次の段階に向けてモデルバージョンを準備するために、[*モデルエイリアス*]({{< relref path="./model-management-concepts.md#model-alias" lang="ja" >}}) でマークします。各 registered model は 1 つ以上のモデルエイリアスを持つことができます。モデルエイリアスは、一度に 1 つのモデルバージョンにのみ属することができます。
たとえば、モデルのパフォーマンスを評価した後、そのモデルがプロダクションに対応できると確信しているとします。そのモデルバージョンを昇格させるには、特定のモデルバージョンに `production` エイリアスを追加します。
{{% alert %}}
`production` エイリアスは、モデルをプロダクション対応としてマークするために最も一般的に使用されるエイリアスの 1 つです。
{{% /alert %}}
W&B App UI で対話的に、または Python SDK を使用してプログラムで、モデルバージョンにエイリアスを追加できます。以下の手順は、W&B Model Registry App でエイリアスを追加する方法を示しています。
1. [Model Registry App](https://wandb.ai/registry/model) に移動します。
2. registered model の名前の横にある **View details** をクリックします。
3. **Versions** セクション内で、昇格させたいモデルバージョンの名前の横にある **View** ボタンをクリックします。
4. **Aliases** フィールドの横にあるプラスアイコン (**+**) をクリックします。
5. 表示されるフィールドに `production` と入力します。
6. キーボードの Enter キーを押します。
{{< img src="/images/models/promote_model_production.gif" alt="モデルをプロダクションに昇格する" >}}