---
title: 'チュートリアル: W&B でモデル管理を行う'
description: W&B を使った Model Management の方法を学ぶ
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-walkthrough
    parent: model-registry
weight: 1
---

以下のウォークスルーでは、W&B にモデルをログする方法を紹介します。このウォークスルーを通して、次のことができるようになります。

* MNIST データセットと Keras フレームワークを使ってモデルを作成・トレーニングする
* トレーニングしたモデルを W&B プロジェクトにログする
* 利用したデータセットを作成したモデルの依存関係としてマークする
* モデルを W&B Registry にリンクする
* Registry にリンクしたモデルのパフォーマンスを評価する
* モデルバージョンをプロダクション用としてマークする

{{% alert %}}
* このガイドで紹介している順番にコードスニペットをコピーしてご利用ください。
* Model Registry 固有でないコードは折りたたみセル内に隠されています。
{{% /alert %}}

## セットアップ

始める前に、このウォークスルーに必要な Python の依存ライブラリをインポートしてください。

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split
```

W&B のエンティティを `entity` 変数に入力します。

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

# データセットの作成
(x_train, y_train), (x_eval, y_eval) = generate_raw_data()
```

次に、データセットを W&B にアップロードします。そのためには、[artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) オブジェクトを作成し、その artifact にデータセットを追加します。

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# W&B run を初期化
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

# artifact オブジェクトを作成
artifact_name = "{}_dataset".format(model_use_case_id)
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# wandb.WBValue オブジェクトを artifact に追加
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# artifact に加えた変更を保存
artifact.save()

# この run の終了を W&B に通知
run.finish()
```

{{% alert %}}
ファイル（データセットなど）を artifact に保存することは、モデルのログ時に依存関係を追跡できるため便利です。
{{% /alert %}}


## モデルのトレーニング
前のステップで作成した artifact のデータセットを使ってモデルをトレーニングします。

### データセット Artifact を run の入力として宣言する

前のステップで作成したデータセット artifact を、W&B run の入力として宣言します。artifact を run の入力として宣言することで、どのデータセット（およびそのバージョン）が特定のモデルの学習に使われたかをトラッキングできます。W&B はこの情報から [リネージマップ]({{< relref path="./model-lineage.md" lang="ja" >}}) を作成します。

`use_artifact` API を利用し、データセット artifact を run の入力として宣言し、同時に artifact を取得できます。

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

# データセット artifact を取得
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# データフレームから特定の内容を取得
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```

モデルの入出力をトラッキングする詳細については、[モデルリネージの作成]({{< relref path="./model-lineage.md" lang="ja" >}}) をご覧ください。

### モデルの定義とトレーニング

このウォークスルーでは、Keras を用いて 2D 畳み込みニューラルネットワーク（CNN）を定義し、MNIST データセットの画像を分類します。

<details>
<summary>MNIST データで CNN を学習する</summary>

```python
# config 辞書から値を取り出して変数に格納
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
validation_split = run.config["validation_split"]

# モデルアーキテクチャの作成
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

# トレーニングデータ用のラベルを生成
y_train = keras.utils.to_categorical(y_train, num_classes)

# トレーニングセットとテストセットの作成
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

最後に、モデルをローカルに保存します。

```python
# モデルをローカルに保存
path = "model.h5"
model.save(path)
```
</details>



## モデルを Model Registry へログしてリンクする
[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) API を使い、1つ以上のモデルファイルを W&B run にログし、[W&B Model Registry]({{< relref path="./" lang="ja" >}}) にリンクします。

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

指定した `registered-model-name` 名が未登録の場合、W&B が自動的に Registered Model を作成します。

オプションのパラメータについては、API リファレンスガイドの [`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) をご覧ください。

## モデルのパフォーマンスを評価する
1つまたは複数のモデルのパフォーマンスを評価することは一般的です。

まず、前のステップで W&B に保存した評価用データセット artifact を取得します。

```python
job_type = "evaluate_model"

# run を初期化
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# データセット artifact を取得し、依存関係としてマーク
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# 必要なデータフレームを取得
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

W&B から評価したい[モデルバージョン]({{< relref path="./model-management-concepts.md#model-version" lang="ja" >}}) をダウンロードします。`use_model` API を使ってモデルへアクセス・ダウンロードできます。

```python
alias = "latest"  # エイリアス
name = "mnist_model"  # モデル artifact の名前

# モデルにアクセスし、ダウンロード。ダウンロードされた artifact のパスを返す
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Keras モデルを読み込み、損失値を計算します。

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

最後に、この損失値メトリクスを W&B run にログします。

```python
# # 評価に役立つメトリクス、画像、テーブル、その他のデータをログする
run.log(data={"loss": (loss, _)})
```


## モデルバージョンをプロモートする
[*モデルエイリアス*]({{< relref path="./model-management-concepts.md#model-alias" lang="ja" >}}) で機械学習ワークフローの次のステージに進むモデルバージョンをマークします。1つの Registered Model には 1つ以上のモデルエイリアスが設定できます。モデルエイリアスは1度に1つのモデルバージョンにのみ所属できます。

例えば、モデルのパフォーマンスを評価した後「このモデルはプロダクションに出せる」と判断した場合、そのバージョンへ `production` エイリアスを追加してプロモートします。

{{% alert %}}
`production` エイリアスは、プロダクション対応モデルを示す際によく使われます。
{{% /alert %}}

W&B アプリの UI から、または Python SDK からエイリアスをモデルバージョンに追加可能です。以下の手順は、W&B Model Registry アプリを利用してエイリアスを追加する方法です。

1. [Model Registry アプリ](https://wandb.ai/registry/model) にアクセス
2. Registered Model 名の横の **View details** をクリック
3. **Versions** セクションでプロモートしたいモデルバージョン名の横にある **View** ボタンをクリック
4. **Aliases** 欄の横にあるプラスアイコン（**+**）をクリック
5. 表示されたフィールドに `production` と入力
6. キーボードの Enter を押す

{{< img src="/images/models/promote_model_production.gif" alt="プロダクション用モデルへプロモート" >}}