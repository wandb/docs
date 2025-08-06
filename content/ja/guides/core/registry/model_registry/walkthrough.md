---
title: 'チュートリアル: W&B でモデル管理を行う'
description: W&B を使った Model Management の方法を学ぶ
menu:
  default:
    identifier: walkthrough_model_registry
    parent: model-registry
weight: 1
---

以下のウォークスルーでは、W&B へのモデルの記録方法を説明します。このウォークスルーの終わりには、以下のことができるようになります。

* MNIST データセットと Keras フレームワークを使ってモデルを作成・学習します
* 学習したモデルを W&B Project にログします
* 使用したデータセットを作成したモデルの依存関係としてマークします
* モデルを W&B Registry にリンクします
* Registry にリンクしたモデルのパフォーマンスを評価します
* モデルバージョンをプロダクション対応としてマークします

{{% alert %}}
* このガイドの順番通りにコードスニペットをコピーしてください。
* Model Registry 固有でないコードは折りたたみセルに隠されています。
{{% /alert %}}

## セットアップ

始める前に、このウォークスルーで必要な Python 依存関係をインポートしてください：

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split
```

`entity` 変数にご自身の W&B entity を指定します：

```python
entity = "<entity>"
```


### データセットアーティファクトの作成

最初に、データセットを作成します。以下のコードスニペットは、MNIST データセットをダウンロードする関数を作成します。
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

次に、データセットを W&B にアップロードします。このためには、[artifact]({{< relref "/guides/core/artifacts/" >}}) オブジェクトを作成し、その artifact にデータセットを追加します。

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# W&B run を初期化
run = wandb.init(entity=entity, project=project, job_type=job_type)

# トレーニングデータの W&B Table を作成
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
アーティファクトにファイル（データセットなど）を保存することは、モデルのログの文脈で非常に役立ちます。モデルの依存関係を追跡できるからです。
{{% /alert %}}


## モデルの学習
前のステップで作成した artifact データセットを使ってモデルを学習します。

### データセットアーティファクトを run の入力として宣言

前のステップで作成したデータセットアーティファクトを W&B run の入力として宣言します。アーティファクトを run の入力として指定することで、その run で使用したデータセット（およびそのバージョン）を追跡できるため、モデルのログにとって非常に有用です。W&B はこの情報を収集し、[リネージマップ]({{< relref "./model-lineage.md" >}}) を作成します。

`use_artifact` API を使用して、データセットアーティファクトを run の入力として宣言しつつ、それ自体も取得できます。

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

# データセットアーティファクトの取得
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# データフレームから必要な内容を取得
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```

モデルの入力と出力の追跡方法について詳しくは、[モデルリネージの作成]({{< relref "./model-lineage.md" >}}) をご覧ください。

### モデルの定義と学習

このウォークスルーでは、Keras を使って 2 次元畳み込みニューラルネットワーク（CNN）を定義し、MNIST データセットの画像を分類します。

<details>
<summary>MNIST データで CNN を学習</summary>

```python
# config 辞書から値を取得しやすいように変数に格納
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
```
次に、モデルを学習します：

```python
# モデルを学習
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)
```

最後に、モデルをローカルに保存します：

```python
# モデルをローカルに保存
path = "model.h5"
model.save(path)
```
</details>



## Model Registry へのモデルの記録とリンク
[`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) API を使って、1つまたは複数のモデルファイルを W&B run に記録し、[W&B Model Registry]({{< relref "./" >}}) にリンクします。

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

指定した `registered-model-name` がまだ存在しない場合、W&B は自動的に Registered Model を作成します。

オプション引数などの詳細は、API リファレンスガイドの [`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) をご覧ください。

## モデルのパフォーマンスを評価する
1つまたは複数のモデルの性能を評価するのは一般的な手順です。

まず、前のステップで W&B に保存した評価用データセットアーティファクトを取得します。

```python
job_type = "evaluate_model"

# Run を初期化
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# データセットアーティファクトの取得と依存関係としてのマーク
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# 必要なデータフレームを取得
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

評価したい [モデルバージョン]({{< relref "./model-management-concepts.md#model-version" >}}) を W&B からダウンロードします。`use_model` API を使うことで、モデルにアクセスしダウンロードできます。

```python
alias = "latest"  # エイリアス
name = "mnist_model"  # モデル artifact の名前

# モデルにアクセス・ダウンロード。ダウンロードした artifact のパスを返す
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Keras モデルをロードし、ロスを計算します：

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

最後に、ロスのメトリクスを W&B run に記録します：

```python
# # 評価に役立つメトリクス、画像、テーブル、または他のデータをログ
run.log(data={"loss": (loss, _)})
```


## モデルバージョンのプロモート
[*モデルエイリアス*]({{< relref "./model-management-concepts.md#model-alias" >}}) を使って、機械学習ワークフローの次のステージに向けてモデルバージョンを準備できます。各 Registered Model には 1つまたは複数のモデルエイリアスを持たせることができます。モデルエイリアスは、同時に 1つのモデルバージョンにしか割り当てられません。

例えば、モデルの性能を評価した後、そのモデルがプロダクションに十分対応できると判断したとします。そのモデルバージョンをプロモートするには、特定のモデルバージョンに `production` エイリアスを追加します。

{{% alert %}}
`production` エイリアスはモデルを本番運用可能としてマークする際によく使われるエイリアス名です。
{{% /alert %}}

モデルバージョンにエイリアスを追加するには、W&B App の UI からインタラクティブに操作するか、Python SDK でプログラム的に追加できます。以下は W&B Model Registry App でエイリアスを追加する手順です：

1. [Model Registry App](https://wandb.ai/registry/model) にアクセスします
2. Registered Model 名の横にある **View details** をクリックします
3. **Versions** セクション内で、プロモートしたいモデルバージョン名の横の **View** ボタンをクリックします
4. **Aliases** フィールドの横にあるプラスアイコン（**+**）をクリックします
5. フィールドに `production` と入力します
6. キーボードの Enter を押します


{{< img src="/images/models/promote_model_production.gif" alt="モデルをプロダクションにプロモート" >}}