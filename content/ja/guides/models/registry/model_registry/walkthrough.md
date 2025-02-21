---
title: 'Tutorial: Use W&B for model management'
description: W&B を使用した Model Management の方法を学ぶ
menu:
  default:
    identifier: ja-guides-models-registry-model_registry-walkthrough
    parent: model-registry
weight: 1
---

以下のウォークスルーでは、モデルを W&B に記録する方法について説明します。ウォークスルーを終えるまでに、以下のことができるようになります。

* MNIST データセットと Keras フレームワークを使用してモデルを作成およびトレーニングします。
* トレーニングしたモデルを W&B の project に記録します。
* 使用したデータセットを作成したモデルへの依存関係としてマークします。
* モデルを W&B のモデルレジストリにリンクします。
* レジストリにリンクするモデルのパフォーマンスを評価します。
* モデルの バージョン を production 用としてマークします。

{{% alert %}}
* このガイドに記載されている順序で コードスニペット をコピーしてください。
* モデルレジストリに固有ではないコードは、折りたたみ可能なセルに隠されています。
{{% /alert %}}

## セットアップ

始める前に、このウォークスルーに必要な Python の依存関係をインポートします。

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split
```

W&B の entity を `entity` 変数に指定します。

```python
entity = "<entity>"
```

### データセットアーティファクトの作成

まず、データセットを作成します。次の コードスニペット は、MNIST データセットをダウンロードする関数を作成します。
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

次に、データセットを W&B にアップロードします。これを行うには、[アーティファクト]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) オブジェクトを作成し、そのアーティファクトにデータセットを追加します。

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# W&B run を初期化
run = wandb.init(entity=entity, project=project, job_type=job_type)

# トレーニングデータ用に W&B テーブル を作成
train_table = wandb.Table(data=[], columns=[])
train_table.add_column("x_train", x_train)
train_table.add_column("y_train", y_train)
train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

# 評価データ用に W&B テーブル を作成
eval_table = wandb.Table(data=[], columns=[])
eval_table.add_column("x_eval", x_eval)
eval_table.add_column("y_eval", y_eval)
eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_eval"])})

# アーティファクトオブジェクトを作成
artifact_name = "{}_dataset".format(model_use_case_id)
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# wandb.WBValue obj をアーティファクトに追加
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# アーティファクトに加えられた変更を永続化
artifact.save()

# W&B にこの run が終了したことを通知
run.finish()
```

{{% alert %}}
（データセットなどの）ファイルをアーティファクトに保存することは、モデルの依存関係を追跡できるため、モデルのログ記録のコンテキストで役立ちます。
{{% /alert %}}

## モデルのトレーニング
前の手順で作成したアーティファクトデータセットを使用して、モデルをトレーニングします。

### データセットアーティファクトを run への入力として宣言する

前の手順で作成したデータセットアーティファクトを W&B の run への入力として宣言します。アーティファクトを run への入力として宣言すると、特定のモデルのトレーニングに使用されたデータセット（およびデータセットの バージョン ）を追跡できるため、モデルのログ記録のコンテキストで特に役立ちます。W&B は、収集された情報を使用して[リネージマップ]({{< relref path="./model-lineage.md" lang="ja" >}})を作成します。

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

モデルの入力と出力を追跡する方法の詳細については、[モデルリネージ]({{< relref path="./model-lineage.md" lang="ja" >}})マップの作成を参照してください。

### モデルの定義とトレーニング

このウォークスルーでは、Keras を使用して MNIST データセットから画像を分類するために、2D 畳み込みニューラルネットワーク（CNN）を定義します。

<details>
<summary>MNIST データで CNN をトレーニング</summary>

```python
# 設定辞書から変数に値を格納して、簡単にアクセスできるようにする
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

# トレーニングデータ用のラベルを生成
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
# ローカルにモデルを保存
path = "model.h5"
model.save(path)
```
</details>

## モデルをログに記録してモデルレジストリにリンクする
[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) API を使用して、1 つ以上のモデルファイルを W&B の run に記録し、[W&B モデルレジストリ]({{< relref path="./" lang="ja" >}})にリンクします。

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

`registered-model-name` に指定した名前がまだ存在しない場合、W&B は登録済みモデルを作成します。

オプションのパラメータの詳細については、API リファレンスガイドの[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}})を参照してください。
## モデルのパフォーマンスを評価する
1 つ以上のモデルのパフォーマンスを評価することは一般的な方法です。

まず、前の手順で W&B に保存されている評価データセットアーティファクトを取得します。

```python
job_type = "evaluate_model"

# run を初期化
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# データセットアーティファクトを取得し、依存関係としてマーク
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# 必要なデータフレームを取得
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

評価する [モデル バージョン]({{< relref path="./model-management-concepts.md#model-version" lang="ja" >}})を W&B からダウンロードします。`use_model` API を使用して、モデルにアクセスしてダウンロードします。

```python
alias = "latest"  # エイリアス
name = "mnist_model"  # モデルアーティファクトの名前

# モデルにアクセスしてダウンロード。ダウンロードされたアーティファクトへのパスを返す
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Keras モデルをロードして、損失を計算します。

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

最後に、損失 メトリクス を W&B の run に記録します。

```python
# # 評価に役立つメトリクス、画像、テーブル、またはその他のデータをログに記録
run.log(data={"loss": (loss, _)})
```

## モデル バージョン を昇格させる
[*モデル エイリアス*]({{< relref path="./model-management-concepts.md#model-alias" lang="ja" >}})を使用して、機械学習 ワークフロー の次の段階に向けてモデル バージョン をマークします。各登録済みモデルには、1 つ以上のモデル エイリアス を設定できます。モデル エイリアス は、一度に 1 つのモデル バージョン にのみ属することができます。

たとえば、モデルのパフォーマンスを評価した後、モデルが production の準備ができていると確信しているとします。そのモデル バージョン を昇格させるには、`production` エイリアス をその特定のモデル バージョン に追加します。

{{% alert %}}
`production` エイリアス は、モデルを production 対応としてマークするために使用される最も一般的な エイリアス の 1 つです。
{{% /alert %}}

モデル バージョン に エイリアス を追加するには、W&B アプリ UI を使用してインタラクティブに追加するか、Python SDK を使用してプログラムで追加できます。次の手順では、W&B モデルレジストリアプリで エイリアス を追加する方法を示します。

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) でモデルレジストリアプリに移動します。
2. 登録済みモデルの名前の横にある [**詳細を表示**] をクリックします。
3. [**バージョン**] セクション内で、昇格させるモデル バージョン の名前の横にある [**表示**] ボタンをクリックします。
4. [**エイリアス**] フィールドの横にあるプラスアイコン（**+**）をクリックします。
5. 表示されるフィールドに「`production`」と入力します。
6. キーボードの Enter キーを押します。

{{< img src="/images/models/promote_model_production.gif" alt="" >}}
```