---
title: モデルをログする
menu:
  default:
    identifier: ja-guides-models-track-log-log-models
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" >}}
# モデルをログする

このガイドでは、W&B の run にモデルをログし、モデルとやり取りする方法を説明します。

{{% alert %}}
以下の API は、実験管理 ワークフローの一部としてモデルをトラッキングするのに役立ちます。このページで紹介する API を使って、モデルを run にログし、メトリクス、テーブル、メディア、その他のオブジェクトにアクセスしてください。

次のような場合は [W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の使用をおすすめします:
- モデル以外のシリアライズ済みデータ（データセット、プロンプトなど）のさまざまなバージョンを作成し、追跡したい場合
- W&B でトラッキングしているモデルやその他のオブジェクトの [リネージ グラフ]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) を探索したい場合
- これらのメソッドで作成されたモデルの Artifacts と対話したい場合（[プロパティの更新]({{< relref path="/guides/core/artifacts/update-an-artifact.md" lang="ja" >}}) など。メタデータ、エイリアス、説明）

W&B Artifacts と高度なバージョン管理のユースケースについては、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) ドキュメントをご覧ください。
{{% /alert %}}

## run にモデルをログする
[`log_model`]({{< relref path="/ref/python/sdk/classes/run.md#log_model" lang="ja" >}}) を使って、指定したディレクトリー内の内容を含むモデル artifact をログします。[`log_model`]({{< relref path="/ref/python/sdk/classes/run.md#log_model" lang="ja" >}}) メソッドは、作成されるモデル artifact を W&B の run の出力としてもマークします。

モデルを W&B の run の入力または出力としてマークすると、モデルの依存関係や関連付けを追跡できます。モデルのリネージは W&B の App UI で確認できます。詳しくは、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) チャプターの [Explore and traverse artifact graphs]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) を参照してください。

モデル ファイルが保存されているパスを `path` パラメータに指定します。パスにはローカル ファイル、ディレクトリー、または `s3://bucket/path` のような外部バケットへの [参照 URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ja" >}}) を指定できます。

`<>` で囲まれた値はご自身の値に置き換えてください。

```python
import wandb

# W&B の run を初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルをログする
run.log_model(path="<path-to-model>", name="<name>")
```

任意で `name` にモデル artifact の名前を指定できます。`name` を指定しない場合、W&B は入力パスのベース名に run ID を前置したものを名前として使用します。

{{% alert %}}
自分で、または W&B によってモデルに割り当てられた `name` を控えておいてください。後で [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) メソッドでモデルのパスを取得する際に、この名前が必要です。
{{% /alert %}}

パラメータについては API リファレンスの [`log_model`]({{< relref path="/ref/python/sdk/classes/run.md#log_model" lang="ja" >}}) を参照してください。

<details>

<summary>例: run にモデルをログする</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B の run を初期化
run = wandb.init(entity="charlie", project="mnist-experiments", config=config)

# ハイパーパラメーター
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

# トレーニング アルゴリズム
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

# トレーニング用にモデルを設定
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# モデルを保存
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# W&B の run にモデルをログする
run.log_model(path=full_path, name="MNIST")
run.finish()
```

ユーザーが `log_model` を呼び出すと、`MNIST` という名前のモデル artifact が作成され、`model.h5` ファイルがそのモデル artifact に追加されました。ターミナルやノートブックには、モデルがログされた run の詳細を確認できる場所が表示されます。

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>


## ログ済みモデルをダウンロードして使用する
[`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) 関数を使って、W&B の run に以前ログしたモデル ファイルにアクセスし、ダウンロードします。

取得したいモデル ファイルが保存されているモデル artifact の名前を指定してください。指定する名前は、既にログ済みのモデル artifact の名前と一致している必要があります。

`log_model` でファイルをログした際に `name` を定義していない場合、割り当てられるデフォルト名は、入力パスのベース名の前に run ID を付けたものです。

`<>` で囲まれた値はご自身の値に置き換えてください:
 
```python
import wandb

# run を初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルにアクセスしてダウンロードします。ダウンロードされた artifact へのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[use_model]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) 関数は、ダウンロードされたモデル ファイルのパスを返します。後でこのモデルをリンクしたい場合に備えて、このパスを控えておいてください。上のコードスニペットでは、返されたパスは `downloaded_model_path` という変数に保存されています。

<details>

<summary>例: ログ済みモデルをダウンロードして使用する</summary>

例えば、次のコードスニペットではユーザーが `use_model` API を呼び出しています。取得したいモデル artifact の名前を指定し、併せてバージョン/エイリアスも指定しています。その後、API が返すパスを `downloaded_model_path` 変数に保存しています。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデル バージョンの意味的なニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# run を初期化
run = wandb.init(project=project, entity=entity)
# モデルにアクセスしてダウンロードします。ダウンロードされた artifact へのパスを返します
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

パラメータと戻り値については、API リファレンスの [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) を参照してください。

## W&B Model Registry にモデルをログしてリンクする

{{% alert %}}
[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) メソッドは現在、まもなく非推奨となるレガシーの W&B Model Registry としか互換性がありません。新しいエディションの Model Registry にモデル artifact をリンクする方法は、[レジストリへのリンク ガイド]({{< relref path="/guides/core/registry/link_version.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) メソッドを使うと、モデル ファイルを W&B の Run にログし、それを [W&B Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) にリンクできます。既存の registered model がない場合は、`registered_model_name` パラメータで指定した名前で新しいものが作成されます。

モデルのリンクは、チームの他のメンバーが閲覧・利用できる中央集約のモデル リポジトリに、そのモデルを「ブックマーク」または「公開」するイメージです。

モデルをリンクしても、そのモデルが [Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) 内で複製されたり、プロジェクトからレジストリに移動されたりすることはありません。リンクされたモデルは、プロジェクト内の元のモデルを指すポインターです。

[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を使うと、タスクごとに優れたモデルを整理し、モデルのライフサイクルを管理し、ML ライフサイクル全体での容易な追跡と監査を促進し、Webhook やジョブで下流アクションを [自動化]({{< relref path="/guides/core/automations/" lang="ja" >}}) できます。

_Registered Model_ は、[Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) にリンクされたモデル バージョンのコレクション（フォルダー）のことです。Registered models は、単一のモデリング ユースケースやタスクの候補モデルを表すのが一般的です。

次のコードスニペットは、[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) API でモデルをリンクする方法を示しています。`<>` で囲まれた値はご自身の値に置き換えてください:

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

省略可能なパラメータについては、API リファレンス ガイドの [`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) を参照してください。

`registered-model-name` が Model Registry 内に既に存在する registered model の名前と一致する場合、そのモデルはその registered model にリンクされます。そのような registered model が存在しない場合は新しく作成され、そのモデルが最初にリンクされます。

例えば、Model Registry に "Fine-Tuned-Review-Autocompletion" という名前の既存の registered model があり（[こちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models) の例を参照）、すでに v0、v1、v2 のいくつかのモデル バージョンがリンクされているとします。`registered-model-name="Fine-Tuned-Review-Autocompletion"` として `link_model` を呼び出すと、新しいモデルは v3 としてこの既存の registered model にリンクされます。この名前の registered model が存在しない場合は新しく作成され、新しいモデルは v0 としてリンクされます。


<details>

<summary>例: W&B Model Registry にモデルをログしてリンクする</summary>

例えば、次のコードスニペットはモデル ファイルをログし、モデルを registered model 名 `"Fine-Tuned-Review-Autocompletion"` にリンクしています。

これを行うために、ユーザーは `link_model` API を呼び出します。呼び出し時に、モデルの内容を指すローカル ファイルパス（`path`）と、リンク先の registered model の名前（`registered_model_name`）を指定します。

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

{{% alert %}}
リマインダー: registered model は、ブックマークされたモデル バージョンのコレクションを格納します。
{{% /alert %}}

</details>