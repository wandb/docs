---
title: Log models
menu:
  default:
    identifier: ja-guides-models-track-log-log-models
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" >}}
# モデルのログ

以下のガイドでは、モデルを W&B の run に記録し、それらとやり取りする方法について説明します。

{{% alert %}}
以下の API は、実験管理 ワークフローの一部としてモデルを追跡するのに役立ちます。このページにリストされている API を使用して、モデルを run に記録し、メトリクス、テーブル、メディア、その他のオブジェクトにアクセスします。

以下を行いたい場合は、[W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を使用することをお勧めします。
- モデル以外に、データセット、プロンプトなど、シリアライズされたデータのさまざまなバージョンを作成および追跡する。
- W&B で追跡されるモデルやその他のオブジェクトの[リネージグラフ]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}})を探索する。
- [プロパティの更新]({{< relref path="/guides/core/artifacts/update-an-artifact.md" lang="ja" >}}) (メタデータ、エイリアス、および説明) など、これらのメソッドが作成したモデル artifacts を操作する

W&B Artifacts と高度な バージョン管理 の ユースケース の詳細については、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) のドキュメントを参照してください。
{{% /alert %}}

## モデルを run に記録する
[`log_model`]({{< relref path="/ref/python/run.md#log_model" lang="ja" >}}) を使用して、指定したディレクトリー内のコンテンツを含むモデル artifact を記録します。[`log_model`]({{< relref path="/ref/python/run.md#log_model" lang="ja" >}}) メソッドは、結果のモデル artifact を W&B run の出力としてもマークします。

モデルを W&B run の入力または出力としてマークすると、モデルの依存関係とモデルの関連付けを追跡できます。W&B App UI 内でモデルの リネージ を表示します。詳細については、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) チャプターの [アーティファクトグラフの探索とトラバース]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) ページを参照してください。

モデルファイルが保存されているパスを `path` パラメータに指定します。パスは、ローカルファイル、ディレクトリー、または `s3://bucket/path` などの外部 バケット への[参照 URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ja" >}})にすることができます。

`<>` で囲まれた値は必ず独自の値に置き換えてください。

```python
import wandb

# W&B run を初期化する
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルを記録する
run.log_model(path="<path-to-model>", name="<name>")
```

オプションで、`name` パラメータにモデル artifact の名前を指定します。`name` が指定されていない場合、W&B は入力パスの basename に run ID を付加したものを名前として使用します。

{{% alert %}}
あなた、または W&B がモデルに割り当てる `name` を追跡してください。[`use_model`]({{< relref path="/ref/python/run#use_model" lang="ja" >}}) メソッドでモデルパスを取得するには、モデルの名前が必要です。
{{% /alert %}}

可能なパラメータの詳細については、API Reference ガイドの [`log_model`]({{< relref path="/ref/python/run.md#log_model" lang="ja" >}}) を参照してください。

<details>

<summary>例: モデルを run に記録する</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run を初期化する
run = wandb.init(entity="charlie", project="mnist-experiments", config=config)

# ハイパーパラメータ
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

# トレーニング用にモデルを構成する
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# モデルを保存する
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# モデルを W&B run に記録する
run.log_model(path=full_path, name="MNIST")
run.finish()
```

ユーザーが `log_model` を呼び出すと、`MNIST` という名前のモデル artifact が作成され、ファイル `model.h5` がモデル artifact に追加されました。ターミナルまたは ノートブック に、モデルが記録された run に関する情報が記載された場所が表示されます。

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>


## ログに記録されたモデルをダウンロードして使用する
[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) 関数を使用して、以前に W&B run に記録されたモデルファイルにアクセスしてダウンロードします。

取得するモデルファイルが保存されているモデル artifact の名前を指定します。指定する名前は、既存のログに記録されたモデル artifact の名前と一致する必要があります。

`log_model` でファイルを最初にログに記録したときに `name` を定義しなかった場合、割り当てられるデフォルトの名前は、入力パスの basename に run ID を付加したものです。

`<>` で囲まれた他の値は必ず独自の値に置き換えてください。
 
```python
import wandb

# run を初期化する
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルにアクセスしてダウンロードします。ダウンロードされた artifact へのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[use_model]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) 関数は、ダウンロードされたモデルファイルのパスを返します。後でこのモデルをリンクする場合は、このパスを追跡してください。上記のコードスニペットでは、返されたパスは `downloaded_model_path` という変数に保存されます。

<details>

<summary>例: ログに記録されたモデルをダウンロードして使用する</summary>

たとえば、上記のコードスニペットでは、ユーザーが `use_model` API を呼び出しました。フェッチするモデル artifact の名前を指定し、バージョン/エイリアス も指定しました。次に、API から返されるパスを `downloaded_model_path` 変数に保存しました。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョンのセマンティックニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# run を初期化する
run = wandb.init(project=project, entity=entity)
# モデルにアクセスしてダウンロードします。ダウンロードされた artifact へのパスを返します
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

可能なパラメータと戻り値の型の詳細については、API Reference ガイドの [`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) を参照してください。

## モデルをログに記録して W&B Model Registry にリンクする

{{% alert %}}
[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) メソッドは現在、間もなく非推奨になる従来の W&B Model Registry とのみ互換性があります。モデル artifact を新しいエディションのモデルレジストリにリンクする方法については、レジストリ[ドキュメント]({{< relref path="../../registry/link_version.md" lang="ja" >}})を参照してください。
{{% /alert %}}

[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) メソッドを使用して、モデルファイルを W&B run に記録し、[W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) にリンクします。登録済みモデルが存在しない場合、W&B は `registered_model_name` パラメータに指定した名前で新しいモデルを作成します。

{{% alert %}}
モデルのリンクは、チームの他のメンバーが表示および使用できるモデルの一元化されたチーム リポジトリへのモデルの「ブックマーク」または「公開」に似ていると考えることができます。

モデルをリンクすると、そのモデルは[Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) に複製されないことに注意してください。また、そのモデルは プロジェクト から移動して レジストリ に導入されることもありません。リンクされたモデルは、プロジェクト 内の元のモデルへのポインタです。

[Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) を使用して、タスクごとに最適なモデルを整理し、モデルのライフサイクルを管理し、ML ライフサイクル全体で簡単な追跡と監査を容易にし、Webhooks または ジョブ でダウンストリーム アクションを[自動化]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ja" >}})します。
{{% /alert %}}

*Registered Model* は、[Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) 内のリンクされたモデルバージョンのコレクションまたはフォルダーです。登録済みモデルは通常、単一のモデリング ユースケース または タスク の候補モデルを表します。

上記のコードスニペットは、[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) API を使用してモデルをリンクする方法を示しています。`<>` で囲まれた他の値は必ず独自の値に置き換えてください。

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

オプションのパラメータの詳細については、API Reference ガイドの [`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) を参照してください。

`registered-model-name` が Model Registry 内に既に存在する登録済みモデルの名前と一致する場合、モデルはその登録済みモデルにリンクされます。そのような登録済みモデルが存在しない場合は、新しいモデルが作成され、モデルが最初にリンクされます。

たとえば、Model Registry に「Fine-Tuned-Review-Autocompletion」という名前の既存の登録済みモデルがあるとします ([こちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)の例を参照)。また、いくつかのモデルバージョンが既にリンクされているとします: v0、v1、v2。`registered-model-name="Fine-Tuned-Review-Autocompletion"` を指定して `link_model` を呼び出すと、新しいモデルはこの既存の登録済みモデルに v3 としてリンクされます。この名前の登録済みモデルが存在しない場合は、新しいモデルが作成され、新しいモデルが v0 としてリンクされます。


<details>

<summary>例: モデルをログに記録して W&B Model Registry にリンクする</summary>

たとえば、上記のコードスニペットはモデルファイルをログに記録し、モデルを登録済みモデル名 `"Fine-Tuned-Review-Autocompletion"` にリンクします。

これを行うために、ユーザーは `link_model` API を呼び出します。API を呼び出すときに、モデルのコンテンツを指すローカル ファイルパス (`path`) と、リンクする登録済みモデルの名前 (`registered_model_name`) を指定します。

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

{{% alert %}}
リマインダー: 登録済みモデルには、ブックマークされたモデルバージョンのコレクションが格納されます。
{{% /alert %}}

</details>
