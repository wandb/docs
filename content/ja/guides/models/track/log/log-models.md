---
title: モデルをログする
menu:
  default:
    identifier: ja-guides-models-track-log-log-models
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" >}}
# モデルをログする

このガイドでは、W&B run にモデルをログし、それらとやりとりする方法について説明します。

{{% alert %}}
以下の API は、実験管理ワークフローの一部としてモデルを管理するのに便利です。このページに記載されている API を使って run にモデルをログし、メトリクス、テーブル、メディア、その他のオブジェクトにアクセスできます。

もしモデル以外にもデータセットやプロンプトなど、さまざまなシリアライズ済みデータのバージョンを管理したい場合は、[W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の利用をおすすめします。
- モデルやその他のオブジェクトの異なるバージョン（データセットやプロンプトなど）を作成・管理できます。
- モデルや W&B で管理している他のオブジェクトの [リネージグラフ]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) を探索できます。
- 作成したモデルアーティファクトの[プロパティの更新]({{< relref path="/guides/core/artifacts/update-an-artifact.md" lang="ja" >}})（メタデータ、エイリアス、説明など）が行えます。

W&B Artifacts や高度なバージョン管理ユースケースについて詳しくは [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) ドキュメントをご覧ください。
{{% /alert %}}

## モデルを run にログする

[`log_model`]({{< relref path="/ref/python/sdk/classes/run.md#log_model" lang="ja" >}}) を使うと、指定したディレクトリーにある内容を持つモデルアーティファクトを run にログできます。[`log_model`]({{< relref path="/ref/python/sdk/classes/run.md#log_model" lang="ja" >}}) メソッドは、このモデルアーティファクトを W&B run の出力として記録します。

モデルを W&B run の入力または出力としてマークすることで、モデルの依存関係や関連情報も管理できます。W&B App UI でモデルのリネージを見ることができます。詳細は [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) チャプター内の [リネージグラフの探索とトラバース]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) ページをご覧ください。

`path` パラメータにはモデルファイルが保存されているパスを指定してください。パスはローカルファイル、ディレクトリ、または `s3://bucket/path` など外部バケットへの [リファレンス URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ja" >}}) でも構いません。

`<>` で囲まれている値は、ご自身のものに置き換えてください。

```python
import wandb

# W&B の run を初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルをログ
run.log_model(path="<path-to-model>", name="<name>")
```

`name` パラメータにモデルアーティファクトの名前をオプションで指定できます。`name` を指定しない場合、W&B は入力パスのベース名に run ID を付加したものを名前として使います。

{{% alert %}}
自身または W&B が割り当てたモデルの `name` を必ず控えておいてください。[`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) メソッドでモデルのパスを取得する際に必要です。
{{% /alert %}}

パラメータの詳細は APIリファレンスの [`log_model`]({{< relref path="/ref/python/sdk/classes/run.md#log_model" lang="ja" >}}) をご覧ください。

<details>

<summary>例：モデルを run にログする</summary>

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

# トレーニングアルゴリズム
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

# モデルをトレーニング用に設定
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# モデルを保存
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# モデルを W&B run にログ
run.log_model(path=full_path, name="MNIST")
run.finish()
```

ユーザーが `log_model` を実行すると、`MNIST` という名前のモデルアーティファクトが作成され、その中に `model.h5` ファイルが追加されます。ターミナルやノートブックには、run の情報やモデルの保存先が表示されます。

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>

## ログしたモデルをダウンロードして利用する

過去に W&B run にログしたモデルファイルへアクセスしてダウンロードするには、[`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) 関数を利用します。

取得したいモデルファイルを保存したモデルアーティファクトの名前を指定してください。指定する名前は、すでにログされているモデルアーティファクト名と一致している必要があります。

`log_model` でファイルを初めてログした際に `name` を指定しなかった場合は、run ID を前に付けた入力パスのベース名がデフォルトの名前となります。

`<>` で囲まれている他の値も、ご自身のものに置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルにアクセスしダウンロード（返り値はダウンロードしたアーティファクトのパス）
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[use_model]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) 関数は、ダウンロードしたモデルファイルのパスを返します。後ほどこのモデルをリンクする場合は、このパスを控えておいてください。上記コード例では、返り値が `downloaded_model_path` という変数に格納されています。

<details>

<summary>例：ログしたモデルをダウンロード・利用する</summary>

例えば、次のコードでは `use_model` API を呼び出し、取得したいモデルアーティファクトの名前、さらにバージョンやエイリアスも指定しています。API から返却されたパスは `downloaded_model_path` 変数で受け取っています。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョン用の名前や識別子
model_artifact_name = "fine-tuned-model"

# run を初期化
run = wandb.init(project=project, entity=entity)
# モデルにアクセスしてダウンロード（返り値はアーティファクトのパス）
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

パラメータや返り値の型などについては、APIリファレンスの [`use_model`]({{< relref path="/ref/python/sdk/classes/run.md#use_model" lang="ja" >}}) をご参照ください。

## モデルをログし、W&B Model Registry にリンクする

{{% alert %}}
[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) メソッドは現在レガシー版の W&B Model Registry のみで利用可能であり、近く廃止される予定です。新しいモデルレジストリにモデルアーティファクトをリンクする方法は [Registry linking guide]({{< relref path="/guides/core/registry/link_version.md" lang="ja" >}}) をご覧ください。
{{% /alert %}}

[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) メソッドは、モデルファイルを W&B Run にログし、それを [W&B Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) にリンクします。もし登録済みモデルが存在しない場合は、`registered_model_name` パラメータで指定した名前で新たに登録モデルが作成されます。

モデルをリンクすることは、モデルを「ブックマーク」したり「公開」したりして、チームの中央リポジトリで管理できるようにするイメージです。他のチームメンバーもそのモデルを閲覧・活用できます。

リンクしたモデルは、[Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) で重複して保存されたり、プロジェクト外に移動したりすることはありません。リンクされたモデルは、プロジェクト内のオリジナルモデルへの参照（ポインタ）です。

[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を活用することで、タスクごとにベストなモデルを整理したり、モデルのライフサイクルを管理したり、ML の全工程でのトラッキングや監査、そして webhook やジョブによる [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) も簡単に行えます。

*Registered Model* は、[Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) でリンクされたモデルバージョンのコレクションやフォルダーです。通常、1つのユースケースやタスクに対する候補モデル群を指します。

以下のコード例は、[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) API を使ってモデルをリンクする方法を示しています。`<>` で囲んだ値はご自身のものに置き換えてください。

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

オプションパラメータについては、APIリファレンスの [`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ja" >}}) をご参照ください。

`registered-model-name` が Model Registry 内ですでに存在する登録済みモデル名と一致する場合、新しいモデルはその登録モデルにリンクされます。同じ名前の登録済みモデルが存在しない場合は新しく作成され、このモデルが最初にリンクされます。

例えば、Model Registry に "Fine-Tuned-Review-Autocompletion" という登録モデルがすでに存在し、v0、v1、v2 というバージョンがリンクされている場合、`registered-model-name="Fine-Tuned-Review-Autocompletion"` で `link_model` を呼べば、新しいモデルは v3 としてリンクされます。存在しない場合は、新たな登録モデルが作成され、v0（最初のバージョン）としてリンクされます。

<details>

<summary>例：モデルを W&B Model Registry にログ・リンクする</summary>

例えば、以下のコードスニペットでは、モデルファイルをログし、モデルを登録済みモデル `"Fine-Tuned-Review-Autocompletion"` にリンクしています。

この場合、ユーザーは `link_model` API を呼び出し、ローカルファイルパス（`path`）と、紐づけたい登録済みモデル名（`registered_model_name`）を指定しています。

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

{{% alert %}}
補足：登録済みモデルは、ブックマークされたモデルバージョンのコレクションです。
{{% /alert %}}

</details>