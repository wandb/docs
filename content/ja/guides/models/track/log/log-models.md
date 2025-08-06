---
title: モデルをログする
menu:
  default:
    identifier: log-models
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" >}}
# モデルをログする

このガイドでは、W&B run にモデルをログし、そのモデルとやり取りする方法について説明します。

{{% alert %}}
以下の API は、実験管理ワークフローの一部としてモデルをトラッキングする際に便利です。このページに記載されている API を使用して、run にモデルをログし、メトリクス、テーブル、メディア、その他のオブジェクトにアクセスできます。

次のような用途の場合、[W&B Artifacts]({{< relref "/guides/core/artifacts/" >}}) の利用をおすすめします。
- モデル以外のシリアライズされたデータ、例えばデータセットやプロンプトなど、さまざまなバージョンを作成・管理したい場合
- モデルや他のオブジェクトの [リネージグラフ]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" >}}) を可視化したい場合
- これらのメソッドで作成したモデルアーティファクトの[プロパティを更新する]({{< relref "/guides/core/artifacts/update-an-artifact.md" >}})（メタデータ、エイリアス、説明文 など）

W&B Artifacts の詳細や高度なバージョン管理ユースケースについては、[Artifacts]({{< relref "/guides/core/artifacts/" >}}) ドキュメントもご覧ください。
{{% /alert %}}

## モデルを run にログする
[`log_model`]({{< relref "/ref/python/sdk/classes/run.md#log_model" >}}) を使用して、指定したディレクトリー内の内容を含むモデルアーティファクトを run にログできます。[`log_model`]({{< relref "/ref/python/sdk/classes/run.md#log_model" >}}) メソッドは、そのモデルアーティファクトを W&B run の出力としてもマークします。

モデルを W&B run の入力または出力としてマークすると、依存関係や関連付けをトラッキングできます。W&B App の UI でモデルのリネージも閲覧できます。詳細は、[Artifacts]({{< relref "/guides/core/artifacts/" >}}) チャプター内の [リネージグラフの探索とトラバース]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" >}}) ページをご参照ください。

モデルファイルが保存されているパスを `path` パラメータに指定します。パスとしてはローカルファイル、ディレクトリー、あるいは `s3://bucket/path` のような[外部バケットの参照 URI]({{< relref "/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" >}})を指定できます。

`<>` で囲まれた値はご自身のものに置き換えてください。

```python
import wandb

# W&B run を初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルをログ
run.log_model(path="<path-to-model>", name="<name>")
```

`name` パラメータにモデルアーティファクトの名前を任意で指定できます。`name` を指定しない場合は、入力パスのベース名に run ID を付けたものがデフォルトで名前になります。

{{% alert %}}
ご自身、あるいは W&B が割り当てた `name` を必ず把握しておきましょう。モデルのパスを [`use_model`]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) メソッドで取得する際に必要となります。
{{% /alert %}}

パラメータの詳細は API Reference の [`log_model`]({{< relref "/ref/python/sdk/classes/run.md#log_model" >}}) をご覧ください。

<details>

<summary>例：モデルを run にログする</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run を初期化
run = wandb.init(entity="charlie", project="mnist-experiments", config=config)

# ハイパーパラメーターの設定
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

# トレーニング用にモデルを設定
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

この例では、`log_model` が呼ばれると、`MNIST` という名前のモデルアーティファクトが作成され、`model.h5` ファイルがそのアーティファクトに追加されます。ターミナルやノートブックには、その run やモデルの参照情報が出力されます。

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>

## ログ済みモデルをダウンロード・利用する
[`use_model`]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) 関数を使うことで、過去に W&B run にログしたモデルファイルへアクセスし、ダウンロードできます。

取得したいモデルファイルが保存されているアーティファクトの名前を指定してください。指定する名前は、すでにログ済みのモデルアーティファクトの名前と一致している必要があります。

ファイルを `log_model` でログしたときに `name` を指定しなかった場合は、デフォルトで入力パスのベース名に run ID が前置されたものがアーティファクト名となります。

`<>` で囲まれた値はご自身のものに置き換えてください。

```python
import wandb

# run を初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルへアクセスしてダウンロード。ダウンロードされたアーティファクトのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[use_model]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) 関数は、ダウンロードしたモデルファイルのパスを返します。後からこのモデルを参照したい場合は、このパスを控えておきましょう。上記のコード例では、戻り値のパスを変数 `downloaded_model_path` に保存しています。

<details>

<summary>例：ログ済みモデルをダウンロード・利用する</summary>

例えば、下記のコードスニペットでは `use_model` API を呼び出し、取得したいモデルアーティファクトの名前に加え、バージョンやエイリアスも指定しています。API が返すパスを `downloaded_model_path` 変数に格納しています。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョンのセマンティックなニックネームや識別子
model_artifact_name = "fine-tuned-model"

# run を初期化
run = wandb.init(project=project, entity=entity)
# モデルへアクセスしてダウンロード。ダウンロードされたアーティファクトのパスを返します
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

API Reference の [`use_model`]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) を参照し、パラメータや返り値をご確認ください。

## モデルをログして W&B Model Registry にリンクする

{{% alert %}}
[`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) メソッドは、現在レガシー版の W&B Model Registry のみ対応しており、近日中に非推奨となる予定です。新しいモデルレジストリへモデルアーティファクトをリンクする方法については、[Registry linking guide]({{< relref "/guides/core/registry/link_version.md" >}}) をご覧ください。
{{% /alert %}}

[`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) メソッドを使うと、モデルファイルを W&B Run にログし、[W&B Model Registry]({{< relref "/guides/core/registry/model_registry/" >}}) にリンクできます。`registered_model_name` パラメータで指定した名前の登録済みモデルが存在しない場合、新たに作成されます。

モデルをリンクすることは、モデルをチームの中央リポジトリに「ブックマーク」や「公開」するようなイメージです。他のチームメンバーも、そのモデルを確認・利用できるようになります。

モデルをリンクしても、[Registry]({{< relref "/guides/core/registry/model_registry/" >}}) にモデルが複製されたり、プロジェクトから移動されるわけではありません。リンクされたモデルは、元のプロジェクト内のモデルへのポインタです。

[Registry]({{< relref "/guides/core/registry/" >}}) を使うと、用途ごとに優れたモデルを整理したり、モデルのライフサイクルを管理したり、ML ライフサイクル全体でのトラッキングや監査を容易にし、Webhook や ジョブによる[オートメーション]({{< relref "/guides/core/automations/" >}})も実現できます。

*Registered Model* とは、[Model Registry]({{< relref "/guides/core/registry/model_registry/" >}}) 内でリンクされたモデルバージョンの集合やフォルダを指します。登録モデルは、通常、1つの利用ケースやタスクのための候補モデル群を表します。

下記のコードスニペットは、[`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) API でモデルをリンクする例です。`<>` で囲まれた値はご自身のものに置き換えてください。

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

API Reference の [`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) では、省略可能な他のパラメータもご確認いただけます。

`registered-model-name` がすでに Model Registry に存在する登録モデル名に該当する場合、そのモデルに新たなバージョンとしてリンクされます。該当する登録モデルが存在しない場合は新しく作成され、最初のモデルがリンクされます。

例えば、既存の Model Registry に "Fine-Tuned-Review-Autocompletion" という登録モデル（例: [こちら](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)）があり、既にいくつかのモデルバージョン（v0, v1, v2）がリンクされているとします。このとき `link_model` を `registered-model-name="Fine-Tuned-Review-Autocompletion"` で実行すると、そのモデルは v3 としてリンクされます。同名の登録モデルが存在しない場合は新たに作成され、そのモデルが v0 になります。

<details>

<summary>例：モデルをログして W&B Model Registry にリンクする</summary>

例えば、下記のコードスニペットはモデルファイルをログし、"`Fine-Tuned-Review-Autocompletion`" という登録モデル名にリンクする例です。

このためには、`link_model` API を呼び出し、モデルのコンテンツが格納されているローカルファイルパス（`path`）と、リンク先登録モデルの名前（`registered_model_name`）を指定します。

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

{{% alert %}}
ご注意：Registered Model は複数のブックマーク済みモデルバージョンを管理できるコレクションです。
{{% /alert %}}

</details>