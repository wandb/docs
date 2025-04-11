---
title: モデルをログする
menu:
  default:
    identifier: ja-guides-models-track-log-log-models
    parent: log-objects-and-media
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" >}}
# モデルをログする

以下のガイドでは、W&B run にモデルをログし、それと対話する方法を説明します。

{{% alert %}}
以下の API は、実験管理ワークフローの一環としてモデルを追跡するのに便利です。このページに記載されている API を使用して、run にモデルをログし、メトリクス、テーブル、メディア、その他のオブジェクトにアクセスします。

モデル以外にも、データセットやプロンプトなど、シリアライズされたデータの異なるバージョンを作成し追跡したい場合は、[W&B Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を使用することをお勧めします。
- モデルやその他のオブジェクトを W&B で追跡するための [リネージグラフ]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}})を探索します。
- これらのメソッドで作成されたモデル アーティファクトとの対話（プロパティの更新、メタデータ、エイリアス、説明など）を行います。

W&B Artifacts や高度なバージョン管理ユースケースの詳細については、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) ドキュメントをご覧ください。
{{% /alert %}}

## モデルを run にログする
[`log_model`]({{< relref path="/ref/python/run.md#log_model" lang="ja" >}}) を使用して、指定したディレクトリ内にコンテンツを含むモデルアーティファクトをログします。 [`log_model`]({{< relref path="/ref/python/run.md#log_model" lang="ja" >}}) メソッドは、結果のモデルアーティファクトを W&B run の出力としてもマークします。

モデルを W&B run の入力または出力としてマークすると、モデルの依存関係とモデルの関連付けを追跡できます。W&B アプリ UI 内でモデルのリネージを確認します。詳細については、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) チャプターの [アーティファクトグラフを探索して移動する]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ja" >}}) ページを参照してください。

モデルファイルが保存されているパスを `path` パラメータに指定します。パスには、ローカルファイル、ディレクトリ、または `s3://bucket/path` などの外部バケットへの [参照 URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ja" >}}) を指定できます。

`<>` 内に囲まれた値を自分のもので置き換えることを忘れないでください。

import wandb

# W&B run を初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルをログする
run.log_model(path="<path-to-model>", name="<name>")

オプションで、`name` パラメータにモデルアーティファクトの名前を指定できます。`name` が指定されていない場合、W&B は入力パスのベース名に run ID をプレフィックスとして使用して名前を生成します。

{{% alert %}}
モデルに W&B が割り当てた `name` またはユーザーが指定した `name`を追跡してください。モデルのパスを取得するには、[`use_model`]({{< relref path="/ref/python/run#use_model" lang="ja" >}}) メソッドでモデルの名前が必要です。
{{% /alert %}}

`log_model` の詳細については、API リファレンスガイドを参照してください。

<details>

<summary>例: モデルを run にログする</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run を初期化
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

# トレーニング用のモデルを設定
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# モデルを保存
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# モデルを W&B run にログする
run.log_model(path=full_path, name="MNIST")
run.finish()
```

ユーザーが `log_model` を呼び出したとき、`MNIST`という名前のモデルアーティファクトが作成され、ファイル `model.h5` がモデルアーティファクトに追加されました。あなたのターミナルまたはノートブックは、モデルがログされた run に関する情報を見つける場所についての情報を出力します。

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>

## ログされたモデルをダウンロードして使用する
以前に W&B run にログされたモデルファイルにアクセスしてダウンロードするには、[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) 関数を使用します。

取得したいモデルファイルが保存されているモデルアーティファクトの名前を指定します。提供した名前は、既存のログされたモデルアーティファクトの名前と一致している必要があります。

最初に `log_model` でファイルをログした際に `name` を定義しなかった場合、割り当てられたデフォルト名は、入力パスのベース名にrun ID をプレフィックスとして付けたものになります。

`<>` 内に囲まれた他の値を自分のもので置き換えることを忘れないでください。

```python
import wandb

# run を初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルにアクセスしてダウンロードする。ダウンロードされたアーティファクトのパスが返されます
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) 関数は、ダウンロードされたモデルファイルのパスを返します。このパスを追跡して、後でこのモデルにリンクしたい場合に備えてください。上記のコードスニペットでは、返されたパスが `downloaded_model_path` という変数に保存されています。

<details>

<summary>例: ログされたモデルをダウンロードして使用する</summary>

たとえば、以下のコードスニペットでは、ユーザーが `use_model` API を呼び出しています。彼らは取得したいモデルアーティファクトの名前を指定し、またバージョン/エイリアスも提供しています。そして、API から返されるパスを `downloaded_model_path` 変数に保存しました。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョンのセマンティックなニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# run を初期化
run = wandb.init(project=project, entity=entity)
# モデルにアクセスしてダウンロードする。ダウンロードされたアーティファクトのパスが返されます
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

[`use_model`]({{< relref path="/ref/python/run.md#use_model" lang="ja" >}}) API リファレンスガイドでは、利用可能なパラメータや返り値の型についての詳細情報が記載されています。

## モデルを W&B モデルレジストリにログしリンクする

{{% alert %}}
[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) メソッドは、現在のところレガシー W&B モデルレジストリとしか互換性がありませんが、これは間もなく廃止される予定です。モデルアーティファクトを新しいバージョンのモデルレジストリにリンクする方法については、レジストリの[ドキュメント]({{< relref path="/guides/core/registry/link_version.md" lang="ja" >}})をご覧ください。
{{% /alert %}}

[`link_model`]({{< relref path="/ref/python/run.md#link_model" lang="ja" >}}) メソッドを使用して、モデルファイルを W&B run にログし、それを [W&B モデルレジストリ]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) にリンクします。登録されたモデルが存在しない場合、W&B は `registered_model_name` パラメータにあなたが提供した名前で新しいものを作成します。

モデルをリンクすることは、他のチームメンバーが視聴および利用できる中央集権的なチームのリポジトリにモデルを「ブックマーク」または「公開」することに類似しています。

モデルをリンクすると、そのモデルは [Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) に重複されることも、プロジェクトから移動してレジストリに入れられることもありません。リンクされたモデルは、プロジェクト内の元のモデルへのポインターです。

[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を使用して、タスクごとに最高のモデルを整理したり、モデルのライフサイクルを管理したり、MLライフサイクル全体での追跡や監査を容易にしたり、Webhooks やジョブでの下流アクションを[自動化]({{< relref path="/guides/core/automations/" lang="ja" >}}) することができます。

*Registered Model* は、[Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ja" >}}) にリンクされたモデルバージョンのコレクションまたはフォルダーです。登録されたモデルは通常、単一のモデリングユースケースまたはタスクの候補モデルを表します。

以下のコードスニペットは、`link_model` API を使用してモデルをリンクする方法を示しています。`<>` 内に囲まれた他の値を自分のもので置き換えることを忘れないでください。

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

`link_model` API リファレンスガイドでは、オプションのパラメータに関する詳細情報が記載されています。

`registered-model-name` が Model Registry 内に既に存在する登録済みのモデル名と一致する場合、そのモデルはその登録済みモデルにリンクされます。そのような登録済みモデルが存在しない場合、新しいものが作成され、そのモデルが最初にリンクされます。

例えば、既に Model Registry に "Fine-Tuned-Review-Autocompletion"という名前の登録済みモデルが存在し、いくつかのモデルバージョンが既にリンクされていると仮定します: v0, v1, v2。`link_model` を `registered-model-name="Fine-Tuned-Review-Autocompletion"`を使用して呼び出した場合、新しいモデルは既存の登録済みモデルに v3 としてリンクされます。この名前の登録済みモデルが存在しない場合、新しいものが作成され、新しいモデルが v0 としてリンクされます。

<details>

<summary>例: モデルを W&B モデルレジストリにログしリンクする</summary>

例えば、以下のコードスニペットでは、モデルファイルをログし、登録済みのモデル名 `"Fine-Tuned-Review-Autocompletion"`にモデルをリンクする方法を示しています。

これを行うために、ユーザーは `link_model` API を呼び出します。API を呼び出す際に、モデルの内容を示すローカルファイルパス (`path`) と、リンクするための登録済みモデルの名前 (`registered_model_name`) を提供します。

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

{{% alert %}}
リマインダー: 登録済みモデルは、ブックマークされたモデルバージョンのコレクションを管理します。
{{% /alert %}}

</details>