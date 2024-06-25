---
displayed_sidebar: default
---


# モデルのログ

このガイドでは、W&Bのrunにモデルをログし、それと対話する方法について説明します。

:::tip
以下のAPIは、実験管理ワークフローの一部としてモデルを追跡するのに便利です。このページにリストされたAPIを使用して、モデルに加えてメトリクス、テーブル、メディア、その他のオブジェクトを迅速にログすることができます。

W&Bは以下の場合、[W&B Artifacts](../../artifacts/intro.md)を使用することを推奨します：
- モデル以外のシリアライズされたデータ（データセット、プロンプトなど）の異なるバージョンを作成し、追跡したい場合。
- モデルやその他のオブジェクトの[リネージグラフ](../../artifacts/explore-and-traverse-an-artifact-graph.md)を探索したい場合。
- これらのメソッドで作成されたモデルアーティファクトと対話し、[プロパティを更新](../../artifacts/update-an-artifact.md)（メタデータ、エイリアス、説明など）したい場合。

W&B Artifactsや高度なバージョン管理ユースケースの詳細については、[Artifacts](../../artifacts/intro.md)のドキュメントを参照してください。
:::

:::info
この[Colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb)を参照して、このページで説明されているAPIを使用するエンドツーエンドの例を確認してください。
:::

## モデルをW&Bのrunにログする
[`log_model`](../../../ref/python/run.md#log_model)を使用して、指定したディレクトリー内にコンテンツを含むモデルアーティファクトをログします。[`log_model`](../../../ref/python/run.md#log_model)メソッドは、結果として得られるモデルアーティファクトをW&Bのrunの出力としてもマークします。

モデルをW&Bのrunの入力または出力としてマークした場合、そのモデルの依存関係と関連付けを追跡できます。W&BのApp UIでモデルのリネージを表示します。詳細については、[Artifacts](../../artifacts/intro.md)チャプターの[アーティファクトグラフの探索とトラバース](../../artifacts/explore-and-traverse-an-artifact-graph.md)ページを参照してください。

モデルファイルが保存されているパスを`path`パラメータに提供します。パスはローカルファイル、ディレクトリー、または`s3://bucket/path`のような外部バケットへの[参照URI](../../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references)であることができます。

`<>`で囲まれた値は自身の値に置き換えてください。

```python
import wandb

# W&Bのrunを初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルをログする
run.log_model(path="<path-to-model>", name="<name>")
```

オプションで、モデルアーティファクトの名前を`name`パラメータに提供できます。`name`が指定されていない場合、W&Bは入力パスのベース名にrun IDを追加した名前を使用します。

:::tip
モデルに割り当てた、またはW&Bが割り当てた`name`を記憶してください。`use_model`メソッドを使用してモデルパスを取得するために名前が必要です。
:::

可能なパラメータの詳細については、APIリファレンスガイドの[`log_model`](../../../ref/python/run.md#log_model)を参照してください。

<details>

<summary>例：モデルをrunにログする</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&Bのrunを初期化
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

# モデルをトレーニング用にコンフィギュレーション
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# モデルを保存
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# モデルをW&Bのrunにログする
run.log_model(path=full_path, name="MNIST")
run.finish()
```

ユーザーが`log_model`を呼び出すと、`MNIST`という名前のモデルアーティファクトが作成され、`model.h5`ファイルがモデルアーティファクトに追加されました。ターミナルまたはノートブックには、モデルがログされたrunに関する情報が表示されます。

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
5 W&Bファイル、0メディアファイル、1アーティファクトファイル、0その他ファイルを同期しました
ログは以下で見つかります: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>


## ログされたモデルをダウンロードして使用
W&Bのrunに以前ログされていたモデルファイルにアクセスし、ダウンロードするために[`use_model`](../../../ref/python/run.md#use_model)関数を使用します。

取得したいモデルファイルが格納されているモデルアーティファクトの名前を指定します。提供する名前は、既存のログされたモデルアーティファクトの名前と一致する必要があります。

ファイルを`log_model`でログする際に`name`を定義しなかった場合、デフォルト名は入力パスのベース名にrun IDを追加されたものになります。

他の値も`<>`で囲まれた部分を自身の値に置き換えてください：

```python
import wandb

# runを初期化
run = wandb.init(project="<your-project>", entity="<your-entity>")

# モデルにアクセスしてダウンロード。ダウンロードされたアーティファクトへのパスを返します
downloaded_model_path = run.use_model(name="<your-model-name>")
```

[`use_model`](../../../ref/python/run.md#use_model)関数はダウンロードされたモデルファイルのパスを返します。このパスを追跡し、後でこのモデルをリンクする場合に備えてください。上記のコードスニペットでは、返されたパスが`downloaded_model_path`という変数に格納されています。

<details>

<summary>例：ログされたモデルをダウンロードして使用</summary>

例えば、以下のコードスニペットでは、ユーザーが`use_model` APIを呼び出しています。取得したいモデルアーティファクトの名前を指定し、バージョン/エイリアスも提供しています。APIから返されたパスが`downloaded_model_path`変数に格納されています。

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # モデルバージョンのセマンティックなニックネームまたは識別子
model_artifact_name = "fine-tuned-model"

# runを初期化
run = wandb.init(project=project, entity=entity)
# モデルにアクセスしてダウンロード。ダウンロードされたアーティファクトへのパスを返します
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

可能なパラメータと返り値の詳細については、APIリファレンスガイドの[`use_model`](../../../ref/python/run.md#use_model)を参照してください。

## モデルをW&B Model Registryにログしリンクする
[`link_model`](../../../ref/python/run.md#link_model)メソッドを使用して、モデルファイルをW&Bのrunにログし、それを[W&B Model Registry](../../model_registry/intro.md)にリンクします。登録済みのモデルが存在しない場合、`registered_model_name`パラメータに指定した名前で新しいモデルが自動的に作成されます。

:::tip
モデルをリンクすることは、モデルを集中管理されたチームのリポジトリーに「ブックマーク」または「公開」することに似ています。これにより、チームの他のメンバーがそのモデルを閲覧したり使用したりすることができます。

モデルをリンクすると、そのモデルは[Model Registry](../../model_registry/intro.md)に複製されません。また、そのモデルがプロジェクトからレジストリに移動することもありません。リンクされたモデルは、プロジェクト内の元のモデルへのポインターです。

[Model Registry](../../model_registry/intro.md)を使用して、タスクごとに最高のモデルを整理し、モデルのライフサイクルを管理し、MLライフサイクル全体での追跡と監査を簡素化し、Webhooksやジョブで[自動化](../../model_registry/automation.md)された下流のアクションを実行できます。
:::

*Registered Model*は、[Model Registry](../../model_registry/intro.md)にリンクされたモデルバージョンのコレクションまたはフォルダーです。登録済みのモデルは、通常、単一のモデリングユースケースやタスクの候補モデルを表します。

以下のコードスニペットは、[`link_model`](../../../ref/python/run.md#link_model) APIを使用してモデルをリンクする方法を示しています。他の値も`<>`で囲まれた部分を自身の値に置き換えてください：

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

オプションのパラメータについての詳細は、APIリファレンスガイドの[`link_model`](../../../ref/python/run.md#link_model)を参照してください。

もし`registered-model-name`がModel Registry内に既に存在する登録済みモデルの名前と一致する場合、モデルはその登録済みモデルにリンクされます。もしそのような登録済みモデルが存在しない場合、新しいものが作成され、モデルは最初にリンクされるものとして扱われます。

例えば、Model Registryに"Fine-Tuned-Review-Autocompletion"という名前の既存の登録済みモデルがあり、そのモデルに既にいくつかのモデルバージョン（v0, v1, v2）がリンクされているとします。`link_model`を`registered-model-name="Fine-Tuned-Review-Autocompletion"`で呼び出すと、新しいモデルはこの既存の登録済みモデルにv3としてリンクされます。この名前で登録済みモデルが存在しない場合、新しいものが作成され、新しいモデルはv0としてリンクされます。

<details>

<summary>例：モデルをW&B Model Registryにログしリンクする</summary>

例えば、以下のコードスニペットは、モデルファイルをログし、モデルを`"Fine-Tuned-Review-Autocompletion"`という登録済みモデルにリンクしています。

このために、ユーザーは`link_model` APIを呼び出します。APIを呼び出すとき、モデルのコンテンツを指すローカルのファイルパス（`path`）と登録済みモデルの名前（`registered_model_name`）を提供します。

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

:::info
リマインダー：登録済みモデルはブックマークされたモデルバージョンのコレクションを収容します。
:::

</details>