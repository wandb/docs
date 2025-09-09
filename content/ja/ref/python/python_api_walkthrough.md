---
title: API ウォークスルー
menu:
  reference:
    identifier: ja-ref-python-python_api_walkthrough
weight: 1
---

さまざまな W&B API の使い方と使いどころを学び、機械学習ワークフローでモデルの Artifacts をトラッキング・共有・管理しましょう。このページでは、実験のログ、Reports の作成、各タスクに適した W&B API を用いたログ済みデータへのアクセスを扱います。

W&B には次の API があります:

* W&B Python SDK（`wandb.sdk`）: トレーニング中の実験をログ・モニタリングします。
* W&B Public API（`wandb.apis.public`）: ログ済みの実験データをクエリ・分析します。
* W&B Report and Workspace API（`wandb.wandb-workspaces`）: 学び をまとめる Reports を作成します。

## サインアップして API キーを作成
マシンを W&B に認証するには、まず [wandb.ai/authorize](https://wandb.ai/authorize) で API キーを生成します。API キーをコピーし、安全に保管してください。

## パッケージのインストールとインポート

このウォークスルーに必要な W&B ライブラリとその他のパッケージをインストールします。  

```python
pip install wandb
```

W&B Python SDK をインポートします:


```python
import wandb
```

以下のコードブロックで Team の Entity を指定します:


```python
TEAM_ENTITY = "<Team_Entity>" # 自分の Team の Entity に置き換えてください
PROJECT = "my-awesome-project"
```

## モデルをトレーニング

以下のコードは基本的な機械学習ワークフロー（モデルのトレーニング、メトリクスのログ、モデルを Artifact として保存）をシミュレートします。

トレーニング中に W&B とやり取りするには W&B Python SDK（`wandb.sdk`）を使います。[`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}) で損失をログし、[`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) で学習済みモデルを Artifact として保存し、最後に [`Artifact.add_file`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_file" lang="ja" >}}) でモデルファイルを追加します。

```python
import random # データをシミュレートするため

def model(training_data: int) -> int:
    """Model simulation for demonstration purposes."""
    return training_data * 2 + random.randint(-1, 1)  

# 重みとノイズをシミュレート
weights = random.random() # ランダムな重みを初期化
noise = random.random() / 5  # ノイズをシミュレートするための小さなランダムノイズ

# ハイパーパラメーターと設定
config = {
    "epochs": 10,  # 学習するエポック数
    "learning_rate": 0.01,  # オプティマイザーの学習率
}

# コンテキストマネージャーを使って W&B の run を初期化・クローズ
with wandb.init(project=PROJECT, entity=TEAM_ENTITY, config=config) as run:    
    # トレーニングループをシミュレート
    for epoch in range(config["epochs"]):
        xb = weights + noise  # シミュレートした入力トレーニングデータ
        yb = weights + noise * 2  # シミュレートしたターゲット出力（入力ノイズの 2 倍）

        y_pred = model(xb)  # モデルの予測
        loss = (yb - y_pred) ** 2  # 二乗誤差（MSE）の損失

        print(f"epoch={epoch}, loss={y_pred}")
        # エポックと損失を W&B にログ
        run.log({
            "epoch": epoch,
            "loss": loss,
        })

    # モデル Artifact の一意な名前
    model_artifact_name = f"model-demo"  

    # シミュレートしたモデルファイルを保存するローカルパス
    PATH = "model.txt" 

    # モデルをローカルに保存
    with open(PATH, "w") as f:
        f.write(str(weights)) # モデルの重みをファイルに保存

    # Artifact オブジェクトを作成
    # ローカルに保存したモデルを Artifact に追加
    artifact = wandb.Artifact(name=model_artifact_name, type="model", description="My trained model")
    artifact.add_file(local_path=PATH)
    artifact.save()
```

上のコードブロックのポイント:
* トレーニング中のメトリクスをログするには `wandb.Run.log()` を使います。
* モデル（データセットなども）を W&B の Project に Artifact として保存するには `wandb.Artifact` を使います。

モデルをトレーニングして Artifact として保存できたら、W&B のレジストリに公開できます。[`wandb.Run.use_artifact()`]({{< relref path="/ref/python/sdk/classes/run/#method-runuse_artifact" lang="ja" >}}) を使って Project から Artifact を取得し、モデルレジストリで公開する準備をします。`wandb.Run.use_artifact()` は主に次の 2 つの目的に役立ちます:
* Project から Artifact オブジェクトを取得します。
* その Artifact を run の入力としてマークし、再現性とトレーサビリティを担保します。詳細は [リネージ マップの作成と表示]({{< relref path="/guides/core/registry/lineage/" lang="ja" >}}) を参照してください。

## モデルをモデルレジストリに公開

組織内の他の人とモデルを共有するには、`wandb.Run.link_artifact()` を使って [コレクション]({{< relref path="/guides/core/registry/create_collection" lang="ja" >}}) に公開します。次のコードは Artifact を [コア モデルレジストリ]({{< relref path="/guides/core/registry/registry_types/#core-registry" lang="ja" >}}) にリンクし、Team でアクセスできるようにします。

```python
# Artifact 名は、Team の Project 内の特定の Artifact バージョンを指定します
artifact_name = f'{TEAM_ENTITY}/{PROJECT}/{model_artifact_name}:v0'
print("Artifact name: ", artifact_name)

REGISTRY_NAME = "Model" # W&B のレジストリ名
COLLECTION_NAME = "DemoModels"  # レジストリ内のコレクション名

# レジストリ内の Artifact のターゲットパスを作成
target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
print("Target path: ", target_path)

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
model_artifact = run.use_artifact(artifact_or_name=artifact_name, type="model")
run.link_artifact(artifact=model_artifact, target_path=target_path)
run.finish()
```

`wandb.Run.link_artifact()` 実行後、モデル Artifact はレジストリ内の `DemoModels` コレクションに入ります。そこから、バージョン履歴、[リネージ マップ]({{< relref path="/guides/core/registry/lineage/" lang="ja" >}})、その他の [メタデータ]({{< relref path="/guides/core/registry/registry_cards/" lang="ja" >}}) などの詳細を確認できます。

レジストリに Artifact をリンクする方法の詳細は、[Artifacts をレジストリにリンクする]({{< relref path="/guides/core/registry/link_version/" lang="ja" >}}) を参照してください。

## 推論用にレジストリからモデル Artifact を取得

推論でモデルを使うには、`wandb.Run.use_artifact()` を使ってレジストリから公開済み Artifact を取得します。これは Artifact オブジェクトを返し、その後 [`wandb.Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactdownload" lang="ja" >}}) でローカルファイルにダウンロードできます。

```python
REGISTRY_NAME = "Model"  # W&B のレジストリ名
COLLECTION_NAME = "DemoModels"  # レジストリ内のコレクション名
VERSION = 0 # 取得する Artifact のバージョン

model_artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
print(f"Model artifact name: {model_artifact_name}")

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
registry_model = run.use_artifact(artifact_or_name=model_artifact_name)
local_model_path = registry_model.download()
```

レジストリから Artifact を取得する方法の詳細は、[レジストリから Artifact をダウンロードする]({{< relref path="/guides/core/registry/download_use_artifact/" lang="ja" >}}) を参照してください。

使用する機械学習フレームワークによっては、重みを読み込む前にモデルのアーキテクチャーを再構築する必要があります。これは利用するフレームワークやモデルに依存するため、ここでは読者の演習とします。

## 見つけたことを Report で共有

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}

作業を要約するための [Report]({{< relref path="/guides/core/reports/_index.md" lang="ja" >}}) を作成・共有します。プログラムから Report を作成するには、[W&B Report and Workspace API]({{< relref path="/ref/python/wandb_workspaces/reports.md" lang="ja" >}}) を使用します。

まず、W&B Reports API をインストールします:

```python
pip install wandb wandb-workspaces -qqq
```

次のコードブロックは、Markdown、パネル グリッドなど複数のブロックを含む Report を作成します。ブロックを追加したり、既存ブロックの内容を変更したりして Report をカスタマイズできます。

このコードブロックの出力は、作成された Report の URL へのリンクを表示します。ブラウザでこのリンクを開くと Report を閲覧できます。

```python
import wandb_workspaces.reports.v2 as wr

experiment_summary = """This is a summary of the experiment conducted to train a simple model using W&B."""
dataset_info = """The dataset used for training consists of synthetic data generated by a simple model."""
model_info = """The model is a simple linear regression model that predicts output based on input data with some noise."""

report = wr.Report(
    project=PROJECT,
    entity=TEAM_ENTITY,
    title="My Awesome Model Training Report",
    description=experiment_summary,
    blocks= [
        wr.TableOfContents(),
        wr.H2("Experiment Summary"),
        wr.MarkdownBlock(text=experiment_summary),
        wr.H2("Dataset Information"),
        wr.MarkdownBlock(text=dataset_info),
        wr.H2("Model Information"),
        wr.MarkdownBlock(text = model_info),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(title="Train Loss", x="Step", y=["loss"], title_x="Step", title_y="Loss")
                ],
            ),  
    ]

)

# Report を W&B に保存
report.save()
```

プログラムから Report を作成する方法、または W&B アプリでインタラクティブに Report を作成する方法の詳細は、W&B Docs Developer ガイドの [Create a report]({{< relref path="/guides/core/reports/create-a-report.md" lang="ja" >}}) を参照してください。

## レジストリをクエリ
W&B の [W&B Public APIs]({{< relref path="/ref/python/public-api/_index.md" lang="ja" >}}) を使って、履歴データのクエリ、分析、管理を行いましょう。これは Artifacts のリネージを追跡したり、異なるバージョンを比較したり、時間の経過に伴うモデルの性能を分析したりするのに役立ちます。

次のコードブロックは、特定のコレクションにあるすべての Artifact を検索するために モデルレジストリ をクエリする方法を示します。コレクションを取得し、そのバージョンを走査して、各 Artifact の名前とバージョンを出力します。

```python
import wandb

# wandb API を初期化
api = wandb.Api()

# 文字列 `model` を含み、かつ
# タグ `text-classification` または `latest` エイリアスを持つ Artifact バージョンをすべて検索
registry_filters = {
    "name": {"$regex": "model"}
}

# 論理演算子 $or を使って Artifact バージョンをフィルタ
version_filters = {
    "$or": [
        {"tag": "text-classification"},
        {"alias": "latest"}
    ]
}

# フィルタに一致するすべての Artifact バージョンの iterable を返す
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)

# 見つかった各 Artifact の name、collection、aliases、tags、created_at を出力
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```

レジストリのクエリ方法の詳細は、[MongoDB スタイルのクエリでレジストリアイテムを検索]({{< relref path="/guides/core/registry/search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ja" >}}) を参照してください。