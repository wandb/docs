---
title: API ガイド
weight: 1
---

さまざまな W&B API を使って機械学習ワークフロー内でモデルアーティファクトを追跡・共有・管理する方法と、そのタイミングについて解説します。このページでは、実験のログ、レポートの生成、ログデータへのアクセスなど、それぞれのタスクに適切な W&B API を利用する方法を説明します。

W&B では、以下の API が利用できます:

* W&B Python SDK (`wandb.sdk`): トレーニング中の実験のログと監視。
* W&B Public API (`wandb.apis.public`): ログされた実験データのクエリや分析。
* W&B Report and Workspace API (`wandb.wandb-workspaces`): 学びのまとめをレポートとして作成。

## サインアップと APIキー の作成
W&B でマシンを認証するには、まず [wandb.ai/authorize](https://wandb.ai/authorize) で APIキー を作成してください。生成した APIキー を安全に保管してください。

## パッケージのインストールとインポート

このウォークスルーに必要な W&B ライブラリとその他のパッケージをインストールします。  

```python
pip install wandb
```

W&B Python SDK のインポート:


```python
import wandb
```

以下のコードブロックでチームの entity を指定してください:


```python
TEAM_ENTITY = "<Team_Entity>" # チーム entity を入力してください
PROJECT = "my-awesome-project"
```

## モデルのトレーニング

次のコードは、基本的な機械学習ワークフロー（モデルのトレーニング、メトリクスのログ、モデルのアーティファクト保存）をシミュレートします。

トレーニング中に W&B Python SDK (`wandb.sdk`) を活用して W&B と連携します。損失は [`wandb.Run.log()`]({{< relref "/ref/python/sdk/classes/run/#method-runlog" >}}) でログし、トレーニング済みモデルは [`wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) でアーティファクトとして保存、その後 [`Artifact.add_file`]({{< relref "/ref/python/sdk/classes/artifact.md#add_file" >}}) でモデルファイルを追加します。

```python
import random # データのシミュレーション用

def model(training_data: int) -> int:
    """デモ用のモデルシミュレーション。"""
    return training_data * 2 + random.randint(-1, 1)  

# 重みとノイズのシミュレーション
weights = random.random() # ランダムな重みの初期化
noise = random.random() / 5  # 小さなノイズのシミュレーション

# ハイパーパラメーターと設定
config = {
    "epochs": 10,  # トレーニングするエポック数
    "learning_rate": 0.01,  # オプティマイザーの学習率
}

# コンテキストマネージャで W&B run を開始・終了
with wandb.init(project=PROJECT, entity=TEAM_ENTITY, config=config) as run:    
    # トレーニングループのシミュレーション
    for epoch in range(config["epochs"]):
        xb = weights + noise  # トレーニングデータ（入力）のシミュレーション
        yb = weights + noise * 2  # ターゲット出力のシミュレーション（ノイズを2倍）

        y_pred = model(xb)  # モデルによる予測
        loss = (yb - y_pred) ** 2  # 二乗誤差損失

        print(f"epoch={epoch}, loss={y_pred}")
        # エポックと損失を W&B にログ
        run.log({
            "epoch": epoch,
            "loss": loss,
        })

    # モデルアーティファクトのユニークな名前
    model_artifact_name = f"model-demo"  

    # シミュレートしたモデルファイルのローカルパス
    PATH = "model.txt" 

    # モデルをローカルに保存
    with open(PATH, "w") as f:
        f.write(str(weights)) # モデルの重みをファイル保存

    # アーティファクトオブジェクトの作成
    # ローカル保存したモデルをアーティファクトに追加
    artifact = wandb.Artifact(name=model_artifact_name, type="model", description="My trained model")
    artifact.add_file(local_path=PATH)
    artifact.save()
```

上記コードブロックのポイント:

* トレーニング中のメトリクスログには `wandb.Run.log()` を使用
* モデルやデータセット等をプロジェクトのアーティファクトとして保存するには `wandb.Artifact` を利用

これでモデルをトレーニングしアーティファクトとして保存できたので、それを W&B のレジストリに公開できます。プロジェクトからアーティファクトを取得し、Model registry に公開する準備には [`wandb.Run.use_artifact()`]({{< relref "/ref/python/sdk/classes/run/#method-runuse_artifact" >}}) を使います。`wandb.Run.use_artifact()` の主な役割は:

* プロジェクトからアーティファクトオブジェクトを取得
* そのアーティファクトを run の入力としてマークし、再現性とリネージを確保 詳細は [Create and view lineage map]({{< relref "/guides/core/registry/lineage/" >}}) を参照

## モデルを Model registry に公開

組織内の他のメンバーとモデルを共有したい場合、`wandb.Run.link_artifact()` で [collection]({{< relref "/guides/core/registry/create_collection" >}}) にモデルを公開します。次のコードは、アーティファクトを [core Model registry]({{< relref "/guides/core/registry/registry_types/#core-registry" >}}) にリンクし、チームでアクセス可能にします。

```python
# アーティファクト名はチームのプロジェクトでの特定バージョンを指定
artifact_name = f'{TEAM_ENTITY}/{PROJECT}/{model_artifact_name}:v0'
print("Artifact name: ", artifact_name)

REGISTRY_NAME = "Model" # W&B内のレジストリ名
COLLECTION_NAME = "DemoModels"  # レジストリ内コレクション名

# レジストリ内のアーティファクト格納先パスを作成
target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
print("Target path: ", target_path)

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
model_artifact = run.use_artifact(artifact_or_name=artifact_name, type="model")
run.link_artifact(artifact=model_artifact, target_path=target_path)
run.finish()
```

`wandb.Run.link_artifact()` を実行すると、モデルアーティファクトはレジストリ内の `DemoModels` コレクションに登録されます。ここから、バージョン履歴や [lineage map]({{< relref "/guides/core/registry/lineage/" >}})、その他の [metadata]({{< relref "/guides/core/registry/registry_cards/" >}}) などを確認できます。

アーティファクトをレジストリへリンクする手順の詳細は [Link artifacts to a registry]({{< relref "/guides/core/registry/link_version/" >}}) を参照してください。

## レジストリからモデルアーティファクトを取得して推論に利用

モデルを推論に活用するには、`wandb.Run.use_artifact()` で公開済みアーティファクトをレジストリから取得します。取得後、[`wandb.Artifact.download()`]({{< relref "/ref/python/sdk/classes/artifact/#method-artifactdownload" >}}) を用いてローカルファイルとしてダウンロード可能です。

```python
REGISTRY_NAME = "Model"  # W&B内のレジストリ名
COLLECTION_NAME = "DemoModels"  # レジストリ内コレクション名
VERSION = 0 # 取得したいアーティファクトのバージョン

model_artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
print(f"Model artifact name: {model_artifact_name}")

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
registry_model = run.use_artifact(artifact_or_name=model_artifact_name)
local_model_path = registry_model.download()
```

レジストリからアーティファクトを取得する手順の詳細は [Download an artifact from a registry]({{< relref "/guides/core/registry/download_use_artifact/" >}}) を参照してください。

利用する機械学習フレームワークによっては、重みロードの前にモデルのアーキテクチャーを再構築する必要がある場合があります。その部分は実際のフレームワークやモデルに依存するため、実装は読者の演習とします。 

## 学びをレポートで共有する

{{% alert %}}
W&B Report and Workspace API はパブリックプレビューです。
{{% /alert %}}

[report]({{< relref "/guides/core/reports/_index.md" >}}) を作成し、作業内容をまとめて共有しましょう。 レポートをプログラムから作成するには [W&B Report and Workspace API]({{< relref "/ref/python/wandb_workspaces/reports.md" >}}) を使用します。

最初に、W&B Reports API をインストールします:

```python
pip install wandb wandb-workspaces -qqq
```

次のコードでは Markdown やパネルグリッドなど、複数のブロックを含むレポートを作成します。ブロックの追加や内容の変更でレポートを自由にカスタマイズできます。

このコードの出力結果には作成されたレポートの URL リンクが表示されます。ブラウザで開いてレポートを閲覧してください。

```python
import wandb_workspaces.reports.v2 as wr

experiment_summary = """この実験では W&B を用いてシンプルなモデルのトレーニングを行いました。"""
dataset_info = """トレーニングに利用したデータセットは、シンプルなモデルで生成した合成データです。"""
model_info = """モデルは、入力データと多少のノイズに基づく出力を予測するシンプルな線形回帰モデルです。"""

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

# レポートを W&B に保存
report.save()
```

レポートをプログラムから自動生成する方法や、W&B App でインタラクティブに作成する方法の詳細は、W&B Docs Developer guide 内の [Create a report]({{< relref "/guides/core/reports/create-a-report.md" >}}) をご覧ください。

## レジストリのクエリ
[W&B Public APIs]({{< relref "/ref/python/public-api/_index.md" >}}) を活用すると、W&B 上の過去データのクエリ、分析、管理が可能です。アーティファクトのリネージ追跡、バージョン比較、モデルのパフォーマンス分析などに役立ちます。

次のコードは、Model registry で特定コレクションに含まれる全アーティファクトをクエリし、各バージョンを取得して情報を出力する例です。

```python
import wandb

# wandb API の初期化
api = wandb.Api()

# `model` 文字列を含み、`text-classification` タグか `latest` エイリアスを持つアーティファクトバージョンを検索
registry_filters = {
    "name": {"$regex": "model"}
}

# 論理 $or オペレーターを使ってアーティファクトのバージョンを絞り込み
version_filters = {
    "$or": [
        {"tag": "text-classification"},
        {"alias": "latest"}
    ]
}

# 条件に合致する全アーティファクトバージョンのイテラブルを返す
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)

# アーティファクト名やコレクション、エイリアス、タグ、作成日などを出力
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```

レジストリのクエリ方法については [Query registry items with MongoDB-style queries]({{< relref "/guides/core/registry/search_registry.md#query-registry-items-with-mongodb-style-queries" >}}) も参照してください。