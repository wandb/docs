---
title: API ウォークスルー
menu:
  reference:
    identifier: ja-ref-python-python_api_walkthrough
weight: 1
---

さまざまな W&B API を使い分けて、機械学習ワークフローにおけるモデルアーティファクトの追跡・共有・管理を行うタイミングと方法について学びましょう。このページでは、実験のログ取得、レポートの生成、およびタスクごとに適切な W&B API を用いて記録したデータへアクセスする流れを解説します。

W&B では以下の API が利用できます：

* W&B Python SDK (`wandb.sdk`): トレーニング中の実験のログ取得や監視。
* W&B Public API (`wandb.apis.public`): 記録した実験データのクエリや分析。
* W&B Report and Workspace API (`wandb.wandb-workspaces`): 発見内容をまとめるレポートの作成。

## サインアップと API キーの作成
W&B でマシンを認証するには、まず [wandb.ai/authorize](https://wandb.ai/authorize) から APIキー を生成する必要があります。 APIキー をコピーして安全な場所に保管してください。

## パッケージのインストールとインポート

このチュートリアルに必要な W&B ライブラリやその他のパッケージをインストールします。  

```python
pip install wandb
```

W&B Python SDK をインポートします：

```python
import wandb
```

次のコードブロックでチームの Entity を指定します：

```python
TEAM_ENTITY = "<Team_Entity>" # ここにご自身のチーム Entity を入力してください
PROJECT = "my-awesome-project"
```

## モデルのトレーニング

次のコードは、基本的な機械学習ワークフロー（モデルのトレーニング、メトリクスの記録、モデルをアーティファクトとして保存）をシミュレートします。

トレーニング中の W&B との連携には W&B Python SDK (`wandb.sdk`) を使います。[`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}) でロスを記録し、学習済みモデルは [`wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) でアーティファクトとして保存後、 [`Artifact.add_file`]({{< relref path="/ref/python/sdk/classes/artifact.md#add_file" lang="ja" >}}) でファイルを追加します。

```python
import random # データをシミュレートするため

def model(training_data: int) -> int:
    """デモのためのモデルシミュレーション。"""
    return training_data * 2 + random.randint(-1, 1)  

# 重みとノイズをシミュレート
weights = random.random() # 重みをランダムに初期化
noise = random.random() / 5  # 軽微なノイズを追加

# ハイパーパラメーターや設定
config = {
    "epochs": 10,  # トレーニングのエポック数
    "learning_rate": 0.01,  # オプティマイザーの学習率
}

# コンテキストマネージャで W&B run を初期化・終了
with wandb.init(project=PROJECT, entity=TEAM_ENTITY, config=config) as run:    
    # トレーニングループのシミュレート
    for epoch in range(config["epochs"]):
        xb = weights + noise  # シミュレートされた入力トレーニングデータ
        yb = weights + noise * 2  # シミュレートされた目標出力（入力ノイズの2倍）

        y_pred = model(xb)  # モデルの予測値
        loss = (yb - y_pred) ** 2  # 平均二乗誤差ロス

        print(f"epoch={epoch}, loss={y_pred}")
        # エポックとロスを W&B に記録
        run.log({
            "epoch": epoch,
            "loss": loss,
        })

    # モデルアーティファクトの一意な名前
    model_artifact_name = f"model-demo"  

    # シミュレートしたモデルファイルの保存先
    PATH = "model.txt" 

    # モデルをローカルに保存
    with open(PATH, "w") as f:
        f.write(str(weights)) # 重み情報をファイルに保存

    # アーティファクトオブジェクトの作成
    # 保存したモデルファイルをアーティファクトに追加
    artifact = wandb.Artifact(name=model_artifact_name, type="model", description="My trained model")
    artifact.add_file(local_path=PATH)
    artifact.save()
```

上記コードブロックのポイントは以下の通りです：
* トレーニング中に `wandb.Run.log()` でメトリクスを記録します。
* `wandb.Artifact` を用いてモデル（やデータセットなど）を W&B プロジェクトにアーティファクトとして保存します。

モデルをトレーニングしアーティファクトとして保存したら、W&B のレジストリに公開できます。 [`wandb.Run.use_artifact()`]({{< relref path="/ref/python/sdk/classes/run/#method-runuse_artifact" lang="ja" >}}) を使いプロジェクトからアーティファクトを取得し、モデルレジストリ公開用に準備しましょう。`wandb.Run.use_artifact()` の主な役割は次の2つです：
* プロジェクトからアーティファクトオブジェクトを取得します。
* そのアーティファクトを run の入力としてマークし、再現性や追跡性を確保します。 詳しくは [リネージマップの作成と閲覧]({{< relref path="/guides/core/registry/lineage/" lang="ja" >}}) を参照ください。

## モデルを Model registry に公開

組織内でモデルを共有したい場合は、 [コレクション]({{< relref path="/guides/core/registry/create_collection" lang="ja" >}}) に公開します。`wandb.Run.link_artifact()` を使い、次のコードで [core Model registry]({{< relref path="/guides/core/registry/registry_types/#core-registry" lang="ja" >}}) にアーティファクトをリンクできます。これによりチームメンバーが model を利用できるようになります。

```python
# アーティファクト名は、チームプロジェクト内での特定のアーティファクトバージョンを指定
artifact_name = f'{TEAM_ENTITY}/{PROJECT}/{model_artifact_name}:v0'
print("Artifact name: ", artifact_name)

REGISTRY_NAME = "Model" # W&B 内でのレジストリ名
COLLECTION_NAME = "DemoModels"  # レジストリ内のコレクション名

# レジストリ内でのアーティファクトのターゲットパスを作成
target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
print("Target path: ", target_path)

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
model_artifact = run.use_artifact(artifact_or_name=artifact_name, type="model")
run.link_artifact(artifact=model_artifact, target_path=target_path)
run.finish()
```

`wandb.Run.link_artifact()` を実行すると、model アーティファクトはレジストリの `DemoModels` コレクションに保存されます。ここからバージョン履歴や [リネージマップ]({{< relref path="/guides/core/registry/lineage/" lang="ja" >}})、他の [メタデータ]({{< relref path="/guides/core/registry/registry_cards/" lang="ja" >}}) を閲覧可能です。

アーティファクトをレジストリにリンクする手順の詳細は [Link artifacts to a registry]({{< relref path="/guides/core/registry/link_version/" lang="ja" >}}) を参照してください。

## 推論用にレジストリからモデルアーティファクトを取得

推論でモデルを利用する場合、`wandb.Run.use_artifact()` を使いレジストリから公開済みアーティファクトを取得します。これによりアーティファクトオブジェクトが返され、 [`wandb.Artifact.download()`]({{< relref path="/ref/python/sdk/classes/artifact/#method-artifactdownload" lang="ja" >}}) でローカルにダウンロードできます。

```python
REGISTRY_NAME = "Model"  # W&B 内のレジストリ名
COLLECTION_NAME = "DemoModels"  # レジストリ内のコレクション名
VERSION = 0 # 取得したいアーティファクトのバージョン

model_artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{VERSION}"
print(f"Model artifact name: {model_artifact_name}")

run = wandb.init(entity=TEAM_ENTITY, project=PROJECT)
registry_model = run.use_artifact(artifact_or_name=model_artifact_name)
local_model_path = registry_model.download()
```

レジストリからアーティファクトを取得する方法の詳細は [Download an artifact from a registry]({{< relref path="/guides/core/registry/download_use_artifact/" lang="ja" >}}) を参照してください。

利用する機械学習フレームワークによっては、重みをロードする前にモデルのアーキテクチャーを再現する必要があります。これはフレームワークやモデルごとに異なるため、各自で対応してください。

## 発見内容をレポートで共有

{{% alert %}}
W&B Report and Workspace API は Public Preview です。
{{% /alert %}}

[レポート]({{< relref path="/guides/core/reports/_index.md" lang="ja" >}}) を作成して、自分の作業を整理・共有しましょう。プログラムからレポートを作成するには、 [W&B Report and Workspace API]({{< relref path="/ref/python/wandb_workspaces/reports.md" lang="ja" >}}) を利用できます。

まず、W&B Reports API をインストールします：

```python
pip install wandb wandb-workspaces -qqq
```

下記のコードブロックでは、Markdown やパネルグリッドなど複数のブロックを含むレポートを作成しています。ブロックを追加したり内容を変更することで、レポートをカスタマイズ可能です。

実行結果としてできあがったレポートの URL が表示されます。ブラウザで開いてレポートを確認できます。

```python
import wandb_workspaces.reports.v2 as wr

experiment_summary = """W&B を使ってシンプルなモデルをトレーニングした実験の要約です。"""
dataset_info = """トレーニングに使用したデータセットはシンプルなモデルによって生成された合成データです。"""
model_info = """モデルは、入力データにノイズが加えられた値を予測するシンプルな線形回帰モデルです。"""

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

プログラムからのレポート作成方法や、W&B App でインタラクティブにレポートを作る方法については、W&B Docs の [Create a report]({{< relref path="/guides/core/reports/create-a-report.md" lang="ja" >}}) をご覧ください。

## レジストリへのクエリ
[W&B Public APIs]({{< relref path="/ref/python/public-api/_index.md" lang="ja" >}}) を利用して、W&B の記録データの履歴をクエリ・分析・管理可能です。これらはアーティファクトのリネージ追跡、異なるバージョンの比較、モデル性能の時系列分析などに役立ちます。

以下のコードブロックでは、Model registry 内の特定コレクションに含まれる全アーティファクトをクエリする方法を示しています。コレクションを取得し、各バージョンを繰り返し処理しながらアーティファクト名・バージョンを出力します。

```python
import wandb

# wandb API を初期化
api = wandb.Api()

# `model` を含み、`text-classification` タグまたは `latest` エイリアスを持つ
# 全アーティファクトバージョンを検索するフィルター
registry_filters = {
    "name": {"$regex": "model"}
}

# 論理 $or 演算子でバージョンをフィルタリング
version_filters = {
    "$or": [
        {"tag": "text-classification"},
        {"alias": "latest"}
    ]
}

# 条件に一致する全てのアーティファクトバージョンを取得
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)

# 見つかった各アーティファクトの名前、コレクション、エイリアス、タグ、作成日を出力
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")
```

レジストリのクエリ方法については、[Query registry items with MongoDB-style queries]({{< relref path="/guides/core/registry/search_registry.md#query-registry-items-with-mongodb-style-queries" lang="ja" >}}) も参考にしてください。