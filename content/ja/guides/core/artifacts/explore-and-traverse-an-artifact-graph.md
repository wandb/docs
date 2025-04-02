---
title: Explore artifact graphs
description: 自動的に作成された有向非巡回 W&B Artifact グラフをトラバースします。
menu:
  default:
    identifier: ja-guides-core-artifacts-explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B は、特定の run が ログに記録した Artifacts と、特定の run が使用する Artifacts を自動的に追跡します。これらの Artifacts には、データセット、モデル、評価結果などが含まれます。Artifacts のリネージを調べることで、機械学習のライフサイクル全体で生成されるさまざまな Artifacts を追跡および管理できます。

## リネージ
Artifacts のリネージを追跡することには、いくつかの重要な利点があります。

- 再現性: すべての Artifacts のリネージを追跡することで、チームは Experiments、モデル、および結果を再現できます。これは、デバッグ、実験、および機械学習モデルの検証に不可欠です。

- バージョン管理: Artifacts のリネージには、Artifacts のバージョン管理と、時間の経過に伴う変更の追跡が含まれます。これにより、チームは必要に応じてデータまたはモデルの以前のバージョンにロールバックできます。

- 監査: Artifacts とその変換の詳細な履歴を持つことで、組織は規制およびガバナンスの要件を遵守できます。

- コラボレーションと知識の共有: Artifacts のリネージは、試行錯誤の明確な記録を提供することにより、チームメンバー間のより良いコラボレーションを促進します。これは、努力の重複を回避し、開発プロセスを加速するのに役立ちます。

### Artifacts のリネージの検索
[**Artifacts**] タブで Artifacts を選択すると、Artifacts のリネージを確認できます。このグラフビューには、パイプラインの概要が表示されます。

Artifacts グラフを表示するには:

1. W&B App UI で プロジェクト に移動します。
2. 左側の パネル で Artifacts アイコンを選択します。
3. [**Lineage**] を選択します。

{{< img src="/images/artifacts/lineage1.gif" alt="Getting to the Lineage tab" >}}

### リネージグラフのナビゲート

指定した Artifacts またはジョブタイプが名前の前に表示され、Artifacts は青いアイコンで、runs は緑のアイコンで表示されます。矢印は、グラフ上の run または Artifacts の入出力を詳細に示します。

{{< img src="/images/artifacts/lineage2.png" alt="Run and artifact nodes" >}}

{{% alert %}}
Artifacts のタイプと名前は、左側のサイドバーと [**Lineage**] タブの両方で確認できます。
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="Inputs and outputs" >}}

より詳細なビューを表示するには、個々の Artifacts または run をクリックして、特定のオブジェクトに関する詳細情報を取得します。

{{< img src="/images/artifacts/lineage3a.gif" alt="Previewing a run" >}}

### Artifacts クラスター

グラフのレベルに 5 つ以上の runs または Artifacts がある場合、クラスターが作成されます。クラスターには、runs または Artifacts の特定の バージョン を検索するための検索バーがあり、クラスターから個々のノードをプルして、クラスター内のノードのリネージの調査を続行します。

ノードをクリックすると、ノードの概要を示すプレビューが開きます。矢印をクリックすると、個々の run または Artifacts が抽出され、抽出されたノードのリネージを調べることができます。

{{< img src="/images/artifacts/lineage3b.gif" alt="Searching a run cluster" >}}

## API を使用してリネージを追跡する
[W&B API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}})を使用してグラフをナビゲートすることもできます。

Artifacts を作成します。まず、`wandb.init` で run を作成します。次に、`wandb.Artifact` で新しい Artifacts を作成するか、既存の Artifacts を取得します。次に、`.add_file` で Artifacts にファイルを追加します。最後に、`.log_artifact` で Artifacts を run に ログ します。完成した コード は次のようになります。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # Add Files and Assets to the artifact using
    # `.add`, `.add_file`, `.add_dir`, and `.add_reference`
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

Artifacts オブジェクトの [`logged_by`]({{< relref path="/ref/python/artifact.md#logged_by" lang="ja" >}}) メソッドと [`used_by`]({{< relref path="/ref/python/artifact.md#used_by" lang="ja" >}}) メソッドを使用して、Artifacts からグラフをたどります。

```python
# Walk up and down the graph from an artifact:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
## 次のステップ
- [Artifacts をさらに詳しく調べる]({{< relref path="/guides/core/artifacts/artifacts-walkthrough.md" lang="ja" >}})
- [Artifacts ストレージを管理する]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})
- [Artifacts プロジェクト を調べる](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)
