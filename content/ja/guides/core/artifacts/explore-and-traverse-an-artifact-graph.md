---
title: Artifacts グラフを探索する
description: 自動生成された有向非巡回の W&B Artifacts グラフをトラバースします。
menu:
  default:
    identifier: ja-guides-core-artifacts-explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B は、特定の run がログに記録した artifacts と、特定の run が使用する artifacts を自動的に追跡します。これらの Artifacts には、Datasets、Models、評価結果などが含まれます。Artifacts のリネージを探索することで、機械学習ライフサイクル全体で生成された様々な artifacts を追跡し、管理できます。

## リネージ
Artifacts のリネージを追跡することには、いくつかの主要な利点があります。

- 再現性: すべての artifacts のリネージを追跡することで、チームは実験、Models、および結果を再現できます。これは、機械学習モデルのデバッグ、実験、および検証に不可欠です。

- バージョン管理: artifacts のリネージには、artifacts のバージョン管理と経時的な変更の追跡が含まれます。これにより、チームは必要に応じてデータや Models の以前のバージョンにロールバックできます。

- 監査: Artifacts とそれらの変換の詳細な履歴を持つことで、組織は規制およびガバナンス要件に準拠できます。

- コラボレーションと知識共有: Artifacts のリネージは、試行の明確な記録、何がうまくいったか、何がうまくいかなかったかを提供することで、チームメンバー間のより良いコラボレーションを促進します。これにより、作業の重複を避け、開発プロセスを加速するのに役立ちます。

### Artifacts のリネージを見つける
**Artifacts** タブで artifact を選択すると、その artifact のリネージが表示されます。このグラフビューは、パイプラインの全体像を示します。

Artifacts グラフを表示するには：

1. W&B App UI でご自身の Projects に移動します。
2. 左側のパネルにある Artifacts アイコンを選択します。
3. **Lineage** を選択します。

{{< img src="/images/artifacts/lineage1.gif" alt="Lineage タブへの移動" >}}

### リネージグラフの操作

提供された artifact またはジョブタイプは、その名前の前に表示され、artifacts は青いアイコンで、run は緑のアイコンで表されます。矢印は、グラフ上の run または artifact の入力と出力を詳細に示します。

{{< img src="/images/artifacts/lineage2.png" alt="run と artifact のノード" >}}

{{% alert %}}
左側のサイドバーと **Lineage** タブの両方で、artifact のタイプと名前を表示できます。
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="入力と出力" >}}

より詳細な表示については、個々の artifact または run をクリックして、特定のオブジェクトに関する詳細情報を取得します。

{{< img src="/images/artifacts/lineage3a.gif" alt="run のプレビュー" >}}

### Artifacts クラスター

グラフのレベルに 5 つ以上の run または artifacts がある場合、クラスターが作成されます。クラスターには、run または artifact の特定のバージョンを見つけるための検索バーがあり、クラスターから個々のノードを引き出して、クラスター内のノードのリネージを調査し続けることができます。

ノードをクリックすると、ノードの概要を示すプレビューが開きます。矢印をクリックすると、個々の run または artifact が抽出され、抽出されたノードのリネージを調べることができます。

{{< img src="/images/artifacts/lineage3b.gif" alt="run クラスターの検索" >}}

## API を使用してリネージを追跡する
[W&B API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用してグラフを操作することもできます。

artifact を作成します。まず、`wandb.init` で run を作成します。次に、`wandb.Artifact` で新しい artifact を作成するか、既存の artifact を取得します。次に、`.add_file` で artifact にファイルを追加します。最後に、`.log_artifact` で artifact を run にログします。完成したコードは次のようになります。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, および `.add_reference` を使用して、ファイルとアセットをアーティファクトに追加します。
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

Artifact オブジェクトの [`logged_by`]({{< relref path="/ref/python/sdk/classes/artifact.md#logged_by" lang="ja" >}}) メソッドと [`used_by`]({{< relref path="/ref/python/sdk/classes/artifact.md#used_by" lang="ja" >}}) メソッドを使用して、artifact からグラフをたどります。

```python
# artifact からグラフを上流および下流にたどります：
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
## 次のステップ
- [Artifacts をさらに詳しく探索する]({{< relref path="/guides/core/artifacts/artifacts-walkthrough.md" lang="ja" >}})
- [Artifacts ストレージを管理する]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})
- [Artifacts プロジェクトを探索する](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)