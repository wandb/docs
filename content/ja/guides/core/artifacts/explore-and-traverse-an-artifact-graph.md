---
title: Explore artifact graphs
description: W&B アーティファクトの自動作成された有向非巡回グラフをトラバースします。
menu:
  default:
    identifier: ja-guides-core-artifacts-explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B は特定の Run に記録されたアーティファクトと、特定の Run が使用したアーティファクトを自動的に追跡します。これらのアーティファクトは、データセット、モデル、評価結果、その他を含むことがあります。アーティファクトのリネージを探索して、機械学習ライフサイクル全体で生成されたさまざまなアーティファクトを追跡・管理できます。

## リネージ
アーティファクトのリネージを追跡することには、いくつかの重要な利点があります：

- 再現性: すべてのアーティファクトのリネージを追跡することにより、チームは実験、モデル、結果を再現することができ、これはデバッグ、実験、および機械学習モデルの検証にとって不可欠です。

- バージョン管理: アーティファクトのリネージには、アーティファクトのバージョン管理とその変更の追跡が含まれます。これにより、必要に応じて以前のバージョンのデータやモデルに戻ることができます。

- 監査: アーティファクトとその変換の詳細な履歴を持つことにより、組織は規制およびガバナンス要件に準拠することができます。

- コラボレーションと知識共有: アーティファクトのリネージは、試みや成功しなかった点を含む明確な記録を提供することで、チームメンバー間のコラボレーションを促進します。これにより、努力の重複を避け、開発プロセスを加速することができます。

### アーティファクトのリネージを見つける
**Artifacts** タブでアーティファクトを選択すると、アーティファクトのリネージを見ることができます。このグラフビューは、パイプラインの一般的な概要を表示します。

アーティファクトのグラフを表示するには：

1. W&B アプリの UI でプロジェクトに移動します
2. 左のパネルでアーティファクトアイコンを選択します。
3. **Lineage** を選択します。

{{< img src="/images/artifacts/lineage1.gif" alt="リネージタブへの移動" >}}

### リネージグラフのナビゲーション

提供されたアーティファクトまたはジョブタイプは、その名前の前に表示され、アーティファクトは青いアイコンで、Runs は緑のアイコンで表されます。矢印はグラフ上の Run またはアーティファクトの入力と出力を詳細に示します。

{{< img src="/images/artifacts/lineage2.png" alt="Run およびアーティファクトノード" >}}

{{% alert %}}
アーティファクトのタイプと名前は、左のサイドバーと **Lineage** タブの両方で表示できます。
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="入力と出力" >}}

より詳細なビューを表示するには、個々のアーティファクトまたは Run をクリックして、特定のオブジェクトに関する詳細情報を取得します。

{{< img src="/images/artifacts/lineage3a.gif" alt="Run のプレビュー" >}}

### アーティファクトクラスター

グラフのレベルに 5 つ以上の Run またはアーティファクトがあると、クラスターが作成されます。クラスターには、特定のバージョンの Run またはアーティファクトを検索するための検索バーがあり、クラスターから個々のノードを引き出して、クラスター内のノードのリネージを調査し続けます。

ノードをクリックすると、そのノードの概要のプレビューが表示されます。矢印をクリックすると、個々の Run またはアーティファクトを抽出し、抽出したノードのリネージを調べることができます。

{{< img src="/images/artifacts/lineage3b.gif" alt="Run クラスターの検索" >}}

## API を使用してリネージを追跡する
[W&B API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用してグラフをナビゲートすることもできます。

アーティファクトを作成します。まず、`wandb.init` を使用して Run を作成します。次に、`wandb.Artifact` を使用して新しいアーティファクトを作成するか、既存のものを取得します。次に、`.add_file` を使用してファイルをアーティファクトに追加します。最後に、`.log_artifact` を使用してアーティファクトを Run にログします。完成したコードは次のようになります：

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # アーティファクトにファイルとアセットを追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` を使用
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

アーティファクトオブジェクトの [`logged_by`]({{< relref path="/ref/python/artifact.md#logged_by" lang="ja" >}}) と [`used_by`]({{< relref path="/ref/python/artifact.md#used_by" lang="ja" >}}) メソッドを使用して、アーティファクトからグラフを移動します：

```python
# アーティファクトからグラフを上下に移動
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
## 次のステップ
- [アーティファクトをさらに詳しく探る]({{< relref path="/guides/core/artifacts/artifacts-walkthrough.md" lang="ja" >}})
- [アーティファクトストレージを管理する]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})
- [Artifacts プロジェクトを探索する](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)