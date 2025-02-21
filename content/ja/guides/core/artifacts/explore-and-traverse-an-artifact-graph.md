---
title: Explore artifact graphs
description: 自動的に作成される有向非巡回 W&B Artifact グラフをトラバース します。
menu:
  default:
    identifier: ja-guides-core-artifacts-explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B は、特定の run が記録した Artifacts と、特定の run が使用する Artifacts を自動的に追跡します。これらの Artifacts には、データセット、モデル、評価結果などが含まれます。Artifact のリネージを調査して、機械学習ライフサイクル全体で生成されるさまざまな Artifacts を追跡および管理できます。

## リネージ
Artifact のリネージを追跡することには、いくつかの重要な利点があります。

- 再現性: すべての Artifacts のリネージを追跡することで、チームは実験、モデル、および結果を再現できます。これは、デバッグ、実験、および機械学習モデルの検証に不可欠です。

- バージョン管理: Artifact のリネージには、Artifacts のバージョン管理と、経時的な変更の追跡が含まれます。これにより、チームは必要に応じて、以前のバージョンのデータまたはモデルにロールバックできます。

- 監査: Artifacts とその変換の詳細な履歴を持つことで、組織は規制およびガバナンスの要件を遵守できます。

- コラボレーションと知識の共有: Artifact のリネージは、試行の明確な記録と、何がうまくいき、何がうまくいかなかったかを提供することにより、チームメンバー間のより良いコラボレーションを促進します。これは、努力の重複を回避し、開発プロセスを加速するのに役立ちます。

### Artifact のリネージの検索
**Artifacts** タブで Artifact を選択すると、Artifact のリネージを確認できます。このグラフビューには、パイプラインの一般的な概要が表示されます。

Artifact グラフを表示するには:

1. W&B App UI で プロジェクト に移動します。
2. 左側の パネル で Artifact アイコンを選択します。
3. **リネージ** を選択します。

{{< img src="/images/artifacts/lineage1.gif" alt="リネージ タブへのアクセス" >}}

### リネージグラフのナビゲート

指定した Artifact または ジョブタイプ が名前の前に表示され、Artifacts は青いアイコンで、runs は緑のアイコンで表されます。矢印は、グラフ上の run または Artifact の入力と出力を詳細に示します。

{{< img src="/images/artifacts/lineage2.png" alt="Run と Artifact ノード" >}}

{{% alert %}}
Artifact の種類と名前は、左側のサイドバーと **リネージ** タブの両方で確認できます。
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="入力と出力" >}}

より詳細なビューを表示するには、個々の Artifact または run をクリックして、特定の オブジェクト に関する詳細情報を取得します。

{{< img src="/images/artifacts/lineage3a.gif" alt="Run のプレビュー" >}}

### Artifact クラスター

グラフのレベルに 5 つ以上の runs または Artifacts がある場合、 クラスター が作成されます。クラスター には、runs または Artifacts の特定の バージョン を検索するための検索バーがあり、クラスター から個々の ノード をプルして、 クラスター 内の ノード のリネージの調査を継続します。

ノード をクリックすると、ノード の概要を示すプレビューが開きます。矢印をクリックすると、個々の run または Artifact が抽出され、抽出された ノード のリネージを調べることができます。

{{< img src="/images/artifacts/lineage3b.gif" alt="Run クラスター の検索" >}}

## API を使用してリネージを追跡する
[W&B API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用してグラフをナビゲートすることもできます。

Artifact を作成します。まず、`wandb.init` で run を作成します。次に、`wandb.Artifact` で新しい Artifact を作成するか、既存の Artifact を取得します。次に、`.add_file` で Artifact にファイルを追加します。最後に、`.log_artifact` で Artifact を run に ログ します。完成した コード は次のようになります。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`、`.add_file`、`.add_dir`、および `.add_reference` を使用して、ファイルを Artifact に追加します。
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

Artifact オブジェクトの [`logged_by`]({{< relref path="/ref/python/artifact.md#logged_by" lang="ja" >}}) および [`used_by`]({{< relref path="/ref/python/artifact.md#used_by" lang="ja" >}}) メソッドを使用して、Artifact からグラフをたどります。

```python
# Artifact からグラフを上下にたどる:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
## 次のステップ
- [Artifacts の詳細を調べる]({{< relref path="/guides/core/artifacts/artifacts-walkthrough.md" lang="ja" >}})
- [Artifact のストレージを管理する]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})
- [Artifacts の プロジェクト を調べる](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)
