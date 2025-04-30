---
title: アーティファクトグラフの探索
description: W&B アーティファクト の自動生成された有向非巡回グラフを トラバース します。
menu:
  default:
    identifier: ja-guides-core-artifacts-explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B は、特定の run がログしたアーティファクトや、その run が利用したアーティファクトを自動で追跡します。これらのアーティファクトには、データセットやモデル、評価結果、その他が含まれることがあります。機械学習ライフサイクル全体で生成されたさまざまなアーティファクトを追跡および管理するために、アーティファクトのリネージを探索できます。

## リネージ
アーティファクトのリネージを追跡することには、いくつかの主要な利点があります：

- 再現性: すべてのアーティファクトのリネージを追跡することで、チームは実験やモデル、結果を再現でき、デバッグ、実験、および機械学習モデルの検証に不可欠です。

- バージョン管理: アーティファクトのリネージには、アーティファクトのバージョン管理とその変更の追跡が含まれます。必要に応じて、チームはデータやモデルの以前のバージョンに戻すことができます。

- 監査: アーティファクトとその変換の詳細な履歴を持つことで、組織は規制やガバナンスの要件に準拠できます。

- コラボレーションと知識共有: アーティファクトのリネージは、試行された記録が明確に示されており、何がうまくいって何がうまくいかなかったかを提供することで、チームメンバー間のより良いコラボレーションを促進します。これにより、努力の重複を避け、開発プロセスを加速させます。

### アーティファクトのリネージを見つける
**Artifacts** タブでアーティファクトを選択すると、アーティファクトのリネージを見ることができます。このグラフビューは、パイプラインの全体的な概要を示します。

アーティファクトグラフを見るには：

1. W&B App UI でプロジェクトに移動します。
2. 左のパネルでアーティファクトアイコンを選びます。
3. **Lineage** を選択します。

{{< img src="/images/artifacts/lineage1.gif" alt="Getting to the Lineage tab" >}}

### リネージグラフのナビゲート

指定したアーティファクトやジョブタイプは、その名前の前に表示され、アーティファクトは青のアイコン、Runs は緑のアイコンで表されます。矢印は、グラフ上での run またはアーティファクトの入力と出力を示します。

{{< img src="/images/artifacts/lineage2.png" alt="Run and artifact nodes" >}}

{{% alert %}}
アーティファクトのタイプと名前は、左のサイドバーと **Lineage** タブの両方で確認できます。
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="Inputs and outputs" >}}

より詳細なビューを得るために、個別のアーティファクトまたは run をクリックして、特定のオブジェクトに関する詳細情報を取得します。

{{< img src="/images/artifacts/lineage3a.gif" alt="Previewing a run" >}}

### アーティファクトクラスター

グラフのあるレベルに run またはアーティファクトが5つ以上ある場合、クラスターが作成されます。クラスターには、特定のバージョンの run またはアーティファクトを見つけるための検索バーがあり、クラスター内のノードを個別にプルしてそのリネージを調査することができます。

ノードをクリックすると、そのノードのプレビューが表示され、概要が示されます。矢印をクリックすると、個別の run またはアーティファクトが抽出され、抽出されたノードのリネージを調べることができます。

{{< img src="/images/artifacts/lineage3b.gif" alt="Searching a run cluster" >}}

## API を使用してリネージを追跡する
[W&B API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用してグラフをナビゲートすることもできます。

まず run を `wandb.init` で作成します。次に、`wandb.Artifact` で新しいアーティファクトを作成するか、既存のアーティファクトを取得します。次に、`.add_file` を使用してアーティファクトにファイルを追加します。最後に、`.log_artifact` でアーティファクトを run にログします。完成したコードは次のようになります：

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, `.add_reference` を使用して
    # アーティファクトにファイルやアセットを追加します
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

アーティファクトオブジェクトの [`logged_by`]({{< relref path="/ref/python/artifact.md#logged_by" lang="ja" >}}) と [`used_by`]({{< relref path="/ref/python/artifact.md#used_by" lang="ja" >}}) メソッドを使用して、アーティファクトからグラフをたどります：

```python
# アーティファクトからグラフを上下にたどります：
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
## 次のステップ
- [アーティファクトをさらに詳しく探索する]({{< relref path="/guides/core/artifacts/artifacts-walkthrough.md" lang="ja" >}})
- [アーティファクトストレージを管理する]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})
- [アーティファクトプロジェクトを探索する](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)