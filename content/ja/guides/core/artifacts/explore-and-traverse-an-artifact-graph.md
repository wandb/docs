---
title: アーティファクトグラフを探索する
description: 自動生成された有向非巡回 W&B アーティファクト グラフをトラバースします。
menu:
  default:
    identifier: explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B は、各 run がログした Artifacts や、使用した Artifacts を自動的にトラッキングします。これらの Artifacts には、データセット、モデル、評価結果などが含まれます。Artifacts のリネージを探索することで、機械学習ライフサイクル全体で生成されたさまざまな Artifacts の管理や追跡が可能です。

## リネージ
Artifacts のリネージを追跡することで、いくつかの主要なメリットがあります。

- 再現性: すべての Artifacts のリネージをトラッキングすることで、Teams は Experiments、Models、結果を再現できるようになり、デバッグや検証、機械学習モデルの確認に不可欠です。

- バージョン管理: Artifact のリネージは、Artifacts のバージョン管理や、その変更履歴を追跡することを含みます。これにより、必要に応じてデータや Models の過去バージョンにロールバックすることができます。

- 監査: Artifacts およびその変換履歴の詳細な情報が残るため、組織はコンプライアンスやガバナンス要件も満たしやすくなります。

- コラボレーションと知識共有: Artifact のリネージは、試行の記録や成功例・失敗例を明確にチームで共有できるので、チームメンバー間のより良いコラボレーションを促進します。これにより、重複作業を防ぎ、プロセスを加速します。

### Artifacts のリネージを確認する
**Artifacts** タブで対象のアーティファクトを選択すると、そのアーティファクトのリネージが確認できます。このグラフビューは、ワークフロー全体の概要を示します。

Artifacts グラフを表示するには:

1. W&B App UI で対象の Project に移動します
2. 左側のパネルからアーティファクトアイコンを選択します。
3. **Lineage** を選択します。

{{< img src="/images/artifacts/lineage1.gif" alt="Getting to the Lineage tab" >}}

### リネージグラフの操作方法

提供されたアーティファクトやジョブタイプが名前の前に表示され、アーティファクトは青色のアイコン、run は緑色のアイコンで表現されます。矢印は、run やアーティファクトの入力・出力関係をグラフ上で示します。

{{< img src="/images/artifacts/lineage2.png" alt="Run and artifact nodes" >}}

{{% alert %}}
アーティファクトの種類や名前は、左サイドバーおよび **Lineage** タブの両方で確認できます。
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="Inputs and outputs" >}}

より詳細な情報が必要な場合は、個々のアーティファクトまたは run をクリックすると、そのオブジェクトの詳細が表示されます。

{{< img src="/images/artifacts/lineage3a.gif" alt="Previewing a run" >}}

### アーティファクトクラスター

グラフのある階層に run またはアーティファクトが5つ以上存在する場合、クラスターが生成されます。クラスターには検索バーが表示され、特定の run やアーティファクトのバージョンを探せます。また、クラスター内のノードを個別に引き出し、そのノードのリネージをさらに調査することもできます。

ノードをクリックするとノードの概要が表示されます。矢印をクリックすると、個別の run またはアーティファクトが取り出され、抽出したノードのリネージをじっくり確認できます。

{{< img src="/images/artifacts/lineage3b.gif" alt="Searching a run cluster" >}}

## API でリネージを追跡する
[W&B API]({{< relref "/ref/python/public-api/api.md" >}}) を使ってグラフを操作することもできます。

アーティファクトを作成します。まず、`wandb.init` で run を作成します。次に、`wandb.Artifact` で新しいアーティファクトを作成するか、既存のアーティファクトを取得します。そして、`.add_file` でアーティファクトにファイルを追加します。最後に `.log_artifact` で run にアーティファクトをログします。完成したコード例は以下の通りです。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # `.add`, `.add_file`, `.add_dir`, `.add_reference` を使って
    # ファイルやデータをアーティファクトに追加
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

artifact オブジェクトの [`logged_by`]({{< relref "/ref/python/sdk/classes/artifact.md#logged_by" >}}) や [`used_by`]({{< relref "/ref/python/sdk/classes/artifact.md#used_by" >}}) メソッドを使うことで、Artifacts からグラフをたどることができます。

```python
# アーティファクトからグラフを上下に歩く例:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```

## 次のステップ
- [Artifacts をさらに詳しく探索する]({{< relref "/guides/core/artifacts/artifacts-walkthrough.md" >}})
- [Artifacts のストレージを管理する]({{< relref "/guides/core/artifacts/manage-data/delete-artifacts.md" >}})
- [Artifacts Project を体験してみる](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)