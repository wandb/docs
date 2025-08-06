---
title: アーティファクトグラフを探索する
description: 自動的に作成された有向非巡回 W&B Artifact グラフをトラバースします。
menu:
  default:
    identifier: ja-guides-core-artifacts-explore-and-traverse-an-artifact-graph
    parent: artifacts
weight: 9
---

W&B は、run がログした Artifacts や run が使用した Artifacts を自動的にトラッキングします。これらの Artifacts には、データセット、モデル、評価結果などが含まれます。Artifacts のリネージを確認することで、機械学習ライフサイクルを通じて生成された様々な Artifacts を追跡・管理できます。

## リネージ
Artifacts のリネージを追跡することには、いくつか重要な利点があります。

- 再現性: すべての Artifacts のリネージを追跡することで、チームが実験、モデル、結果を再現できるようになります。これはデバッグや実験、機械学習モデルの検証に不可欠です。

- バージョン管理: Artifacts のリネージには Artifacts のバージョン管理と変更履歴の追跡が含まれます。これにより、必要に応じてデータやモデルの以前のバージョンにロールバックすることができます。

- 監査: Artifacts とその変換履歴の詳細な記録を保持することで、組織は規制やガバナンス要件への対応がしやすくなります。

- コラボレーションと知識共有: Artifacts のリネージは、チームメンバー間のコラボレーションを促進し、どんな試みがあったのか、何がうまくいき、何がうまくいかなかったかを明確に記録できます。これにより重複作業を防ぎ、開発プロセスが加速します。

### Artifact のリネージを見つける
**Artifacts** タブで任意の Artifact を選択すると、そのリネージを確認できます。このグラフビューでは、パイプライン全体の概要が表示されます。

Artifact グラフを表示するには:

1. W&B App UI で自身のプロジェクトに移動します。
2. 左側のパネルで Artifacts アイコンを選択します。
3. **Lineage** を選択します。

{{< img src="/images/artifacts/lineage1.gif" alt="リネージタブへの移動" >}}

### リネージグラフの操作

指定した Artifact またはジョブタイプは、その名前の前に表示されます。Artifacts は青いアイコン、Runs は緑色のアイコンで表されます。矢印は run や artifact の入力や出力をグラフ上で示しています。

{{< img src="/images/artifacts/lineage2.png" alt="Run と artifact のノード" >}}

{{% alert %}}
Artifact の種類と名前は、左サイドバーと **Lineage** タブの両方で確認できます。
{{% /alert %}}

{{< img src="/images/artifacts/lineage2a.gif" alt="入力と出力" >}}

より詳細に確認したい場合は、各 Artifact や run をクリックすると、対象オブジェクトの詳細情報が表示されます。

{{< img src="/images/artifacts/lineage3a.gif" alt="run のプレビュー" >}}

### Artifact クラスター

グラフの一段に run や artifact が5つ以上ある場合、それらはクラスターとしてまとめて表示されます。クラスターには検索バーがあり、特定のバージョンの run や artifact を探したり、クラスターから個別ノードを抜き出してそのノードのリネージを確認できます。

ノードをクリックすると、そのノードの概要をプレビューできます。矢印をクリックすると個別の run または artifact を展開でき、展開したノードのリネージをさらに確認することが可能です。

{{< img src="/images/artifacts/lineage3b.gif" alt="run クラスタの検索" >}}

## APIでリネージを追跡する
[W&B API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使ってグラフを操作することも可能です。

Artifact の作成方法を紹介します。まず、`wandb.init` で run を開始します。続いて、`wandb.Artifact` で新しい Artifact を作成するか既存のものを取得します。`.add_file` でファイルを Artifact に追加します。最後に、`.log_artifact` で Artifact を run に記録します。完成例は次のようになります。

```python
with wandb.init() as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")

    # ファイルとアセットを Artifact に追加
    # `.add`, `.add_file`, `.add_dir`, `.add_reference` を利用
    artifact.add_file("image1.png")
    run.log_artifact(artifact)
```

artifact オブジェクトの [`logged_by`]({{< relref path="/ref/python/sdk/classes/artifact.md#logged_by" lang="ja" >}}) や [`used_by`]({{< relref path="/ref/python/sdk/classes/artifact.md#used_by" lang="ja" >}}) メソッドを使えば、Artifact からグラフを上下にたどることができます。

```python
# Artifact からグラフを上り下りしてたどる例:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```
## 次のステップ
- [Artifacts をさらに詳しく探索する]({{< relref path="/guides/core/artifacts/artifacts-walkthrough.md" lang="ja" >}})
- [Artifact のストレージを管理する]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})
- [Artifacts プロジェクトを探索する](https://wandb.ai/wandb-smle/artifact_workflow/artifacts/raw_dataset/raw_data/v0/lineage)