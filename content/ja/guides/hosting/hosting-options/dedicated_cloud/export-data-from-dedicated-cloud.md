---
title: 専用クラウドからデータをエクスポートする
description: 専用クラウドからデータをエクスポートする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: /ja/guides/hosting/export-data-from-dedicated-cloud
---

もし専用クラウドインスタンスで管理されているすべてのデータをエクスポートしたい場合、W&B SDK APIを使用して、[インポートおよびエクスポートAPI]({{< relref path="/ref/python/public-api/" lang="ja" >}})を利用して runs、メトリクス、Artifacts などを抽出できます。以下の表は、いくつかの主要なエクスポートユースケースについて説明しています。

| 目的 | ドキュメント |
|---------|---------------|
| プロジェクトのメタデータをエクスポート | [Projects API]({{< relref path="/ref/python/public-api/projects/" lang="ja" >}}) |
| プロジェクト内の runs をエクスポート | [Runs API]({{< relref path="/ref/python/public-api/runs/" lang="ja" >}}) |
| レポートをエクスポート | [Reports API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ja" >}}) |
| Artifacts をエクスポート | [Explore artifact graphs]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}), [Download and use artifacts]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ja" >}}) |

専用クラウドで保管されている Artifacts を [Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ja" >}}) で管理している場合、W&B SDK APIを使用してアーティファクトをエクスポートする必要はないかもしれません。

{{% alert %}}
大量の runs や Artifacts などがある場合、W&B SDK APIを使用してすべてのデータをエクスポートするのは遅くなる可能性があります。W&Bは、専用クラウドインスタンスを圧迫しないように、適切なサイズのバッチでエクスポートプロセスを実行することを推奨しています。
{{% /alert %}}