---
title: Export data from Dedicated cloud
description: 専用クラウドからデータをエクスポートする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

あなたの専用クラウドインスタンスで管理されているすべてのデータをエクスポートしたい場合、[Import and Export API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使用して、ラン、メトリクス、アーティファクトなどを抽出できます。次の表は、いくつかの主要なエクスポートユースケースを示しています。

| 目的 | ドキュメント |
|-----|--------------|
| プロジェクトメタデータのエクスポート | [Projects API]({{< relref path="/ref/python/public-api/projects/" lang="ja" >}}) |
| プロジェクト内のランのエクスポート | [Runs API]({{< relref path="/ref/python/public-api/runs/" lang="ja" >}}) |
| レポートのエクスポート | [Reports API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ja" >}}) |
| アーティファクトのエクスポート | [Explore artifact graphs]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}), [Download and use artifacts]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ja" >}}) |

[Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ja" >}}) を使用して専用クラウドにストアされているアーティファクトを管理している場合、W&B SDK API を使用してアーティファクトをエクスポートする必要がないかもしれません。

{{% alert %}}
ラン、アーティファクトなどが多数ある場合、W&B SDK API を使用してすべてのデータをエクスポートするのは時間がかかることがあります。W&B は、専用クラウドインスタンスに負担をかけないように、適切なサイズのバッチでエクスポートプロセスを実行することを推奨します。
{{% /alert %}}