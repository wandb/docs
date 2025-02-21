---
title: Export data from Dedicated cloud
description: 専用クラウド からのデータのエクスポート
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

もし Dedicated cloud インスタンスで管理しているすべてのデータをエクスポートしたい場合は、W&B SDK API を使用して、runs、メトリクス、アーティファクト などを [Import and Export API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) で抽出できます。次の表は、主要なエクスポートのユースケースをいくつかまとめたものです。

| 目的 | ドキュメント |
|---------|---------------|
| プロジェクト のメタデータのエクスポート | [Projects API]({{< relref path="/ref/python/public-api/projects/" lang="ja" >}}) |
| プロジェクト 内の runs のエクスポート | [Runs API]({{< relref path="/ref/python/public-api/runs/" lang="ja" >}}) |
| Reports のエクスポート | [Reports API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ja" >}}) |
| アーティファクト のエクスポート | [アーティファクト グラフの探索]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}), [アーティファクト のダウンロードと使用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ja" >}}) |

[Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ja" >}}) を使用して Dedicated cloud に保存されている アーティファクト を管理している場合は、W&B SDK API を使用して アーティファクト をエクスポートする必要がない場合があります。

{{% alert %}}
W&B SDK API を使用してすべてのデータをエクスポートすると、多数の runs や アーティファクト がある場合に時間がかかることがあります。W&B では、Dedicated cloud インスタンスに負荷をかけすぎないように、適切なサイズのバッチでエクスポート プロセス を実行することをお勧めします。
{{% /alert %}}
