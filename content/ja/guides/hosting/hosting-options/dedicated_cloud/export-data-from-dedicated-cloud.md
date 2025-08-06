---
title: 専用クラウドからデータをエクスポートする
description: 専用クラウドからデータをエクスポートする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

専用クラウドインスタンスで管理しているすべてのデータをエクスポートしたい場合、W&B SDK API を使用して Runs、メトリクス、Artifacts などを [インポート／エクスポート API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) で抽出できます。下記の表は主なエクスポートのユースケースをまとめたものです。

| 目的 | ドキュメント |
|---------|---------------|
| プロジェクトのメタデータをエクスポート | [Projects API]({{< relref path="/ref/python/public-api/projects.md" lang="ja" >}}) |
| プロジェクト内の Runs をエクスポート | [Runs API]({{< relref path="/ref/python/public-api/runs.md" lang="ja" >}}) |
| Reports のエクスポート | [Report and Workspace API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ja" >}}) |
| Artifacts のエクスポート | [Explore artifact graphs]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}), [Download and use artifacts]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ja" >}}) |

[Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ja" >}}) で専用クラウド内に保管している Artifacts を管理している場合、W&B SDK API を使って Artifacts をエクスポートする必要がない場合があります。

{{% alert %}}
W&B SDK API を使ってすべてのデータをエクスポートする場合、Runs や Artifacts などの件数が多いとエクスポートに時間がかかることがあります。W&B では、エクスポートプロセスを適切なサイズのバッチに分けて実行し、専用クラウドインスタンスへの負荷を抑えることを推奨しています。
{{% /alert %}}