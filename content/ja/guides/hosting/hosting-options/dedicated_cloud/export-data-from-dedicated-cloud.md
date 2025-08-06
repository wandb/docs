---
title: 専用クラウドからデータをエクスポートする
description: 専用クラウドからデータをエクスポートする
menu:
  default:
    identifier: export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

専用クラウドインスタンスで管理されているすべてのデータをエクスポートしたい場合、W&B SDK API を使って Runs、メトリクス、Artifacts などを [Import and Export API]({{< relref "/ref/python/public-api/index.md" >}}) で抽出できます。下記の表は主なエクスポート用途の例をまとめたものです。

| 目的 | ドキュメント |
|------|--------------|
| プロジェクトのメタデータをエクスポート | [Projects API]({{< relref "/ref/python/public-api/projects.md" >}}) |
| プロジェクト内の Runs をエクスポート | [Runs API]({{< relref "/ref/python/public-api/runs.md" >}}) |
| Reports をエクスポート | [Report and Workspace API]({{< relref "/guides/core/reports/clone-and-export-reports/" >}}) |
| Artifacts をエクスポート | [Explore artifact graphs]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph" >}}), [Download and use artifacts]({{< relref "/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" >}}) |

[Secure Storage Connector]({{< relref "/guides/models/app/settings-page/teams/#secure-storage-connector" >}}) を使って専用クラウドに保存された Artifacts を管理している場合、W&B SDK API を使ったエクスポートは不要な場合があります。

{{% alert %}}
大量の Runs や Artifacts などが存在する場合、W&B SDK API を使ってすべてのデータをエクスポートする処理には時間がかかることがあります。W&B では、専用クラウドインスタンスに過度な負荷がかからないよう、適切なサイズのバッチでエクスポートプロセスを実行することを推奨します。
{{% /alert %}}