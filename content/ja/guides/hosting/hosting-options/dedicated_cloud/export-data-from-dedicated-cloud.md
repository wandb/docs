---
title: Dedicated Cloud からデータをエクスポート
description: Dedicated Cloud からデータをエクスポートする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

Dedicated Cloud インスタンスで管理されているすべてのデータをエクスポートしたい場合は、[Import and Export API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) を使用して W&B SDK API で Runs、メトリクス、Artifacts などを抽出できます。以下の表は、エクスポートの主要なユースケースの一部をまとめたものです。

| 目的 | ドキュメント |
|---|---|
| Projects のメタデータをエクスポート | [Projects API]({{< relref path="/ref/python/public-api/projects.md" lang="ja" >}}) |
| Projects 内の Runs をエクスポート | [Runs API]({{< relref path="/ref/python/public-api/runs.md" lang="ja" >}}) |
| Reports をエクスポート | [Report and Workspace API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ja" >}}) |
| Artifacts をエクスポート | [Artifacts グラフを探索]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}})、[Artifacts をダウンロードして使用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ja" >}}) |

[Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ja" >}}) を使用して Dedicated Cloud に保存されている Artifacts を管理している場合は、W&B SDK API を使用して Artifacts をエクスポートする必要がない場合があります。

{{% alert %}}
W&B SDK API を使用してすべてのデータをエクスポートすると、大量の Runs や Artifacts などがある場合、処理が遅くなる可能性があります。W&B では、Dedicated Cloud インスタンスに過度な負荷をかけないよう、適切なサイズのバッチで実行することを推奨します。
{{% /alert %}}