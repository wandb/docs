---
title: Export data from Dedicated cloud
description: 専用クラウド からのデータのエクスポート
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

もし、 専用クラウド インスタンスで管理されている全てのデータをエクスポートしたい場合は、W&B SDK API を使用して、runs、metrics、Artifacts などを抽出できます。詳しくは、[インポートとエクスポート API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を参照してください。以下の表は、主要なエクスポートの ユースケース をまとめたものです。

| 目的 | ドキュメント |
|---------|---------------|
| プロジェクト の メタデータ をエクスポート | [Projects API]({{< relref path="/ref/python/public-api/projects/" lang="ja" >}}) |
| プロジェクト 内の runs をエクスポート | [Runs API]({{< relref path="/ref/python/public-api/runs/" lang="ja" >}}) |
| Reports をエクスポート | [Reports API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ja" >}}) |
| Artifacts をエクスポート | [アーティファクト グラフの探索]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ja" >}}), [アーティファクト のダウンロードと利用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ja" >}}) |

[Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ja" >}}) で 専用クラウド に保存されている Artifacts を管理している場合、W&B SDK API を使用して Artifacts をエクスポートする必要はないかもしれません。

{{% alert %}}
W&B SDK API を使用してすべてのデータをエクスポートすると、多数の runs や Artifacts がある場合に処理が遅くなる可能性があります。W&B では、 専用クラウド インスタンスに負荷がかかりすぎないように、適切なサイズのバッチでエクスポート プロセス を実行することをお勧めします。
{{% /alert %}}
