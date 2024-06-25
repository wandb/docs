---
description: 専用クラウドからデータをエクスポート
displayed_sidebar: default
---


# Dedicated Cloud からのデータエクスポート

専用クラウドインスタンスで管理されているすべてのデータをエクスポートしたい場合は、W&B SDK API を使用して runs、metrics、artifacts などを抽出し、API に従って別のクラウドまたはオンプレミスストレージにログとして保存できます。

データエクスポートのユースケースについては、[Import and Export Data](../track/public-api-guide#export-data) を参照してください。別のユースケースとして、専用クラウドの利用契約を終了する予定がある場合、W&B がインスタンスを終了する前に関連するデータをエクスポートしたいと思うかもしれません。

データエクスポート API と関連ドキュメントへのリンクについては、以下の表を参照してください:

| 目的 | ドキュメント |
|-----|--------------|
| プロジェクトのメタデータをエクスポート | [Projects API](../../ref/python/public-api/api#projects) |
| プロジェクト内の runs をエクスポート | [Runs API](../../ref/python/public-api/api#runs), [Export run data](../track/public-api-guide#export-run-data), [Querying multiple runs](../track/public-api-guide#querying-multiple-runs) |
| Reports をエクスポート | [Reports API](../../ref/python/public-api/api#reports) |
| Artifacts をエクスポート | [Artifact API](../../ref/python/public-api/api#artifact), [Explore and traverse an artifact graph](../artifacts/explore-and-traverse-an-artifact-graph#traverse-an-artifact-programmatically), [Download and use an artifact](../artifacts/download-and-use-an-artifact#download-and-use-an-artifact-stored-on-wb) |

:::info
専用クラウドに保存されている artifacts は、[Secure Storage Connector](./data-security/secure-storage-connector) を使用して管理します。その場合、W&B SDK API を使用して artifacts をエクスポートする必要はないかもしれません。
:::

:::note
大量の runs や artifacts などのデータをすべてエクスポートするには、W&B SDK API を使用すると時間がかかる場合があります。W&B は、専用クラウドインスタンスが過負荷にならないよう、適切なサイズのバッチでエクスポートプロセスを実行することを推奨します。
:::
