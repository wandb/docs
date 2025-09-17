---
title: 事前署名付き URL を使用して BYOB にアクセス
menu:
  default:
    identifier: ja-guides-hosting-data-security-presigned-urls
    parent: data-security
weight: 2
---

W&B は、AI ワークロードまたはユーザーのブラウザからの blob ストレージへのアクセスを簡素化するために、事前署名済み URL を使用します。事前署名済み URL の基本的な情報については、クラウドプロバイダーのドキュメントを参照してください。
- [AWS S3 の事前署名済み URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)。これは、[CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage) のような S3 互換ストレージにも適用されます。
- [Google Cloud Storage の署名済み URL](https://cloud.google.com/storage/docs/access-control/signed-urls)
- [Azure Blob Storage の共有アクセス署名](https://learn.microsoft.com/azure/storage/common/storage-sas-overview)

仕組み:
1. 必要に応じて、ネットワーク内の AI ワークロードまたはユーザー ブラウザ クライアントは、W&B から事前署名済み URL をリクエストします。
1. W&B は、blob ストレージにアクセスして必要な権限を持つ事前署名済み URL を生成することで、リクエストに応答します。
1. W&B は、事前署名済み URL をクライアントに返します。
1. クライアントは、事前署名済み URL を使用して blob ストレージの読み取りまたは書き込みを行います。

事前署名済み URL は、以下の期間で期限切れになります:
- **読み取り**: 1 時間
- **書き込み**: 24 時間 (大きなオブジェクトをチャンクでアップロードするためにより多くの時間を確保するため)

## チームレベルのアクセス制御
各事前署名済み URL は、W&B プラットフォームの[チームレベルのアクセス制御]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}})に基づいて特定のバケットに制限されます。[セキュアストレージコネクタ]({{< relref path="./secure-storage-connector.md" lang="ja" >}})を使用してストレージ バケットにマッピングされているチームにユーザーが所属しており、そのユーザーがそのチームのみに所属している場合、そのリクエストに対して生成された事前署名済み URL は、他のチームにマッピングされたストレージ バケットにアクセスする権限を持ちません。
{{% alert %}}
W&B は、ユーザーを所属すべきチームのみに追加することを推奨します。
{{% /alert %}}

## ネットワーク制限
W&B は、IAM ポリシーを使用して、事前署名済み URL を使用して外部ストレージにアクセスできるネットワークを制限することを推奨します。これにより、W&B 専用のバケットが、AI ワークロードが実行されているネットワーク、またはユーザー マシンにマップされているゲートウェイ IP アドレスからのみアクセスされるようになります。
- CoreWeave AI Object Storage の詳細については、CoreWeave ドキュメントの[バケットポリシーリファレンス](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy#condition)を参照してください。
- AWS S3 またはオンプレミスでホストされている MinIO のような S3 互換ストレージの詳細については、[S3 ユーザーガイド](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities)、[MinIO ドキュメント](https://github.com/minio/minio)、または S3 互換ストレージプロバイダーのドキュメントを参照してください。

## 監査ログ
W&B は、[W&B 監査ログ]({{< relref path="../monitoring-usage/audit-logging.md" lang="ja" >}})を blob ストレージ固有の監査ログと組み合わせて使用することを推奨します。blob ストレージの監査ログについては、各クラウドプロバイダーのドキュメントを参照してください。
- [CoreWeave 監査ログ](https://docs.coreweave.com/docs/products/storage/object-storage/concepts/audit-logging#audit-logging-policies)
- [AWS S3 アクセスログ](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)
- [Google Cloud Storage 監査ログ](https://cloud.google.com/storage/docs/audit-logging)
- [Azure Blob ストレージの監視](https://learn.microsoft.com/azure/storage/blobs/monitor-blob-storage)

管理者とセキュリティ チームは、監査ログを使用して、どのユーザーが W&B 製品で何をしているかを追跡し、特定のユーザーに対して一部の操作を制限する必要があると判断した場合に、必要な措置を講じることができます。
{{% alert %}}
事前署名済み URL は、W&B でサポートされている唯一の blob ストレージアクセスメカニズムです。W&B は、組織のニーズに応じて、上記のセキュリティ制御のリストの一部またはすべてを構成することを推奨します。
{{% /alert %}}

## 事前署名済み URL をリクエストしたユーザーの特定
W&B が事前署名済み URL を返すと、URL 内のクエリ パラメータにリクエスターのユーザー名が含まれます:
| ストレージプロバイダー            | 署名済み URL クエリ パラメータ |
|-----------------------------|-----------|
| CoreWeave AI Object Storage | `X-User`  |
| AWS S3 Storage              | `X-User`  |
| Google Cloud Storage        | `X-User`  |
| Azure Blob Storage          | `scid`    |