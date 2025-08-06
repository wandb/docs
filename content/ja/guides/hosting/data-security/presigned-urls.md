---
title: 事前署名付き URL を使用して BYOB にアクセスする
menu:
  default:
    identifier: ja-guides-hosting-data-security-presigned-urls
    parent: data-security
weight: 2
---

W&B は、AI ワークロードやユーザーのブラウザからオブジェクトストレージへのアクセスを簡素化するために、事前署名付き URL を利用しています。事前署名付き URL の基本については、ご利用のクラウドプロバイダーのドキュメントを参照してください。

- [AWS S3 の事前署名付き URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)（[CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage) など S3 互換ストレージにも適用されます）
- [Google Cloud Storage の署名付き URL](https://cloud.google.com/storage/docs/access-control/signed-urls)
- [Azure Blob Storage の共有アクセストークン（SAS）](https://learn.microsoft.com/azure/storage/common/storage-sas-overview)

仕組み:
1. 必要なタイミングで、ネットワーク内の AI ワークロードやユーザーブラウザのクライアントが W&B に事前署名付き URL の発行をリクエストします。
1. W&B がオブジェクトストレージにアクセスし、必要な権限で事前署名付き URL を生成します。
1. W&B がクライアントに事前署名付き URL を返します。
1. クライアントは、その URL を使ってオブジェクトストレージの読み込みや書き込みを行います。

事前署名付き URL の有効期限は以下の通りです：
- **読み込み**：1 時間
- **書き込み**：24 時間（大きなオブジェクトを分割アップロードするため余裕をもたせています）

## チームレベルのアクセス制御

各事前署名付き URL は、W&B プラットフォームの[チームレベルアクセス制御]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}})に基づき、特定のバケットのみに制限されます。ユーザーが [secure storage connector]({{< relref path="./secure-storage-connector.md" lang="ja" >}}) でストレージバケットにマッピングされたチームに所属し、そのユーザーがそのチームにしか所属していない場合、そのユーザーへの事前署名付き URL は他チームに紐付くストレージバケットへはアクセス権を持ちません。

{{% alert %}}
ユーザーは、本来所属すべきチームのみに追加することを W&B では推奨しています。
{{% /alert %}}

## ネットワーク制限
事前署名付き URL を利用して外部ストレージへアクセスできるネットワークを IAM ポリシーで制限することを W&B は推奨しています。これにより、W&B 専用バケットへのアクセスを、AI ワークロードが稼働しているネットワークや、ユーザーマシンに対応するゲートウェイ IP アドレスからのみに限定できます。

- CoreWeave AI Object Storage をご利用の場合、CoreWeave ドキュメントの [Bucket policy reference](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy#condition) をご覧ください。
- AWS S3 または社内設置の MiniIO などの S3 互換ストレージの場合、[S3 ユーザーガイド](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities)、[MinIO ドキュメント](https://github.com/minio/minio)、またはご利用ストレージプロバイダーの説明書を参照してください。

## 監査ログ

W&B では、[W&B 監査ログ]({{< relref path="../monitoring-usage/audit-logging.md" lang="ja" >}})とオブジェクトストレージ固有の監査ログを併用することを推奨します。各クラウドプロバイダーのストレージ監査ログについては以下を参考にしてください：
- [CoreWeave 監査ログ](https://docs.coreweave.com/docs/products/storage/object-storage/concepts/audit-logging#audit-logging-policies)
- [AWS S3 アクセスログ](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)
- [Google Cloud Storage 監査ログ](https://cloud.google.com/storage/docs/audit-logging)
- [Azure blob storage の監視](https://learn.microsoft.com/azure/storage/blobs/monitor-blob-storage)

管理者やセキュリティチームは監査ログを活用することで、誰が W&B のプロダクトで何をしているかを把握でき、必要に応じてユーザーごとに操作権限を制限するなどの対応が可能です。

{{% alert %}}
W&B では、事前署名付き URL を唯一サポートされるオブジェクトストレージへのアクセス手段としています。組織の方針に応じ、上記リストのセキュリティコントロールの全てまたは一部を必ず設定してください。
{{% /alert %}}

## 事前署名付き URL のリクエストユーザーの特定方法

W&B が事前署名付き URL を返却する際、URL に含まれるクエリパラメータでリクエストしたユーザー名を確認できます。

| ストレージプロバイダー          | 署名付き URL のクエリパラメータ |
|-----------------------------|-------------------------|
| CoreWeave AI Object Storage | `X-User`                |
| AWS S3 storage              | `X-User`                |
| Google Cloud storage        | `X-User`                |
| Azure  blob storage         | `scid`                  |