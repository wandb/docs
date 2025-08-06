---
title: 署名付き URL を使用して BYOB にアクセスする
menu:
  default:
    identifier: presigned-urls
    parent: data-security
weight: 2
---

W&B は、事前署名付き URL を使用して、AI ワークロードやユーザーブラウザからの BLOB ストレージへのアクセスを簡素化します。事前署名付き URL の基本的な情報については、各クラウドプロバイダーのドキュメントを参照してください。

- [AWS S3 の事前署名付き URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)（[CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage) などの S3 互換ストレージでも利用可能）
- [Google Cloud Storage の署名付き URL](https://cloud.google.com/storage/docs/access-control/signed-urls)
- [Azure Blob Storage の共有アクセス署名](https://learn.microsoft.com/azure/storage/common/storage-sas-overview)

仕組み:
1. 必要に応じて、ネットワーク内の AI ワークロードやユーザーブラウザクライアントが W&B に事前署名付き URL をリクエストします。
1. W&B はリクエストを受信し、BLOB ストレージにアクセスして必要な権限付きの事前署名 URL を生成します。
1. W&B がクライアントに事前署名付き URL を返します。
1. クライアントは、その URL を利用して BLOB ストレージの読み書きを行います。

事前署名付き URL の有効期限:
- **読み込み**: 1 時間
- **書き込み**: 24 時間（大きなオブジェクトをチャンクごとにアップロードするための余裕時間）

## チームレベルのアクセス制御

各事前署名付き URL は、W&B プラットフォームの [チームレベルのアクセス制御]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" >}}) に基づき、特定のバケットへのアクセスに制限されています。ユーザーが [Secure Storage Connector]({{< relref "./secure-storage-connector.md" >}}) を使ってストレージバケットに紐づけられたチームのメンバーで、そのユーザーが他のチームには所属していない場合、そのユーザー向けに生成された事前署名付き URL には他チームに紐づけられたバケットへのアクセス権限は付与されません。

{{% alert %}}
W&B では、必要なチームのみユーザーに追加することを推奨しています。
{{% /alert %}}

## ネットワーク制限
W&B では、IAM ポリシーを利用して、ネットワークごとに事前署名付き URL を使用した外部ストレージへのアクセスを制限することを推奨しています。これにより、W&B 専用バケットへのアクセスを、AI ワークロードが実行されているネットワーク、またはユーザーマシンに対応するゲートウェイ IP アドレスのみに限定できます。

- CoreWeave AI Object Storage の場合、CoreWeave ドキュメントの [バケットポリシーリファレンス](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy#condition) を参照してください。
- AWS S3 や MiniIO などの S3 互換ストレージの場合は、[S3 ユーザーガイド](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities)、[MinIO ドキュメント](https://github.com/minio/minio)、または利用している S3 互換ストレージプロバイダーのドキュメントを参照してください。

## 監査ログ

W&B では、[W&B 監査ログ]({{< relref "../monitoring-usage/audit-logging.md" >}}) と BLOB ストレージ固有の監査ログを併用することを推奨しています。各クラウドプロバイダーの BLOB ストレージ監査ログについては以下をご参照ください。
- [CoreWeave の監査ログ](https://docs.coreweave.com/docs/products/storage/object-storage/concepts/audit-logging#audit-logging-policies)
- [AWS S3 アクセスログ](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)
- [Google Cloud Storage 監査ログ](https://cloud.google.com/storage/docs/audit-logging)
- [Azure Blob Storage の監視](https://learn.microsoft.com/azure/storage/blobs/monitor-blob-storage)

管理者やセキュリティチームは監査ログを用いることで、誰が W&B 製品上で何をしているかを記録・監視し、一部のユーザーに対し制限が必要かどうか判断・対応できます。

{{% alert %}}
W&B 上でサポートされている BLOB ストレージのアクセス手段は事前署名付き URL のみです。上記のセキュリティ制御策のいずれか、またはすべてを組織の要件に合わせて導入してください。
{{% /alert %}}

## 事前署名付き URL をリクエストしたユーザーを特定する
W&B が事前署名付き URL を返す際、URL 内のクエリパラメータにリクエスト元のユーザー名が含まれています。

| ストレージプロバイダー           | 署名付き URL のクエリパラメータ |
|-------------------------------|----------------|
| CoreWeave AI Object Storage   | `X-User`       |
| AWS S3 storage                | `X-User`       |
| Google Cloud storage          | `X-User`       |
| Azure blob storage            | `scid`         |