---
title: Access BYOB using pre-signed URLs
menu:
  default:
    identifier: ja-guides-hosting-data-security-presigned-urls
    parent: data-security
weight: 2
---

W&B は、事前に署名された URL を使用して、AI ワークロードやユーザー ブラウザからの BLOB ストレージへのアクセスを簡素化します。事前署名付き URL の基本情報については、[AWS S3 の事前署名付き URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)、[Google Cloud Storage の署名付き URL](https://cloud.google.com/storage/docs/access-control/signed-urls) および [Azure Blob Storage の共有アクセス署名](https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview) をご参照ください。

必要に応じて、ネットワーク内の AI ワークロードまたはユーザー ブラウザ クライアントは、W&B プラットフォームから事前署名付き URL をリクエストします。W&B プラットフォームは関連する BLOB ストレージにアクセスし、必要な権限を持つ事前署名付き URL を生成してクライアントに返します。その後、クライアントは事前署名付き URL を使用して BLOB ストレージにアクセスし、オブジェクトのアップロードや取得操作を行います。オブジェクト ダウンロードの URL の有効期限は 1 時間、オブジェクト アップロードの場合は 24 時間です。これは、大規模なオブジェクトのアップロードに時間がかかる場合があるためです。

## チームレベルのアクセス制御

各事前署名付き URL は、W&B プラットフォームの [チームレベルのアクセス制御]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) に基づいて特定のバケットに制限されています。ユーザーが [セキュア ストレージ コネクタ]({{< relref path="./secure-storage-connector.md" lang="ja" >}}) を使用して BLOB ストレージ バケットにマッピングされたチームの一員であり、そのユーザーがそのチームにしか属していない場合、その要求に対して生成される事前署名付き URL は、他のチームにマッピングされた BLOB ストレージ バケットにアクセスする権限を持ちません。

{{% alert %}}
W&B は、ユーザーを関連するチームにのみ追加することを推奨します。
{{% /alert %}}

## ネットワーク制限

W&B は、IAM ポリシーに基づくバケットの制限を使用し、BLOB ストレージにアクセスするための事前署名付き URL を使用できるネットワークを制限することを推奨します。

AWS の場合、[VPC または IP アドレスベースのネットワーク制限](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities) を使用できます。これにより、W&B 特定のバケットが AI ワークロードが実行されているネットワークから、またはユーザーが W&B UI を使用して Artifacts にアクセスする場合にユーザーのマシンにマッピングされているゲートウェイ IP アドレスからのみアクセスできるようになります。

## 監査ログ

W&B は、BLOB ストレージ固有の監査ログに加えて、[W&B 監査ログ]({{< relref path="../monitoring-usage/audit-logging.md" lang="ja" >}}) を使用することも推奨します。後者については、[AWS S3 アクセスログ](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)、[Google Cloud Storage 監査ログ](https://cloud.google.com/storage/docs/audit-logging)、[Azure BLOB ストレージの監視](https://learn.microsoft.com/en-us/azure/storage/blobs/monitor-blob-storage) を参照してください。管理者やセキュリティ チームは監査ログを使用して、W&B 製品内で誰が何をしているかを追跡し、特定のユーザーに対して操作を制限する必要があると判断した場合に必要な措置を取ることができます。

{{% alert %}}
事前署名付き URL は、W&B でサポートされている唯一の BLOB ストレージ アクセス メカニズムです。W&B は、リスク許容度に応じて、上記のセキュリティ管理の一部またはすべてを構成することを推奨します。
{{% /alert %}}