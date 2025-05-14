---
title: BYOB に事前署名済みの URL を使用してアクセスする
menu:
  default:
    identifier: ja-guides-hosting-data-security-presigned-urls
    parent: data-security
weight: 2
---

W&B は事前署名付き URL を使用して、AI ワークロードやユーザー ブラウザからの blob ストレージへのアクセスを簡素化します。事前署名付き URL の基本情報については、[AWS S3 の事前署名付き URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)、[Google Cloud Storage の署名付き URL](https://cloud.google.com/storage/docs/access-control/signed-urls)、[Azure Blob Storage の共有アクセス署名](https://learn.microsoft.com/azure/storage/common/storage-sas-overview) を参照してください。

必要に応じて、ネットワーク内の AI ワークロードまたはユーザー ブラウザー クライアントが W&B プラットフォームから事前署名付き URL を要求します。その後、W&B プラットフォームは関連する blob ストレージにアクセスして、必要な権限で事前署名付き URL を生成し、クライアントに返します。クライアントは、事前署名付き URL を使用して blob ストレージにアクセスし、オブジェクトのアップロードまたは取得操作を行います。オブジェクトのダウンロードの URL は 1 時間で期限切れになり、巨大なオブジェクトをチャンクでアップロードするのに時間がかかる可能性があるため、オブジェクトのアップロードについては 24 時間有効です。

## チームレベルのアクセス制御

各事前署名付き URL は、W&B プラットフォームの[チームレベルのアクセス制御]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) に基づいて特定のバケットに限定されます。ユーザーが[セキュア ストレージ コネクタ]({{< relref path="./secure-storage-connector.md" lang="ja" >}})を使用して blob ストレージ バケットにマッピングされているチームの一員であり、そのユーザーがそのチームにのみ属している場合、彼らの要求に対して生成された事前署名付き URL には、他のチームにマッピングされている blob ストレージ バケットにアクセスする権限がありません。

{{% alert %}}
W&B は、ユーザーを所属すべきチームのみに追加することを推奨します。
{{% /alert %}}

## ネットワーク制限

W&B は、IAM ポリシーに基づくバケットの制限を使用して、事前署名付き URL を使用して blob ストレージにアクセスできるネットワークを制限することを推奨します。

AWS の場合、[VPC または IP アドレスに基づくネットワーク制限](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities)を使用できます。これにより、あなたの AI ワークロードが稼働しているネットワークから、または W&B UI を使用してアーティファクトにアクセスする場合にユーザーマシンにマッピングされるゲートウェイの IP アドレスから、のみ W&B に特化したバケットにアクセスできることが保証されます。

## 監査ログ

W&B は、blob ストレージ固有の監査ログに加えて、[W&B 監査ログ]({{< relref path="../monitoring-usage/audit-logging.md" lang="ja" >}})を使用することを推奨します。後者については、[AWS S3 のアクセスログ](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)、[Google Cloud Storage の監査ログ](https://cloud.google.com/storage/docs/audit-logging)、[Azure blob storage の監視](https://learn.microsoft.com/azure/storage/blobs/monitor-blob-storage)を参照してください。管理者とセキュリティ チームは、W&B 製品でどのユーザーが何をしているかを追跡し、特定のユーザーに対していくつかの操作を制限する必要があると判断した場合に必要な対策を講じるために監査ログを使用できます。

{{% alert %}}
事前署名付き URL は、W&B でサポートされている唯一の blob ストレージ アクセス メカニズムです。W&B は、リスク許容度に応じて、セキュリティ制御の上記リストの一部またはすべてを設定することを推奨します。
{{% /alert %}}