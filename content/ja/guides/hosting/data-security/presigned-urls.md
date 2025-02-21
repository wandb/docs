---
title: Access BYOB using pre-signed URLs
menu:
  default:
    identifier: ja-guides-hosting-data-security-presigned-urls
    parent: data-security
weight: 2
---

W&B は、AI ワークロードまたは ユーザー のブラウザーから blob ストレージへのアクセスを簡素化するために、事前署名付き URL を使用します。事前署名付き URL の基本情報については、[AWS S3 の事前署名付き URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)、[Google Cloud Storage の署名付き URL](https://cloud.google.com/storage/docs/access-control/signed-urls) 、および [Azure Blob Storage の Shared Access Signature](https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview) を参照してください。

必要な場合、ネットワーク内の AI ワークロードまたは ユーザー ブラウザー クライアントは、W&B Platform から事前署名付き URL を要求します。次に、W&B Platform は関連する blob ストレージにアクセスして、必要な権限を持つ事前署名付き URL を生成し、クライアントに返します。その後、クライアントは事前署名付き URL を使用して blob ストレージにアクセスし、オブジェクト のアップロードまたは取得操作を行います。オブジェクト のダウンロードの URL 有効期限は 1 時間、オブジェクト のアップロードは 24 時間です。これは、一部の大きなオブジェクト では、チャンク単位でアップロードするのにもっと時間が必要になる場合があるためです。

## チーム レベル の アクセス 制御

各事前署名付き URL は、W&B platform の [チーム レベル の アクセス 制御]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) に基づいて、特定の バケット に制限されます。 [セキュア ストレージ コネクター]({{< relref path="./secure-storage-connector.md" lang="ja" >}}) を使用して blob ストレージ バケット にマッピングされている チーム の一部である ユーザー が、その チーム の一部のみである場合、その要求に対して生成された事前署名付き URL には、他の チーム にマッピングされた blob ストレージ バケット にアクセスする権限はありません。

{{% alert %}}
W&B では、 ユーザー を、参加する必要のある チーム のみに参加させることをお勧めします。
{{% /alert %}}

## ネットワーク制限

W&B では、 バケット で IAM ポリシー ベースの制限を使用することにより、事前署名付き URL を使用して blob ストレージにアクセスできるネットワークを制限することをお勧めします。

AWS の場合、[VPC または IP アドレス ベースのネットワーク制限](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities) を使用できます。これにより、W&B 固有の バケット には、AI ワークロード が実行されているネットワークからのみ、または ユーザー が W&B UI を使用して Artifacts にアクセスする場合、 ユーザー マシンにマッピングされる ゲートウェイ IP アドレスからのみアクセスできるようになります。

## 監査 ログ

W&B では、blob ストレージ 固有の監査 ログ に加えて、[W&B 監査 ログ]({{< relref path="../monitoring-usage/audit-logging.md" lang="ja" >}}) を使用することもお勧めします。後者については、[AWS S3 アクセス ログ](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)、[Google Cloud Storage 監査 ログ](https://cloud.google.com/storage/docs/audit-logging) 、および [Azure blob ストレージ の監視](https://learn.microsoft.com/en-us/azure/storage/blobs/monitor-blob-storage) を参照してください。管理者およびセキュリティ チーム は、監査 ログ を使用して、W&B 製品 でどの ユーザー が何をしているかを追跡し、特定の ユーザー に対して一部の操作を制限する必要があると判断した場合に必要な措置を講じることができます。

{{% alert %}}
事前署名付き URL は、W&B でサポートされている唯一の blob ストレージ アクセス メカニズムです。W&B では、リスク許容度に応じて、上記のセキュリティ コントロール の一部またはすべてを構成することをお勧めします。
{{% /alert %}}
