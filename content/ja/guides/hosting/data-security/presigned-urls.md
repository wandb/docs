---
title: Access BYOB using pre-signed URLs
menu:
  default:
    identifier: ja-guides-hosting-data-security-presigned-urls
    parent: data-security
weight: 2
---

W&B は、AI ワークロードまたは ユーザー のブラウザーからの blob ストレージへの アクセス を簡素化するために、事前署名付き URL を使用します。事前署名付き URL の基本情報については、[AWS S3 の事前署名付き URL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)、[Google Cloud Storage の署名付き URL](https://cloud.google.com/storage/docs/access-control/signed-urls)、[Azure Blob Storage の Shared Access Signature](https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview) を参照してください。

必要な場合、ネットワーク内の AI ワークロードまたは ユーザー ブラウザー クライアントは、W&B Platform から事前署名付き URL を要求します。W&B Platform は、関連する blob ストレージに アクセス し、必要な権限を持つ事前署名付き URL を生成し、クライアントに返します。次に、クライアントは事前署名付き URL を使用して blob ストレージに アクセス し、 オブジェクト のアップロードまたは取得操作を行います。 オブジェクト のダウンロードの URL 有効期限は 1 時間、 オブジェクト のアップロードは 24 時間です。これは、一部の大きな オブジェクト をチャンク単位でアップロードするのに時間がかかる場合があるためです。

## Team レベルの アクセス 制御

各事前署名付き URL は、W&B platform の [team level access control]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" lang="ja" >}}) に基づいて、特定の バケット に制限されています。 ユーザー が [secure storage connector]({{< relref path="./secure-storage-connector.md" lang="ja" >}}) を使用して blob ストレージ バケット にマッピングされている team の一部であり、その ユーザー がその team の一部である場合、リクエスト に対して生成された事前署名付き URL には、他の team にマッピングされた blob ストレージ バケット に アクセス する権限はありません。

{{% alert %}}
W&B は、 ユーザー が所属するはずの team のみに追加することをお勧めします。
{{% /alert %}}

## ネットワーク制限

W&B は、 バケット で IAM ポリシー ベースの制限を使用することにより、事前署名付き URL を使用して blob ストレージに アクセス できるネットワークを制限することをお勧めします。

AWS の場合、[VPC または IP アドレス ベースのネットワーク制限](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities) を使用できます。これにより、W&B 固有の バケット には、AI ワークロードが実行されているネットワークからのみ、または ユーザー が W&B UI を使用して アーティファクト に アクセス する場合、 ユーザー マシンにマッピングされるゲートウェイ IP アドレスからのみ アクセス されるようになります。

## 監査 ログ

W&B は、blob ストレージ 固有の 監査 ログ に加えて、[W&B audit logs]({{< relref path="../monitoring-usage/audit-logging.md" lang="ja" >}}) を使用することもお勧めします。後者については、[AWS S3 access logs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)、[Google Cloud Storage audit logs](https://cloud.google.com/storage/docs/audit-logging) および [Monitor Azure blob storage](https://learn.microsoft.com/en-us/azure/storage/blobs/monitor-blob-storage) を参照してください。管理者およびセキュリティ team は、 監査 ログ を使用して、W&B 製品でどの ユーザー が何をしているかを追跡し、特定の ユーザー に対して一部の操作を制限する必要があると判断した場合に必要なアクションを実行できます。

{{% alert %}}
事前署名付き URL は、W&B でサポートされている唯一の blob ストレージ アクセス メカニズムです。W&B は、リスク許容度に応じて、上記のセキュリティ制御のリストの一部または全部を構成することをお勧めします。
{{% /alert %}}
