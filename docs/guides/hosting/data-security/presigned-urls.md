---
displayed_sidebar: default
---


# プリサインドURL

W&Bは、AIワークロードやユーザーブラウザからのblobストレージへのアクセスを簡素化するために、プリサインドURLを使用します。プリサインドURLの基本情報については、[AWS S3のプリサインドURL](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)、[Google Cloud Storageのサイン付きURL](https://cloud.google.com/storage/docs/access-control/signed-urls)や[Azure Blob Storageの共有アクセス署名](https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview)を参照してください。

必要に応じて、ネットワーク内のAIワークロードやユーザーブラウザクライアントは、W&BプラットフォームからプリサインドURLをリクエストします。W&Bプラットフォームは関連するblobストレージにアクセスし、必要な権限を持つプリサインドURLを生成してクライアントに返します。クライアントはその後、blobストレージへのオブジェクトのアップロードまたは取得操作のためにプリサインドURLを使用します。オブジェクトのダウンロード用URLの有効期限は1時間、アップロード用URLの有効期限は24時間で、大きなオブジェクトの場合、チャンクごとのアップロードに時間がかかるためです。

## チームレベルのアクセス制御

各プリサインドURLは、W&Bプラットフォーム内の[チームレベルのアクセス制御](../iam/manage-users.md#manage-a-team)に基づいて特定のバケットに制限されます。ユーザーが[セキュリティストレージコネクタ](./secure-storage-connector.md)を使用してblobストレージバケットにマップされたチームの一員であり、そのユーザーがそのチームにのみ所属している場合、そのリクエストに対して生成されたプリサインドURLは、他のチームにマップされたblobストレージバケットにアクセスする権限を持ちません。

:::info
W&Bは、ユーザーを所属する必要のあるチームのみに追加することを推奨します。
:::

## ネットワーク制限

W&Bは、バケットに対するIAMポリシーに基づく制限を使用して、プリサインドURLを使用してblobストレージにアクセスできるネットワークを制限することを推奨します。

AWSの場合、[VPCまたはIPアドレスに基づくネットワーク制限](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html#PresignedUrlUploadObject-LimitCapabilities)を使用できます。これにより、W&B特定のバケットがAIワークロードが実行されているネットワークからのみアクセスされるようにするか、ユーザーがW&B UIを使用してArtifactsにアクセスする場合、ユーザーマシンにマップされるゲートウェイIPアドレスからアクセスされるようにします。

## 監査ログ

W&Bは、blobストレージ特有の監査ログに加えて、[W&B監査ログ](../monitoring-usage/audit-logging.md)の使用も推奨します。後者については、[AWS S3アクセスログ](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ServerLogs.html)、[Google Cloud Storage監査ログ](https://cloud.google.com/storage/docs/audit-logging)や[Azure blob storageのモニタリング](https://learn.microsoft.com/en-us/azure/storage/blobs/monitor-blob-storage)を参照してください。監査ログを使用することで、管理者およびセキュリティチームは、W&B製品内でどのユーザーが何をしているかを追跡し、特定のユーザーに対する操作を制限する必要がある場合に適切な対応を取ることができます。

:::note
プリサインドURLは、W&Bでサポートされている唯一のblobストレージアクセスメカニズムです。W&Bは、上記リスト内のいくつかまたはすべてのセキュリティ制御をリスク許容度に応じて設定することを推奨します。
:::