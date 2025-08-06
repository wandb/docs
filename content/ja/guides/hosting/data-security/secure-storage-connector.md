---
title: 自身のバケット（BYOB）の持ち込み
menu:
  default:
    identifier: ja-guides-hosting-data-security-secure-storage-connector
    parent: data-security
weight: 1
---

## 概要
BYOB（Bring Your Own Bucket）は、W&B Artifacts やその他の機密データを自社のクラウドやオンプレミスのインフラストラクチャーに保存できる機能です。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) や [マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、ご自身のバケットに保存したデータは W&B 管理のインフラストラクチャーには複製されません。

{{% alert %}}
* W&B SDK / CLI / UI とバケット間の通信は [事前署名付きURL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を使用して行われます。
* W&B は、W&B Artifacts の削除にガーベジコレクションプロセスを利用します。詳細は[Artifacts の削除]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})をご覧ください。
* バケット設定時にサブパスを指定することで、W&B がバケットのルートフォルダにはファイルを格納しないようにできます。これにより、組織のバケットガバナンスポリシーにより適合することができます。
{{% /alert %}}

### セントラルデータベースとバケットに保存されるデータ
BYOB 機能を利用すると、特定のデータタイプは W&B の中央データベースに、その他のデータタイプはご自身のバケットに保存されます。

#### データベース
- ユーザー、チーム、artifacts、experiments、および projects のメタデータ
- Reports
- 実験ログ
- システムメトリクス
- コンソールログ

#### バケット
- 実験ファイルおよびメトリクス
- Artifact ファイル
- メディアファイル
- Run ファイル
- Parquet 形式でエクスポートされた履歴メトリクスとシステムイベント

### バケットのスコープ
ストレージバケットは、2つのスコープで設定できます。

| スコープ        | 説明 |
|----------------|-------------|
| インスタンスレベル | [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) および [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) では、必要な権限を持つ組織やインスタンス内のどのユーザーも、インスタンスストレージバケット内のファイルにアクセスできます。[マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) には適用されません。|
| チームレベル     | チームにチームレベルストレージバケットを設定すると、チームメンバーはその中にあるファイルにアクセスできます。チームレベルストレージは、機密性の高いデータや厳格なコンプライアンス要件を持つチーム向けに、より細かいデータアクセス制御とデータ分離が可能です。<br><br>チームレベルのストレージは、同一インスタンスを共有する異なる部門や事業単位がインフラや管理リソースを効率よく活用するのに役立ちます。また、プロジェクトごとに異なる顧客案件に対応するAIワークフローを分離管理することも可能です。全てのデプロイメントタイプで利用可能です。チームレベル BYOB はチーム設定時に行います。|

この柔軟な設計により、組織のニーズに応じて様々なストレージ構成が可能です。たとえば:
- 同じバケットをインスタンスと1つ以上のチームで共用できます。
- 各チームが独自のバケットを使ったり、一部のチームがインスタンスバケットに書き込んだり、サブパスを使って複数チームが1つのバケットを共有することもできます。
- 異なるチーム用のバケットは、異なるクラウドインフラやリージョンにホストしたり、ストレージ管理チームごとに管理させることも可能です。

例えば、組織内に Kappa というチームがあるとします。組織（および Team Kappa）はデフォルトでインスタンスレベルバケットを使用しています。次に Omega というチームを作成し、そのチームにはチームレベルストレージバケットを設定します。Team Omega が生成したファイルは Kappa からアクセスできませんが、Kappa のファイルは Omega からアクセスできます。Kappa のデータも分けたい場合は、Kappa 用にもチームレベルストレージバケットを設定してください。

### 提供状況マトリクス
W&B は以下のストレージプロバイダーに接続できます：
- [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage): AIワークロード最適化の高性能S3互換オブジェクトストレージサービス。
- [Amazon S3](https://aws.amazon.com/s3/): 業界最高水準のスケーラビリティ、可用性、セキュリティ、パフォーマンスを提供するオブジェクトストレージサービス。
- [Google Cloud Storage](https://cloud.google.com/storage): 大規模な非構造化データを保存可能なマネージドサービス。
- [Azure Blob Storage](https://azure.microsoft.com/products/storage/blobs): テキスト、バイナリデータ、画像、動画、ログなど大量の非構造化データを保存できるクラウドストレージソリューション。
- S3互換ストレージ（例：[MinIO](https://github.com/minio/minio)）、自社クラウドやオンプレミスでホスト可能。

以下の表は、それぞれのW&BデプロイタイプごとのBYOB利用可否を示します。

| W&Bデプロイタイプ   | インスタンスレベル | チームレベル | 追加情報 |
|--------------------|-------------------|-------------|---------|
| Dedicated Cloud    | &check;           | &check;    | CoreWeave AI Object Storage、Amazon S3、GCP Storage、Microsoft Azure Blob Storage、および [MinIO](https://github.com/minio/minio) などの S3互換ストレージでインスタンス/チームレベルBYOBがサポートされています。|
| マルチテナントクラウド | 該当なし         | &check;    | CoreWeave AI Object Storage、Amazon S3、GCP Storage でチームレベルBYOBがサポートされています。Microsoft Azure のデフォルトおよび専用バケットはW&Bが完全に管理します。|
| セルフマネージド    | &check;           | &check;    | CoreWeave AI Object Storage、Amazon S3、GCP Storage、Microsoft Azure Blob Storage、[MinIO](https://github.com/minio/minio) などの S3互換ストレージでインスタンス/チームレベルBYOBがサポートされています。|

以下のセクションでは、BYOBのセットアップ手順を案内します。

## バケットの用意 {#provision-your-bucket}

[提供可否マトリクスを確認したら]({{< relref path="#availability-matrix" lang="ja" >}})、ストレージバケットの用意（アクセス制御ポリシーやCORSなど）を始めましょう。タブを選択して詳細を確認してください。

{{< tabpane text=true >}}
{{% tab header="CoreWeave" value="coreweave" %}}
<a id="coreweave-requirements"></a>**要件**:
- **Dedicated Cloud** または **Self-Hosted** v0.70.0以降、または **マルチテナントクラウド**
- バケット作成・APIアクセスキー・シークレットキー作成権限を持った CoreWeave アカウント
- W&B インスタンスが CoreWeave ネットワークエンドポイントへ接続できること

詳細は CoreWeave ドキュメントの [Create a CoreWeave AI Object Storage bucket](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/create-bucket) を参照してください。

1. **マルチテナントクラウド**: バケットポリシーに必要な組織IDを取得します。
    1. [W&Bアプリ](https://wandb.ai/) にログイン。
    1. 左ナビゲーションで **Create a new team** をクリック。
    1. 開いたドロワーで **Invite team members** の上にある W&B 組織IDをコピー。
    1. ページは開いたままにしておきます。この後 [W&B 設定]({{< relref path="#configure-byob" lang="ja" >}}) で使用します。
1. CoreWeave で、任意のバケット名と希望する CoreWeave アベイラビリティゾーンでバケットを作成します。W&B が利用するサブパス用フォルダも必要に応じ作成してください。バケット名・アベイラビリティゾーン・APIアクセスキー・シークレットキー・サブパスをメモします。
1. 以下の CORS（クロスオリジンリソースシェアリング）ポリシーをバケットに設定します:
    ```json
    [
      {
        "AllowedHeaders": [
          "*"
        ],
        "AllowedMethods": [
          "GET",
          "HEAD",
          "PUT"
        ],
        "AllowedOrigins": [
          "*"
        ],
        "ExposeHeaders": [
          "ETag"
        ],
        "MaxAgeSeconds": 3000
      }
    ]
    ```
    CoreWeave storage は S3互換です。CORSの詳細は AWS ドキュメント [Configuring cross-origin resource sharing (CORS)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enabling-cors-examples.html) を参照してください。
1. **マルチテナントクラウド**: W&B デプロイメントがバケットへアクセスし[事前署名付きURL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を生成できるよう、必要なアクセス権限を付与したバケットポリシーを設定します。CoreWeave ドキュメントの [Bucket Policy Reference](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy) を参照してください。

    `<cw-bucket>` を CoreWeave バケット名、`<wb-org-id>` を先ほど取得したW&B組織IDで置き換えてください。

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
      {
        "Sid": "AllowWandbUser",
        "Action": [
          "s3:GetObject*",
          "s3:GetEncryptionConfiguration",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:AbortMultipartUpload",
          "s3:DeleteObject",
          "s3:PutObject",
          "s3:GetBucketCORS",
          "s3:GetBucketLocation",
          "s3:GetBucketVersioning"
        ],
        "Effect": "Allow",
        "Resource": [
          "arn:aws:s3:::<cw-bucket>/*",
          "arn:aws:s3:::<cw-bucket>"
        ],
        "Principal": {
          "CW": "arn:aws:iam::wandb:static/wandb-integration"
        },
        "Condition": {
          "StringLike": {
            "wandb:OrgID": [
              "<wb-org-id>"
            ]
          }
        }
      },
      {
        "Sid": "AllowUsersInOrg",
        "Action": "s3:*",
        "Effect": "Allow",
        "Resource": [
          "arn:aws:s3:::<cw-bucket>",
          "arn:aws:s3:::<cw-bucket>/*"
        ],
        "Principal": {
          "CW": "arn:aws:iam::<cw-storage-org-id>:*"
        }
      }]
    }
    ```

"Sid": "AllowUsersInOrg" 以降の記述は、W&B 組織のユーザーに直接バケットアクセスの権限を付与します。この機能が不要な場合は、該当部分を削除しても構いません。

{{% /tab %}}
{{% tab header="AWS" value="aws" %}}
詳細は AWS ドキュメント [Create an S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) を参照してください。
1. KMSキーの用意

    W&B では S3バケット内データの暗号化・復号化のため KMSキーの用意が必須です。キーの使用タイプは `ENCRYPT_DECRYPT` に設定してください。以下のポリシーをキーに割り当てます:

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid" : "Internal",
          "Effect" : "Allow",
          "Principal" : { "AWS" : "<Your_Account_Id>" },
          "Action" : "kms:*",
          "Resource" : "<aws_kms_key.key.arn>"
        },
        {
          "Sid" : "External",
          "Effect" : "Allow",
          "Principal" : { "AWS" : "<aws_principal_and_role_arn>" },
          "Action" : [
            "kms:Decrypt",
            "kms:Describe*",
            "kms:Encrypt",
            "kms:ReEncrypt*",
            "kms:GenerateDataKey*"
          ],
          "Resource" : "<aws_kms_key.key.arn>"
        }
      ]
    }
    ```

    `<Your_Account_Id>` と `<aws_kms_key.key.arn>` をご自身の値にしてください。

    [マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})の場合は、`<aws_principal_and_role_arn>` を下記の値に置き換えてください:

    * [マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
    * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`

    このポリシーにより、AWSアカウントにフルアクセス権を与え、W&B プラットフォーム用のAWSアカウントに必要な権限を付与します。KMSキーARN を記録してください。

1. S3バケットの用意

    AWSアカウントで S3バケットを用意する手順:
    1. 任意の名前で S3 バケットを作成。必要に応じて、W&B ファイル格納用のサブパスに利用するフォルダも作成します。
    1. サーバーサイド暗号化を有効にし、上記KMSキーを使用してください。
    1. 下記のポリシーでCORSを構成します:

        ```json
        [
          {
              "AllowedHeaders": [
                  "*"
              ],
              "AllowedMethods": [
                  "GET",
                  "HEAD",
                  "PUT"
              ],
              "AllowedOrigins": [
                  "*"
              ],
              "ExposeHeaders": [
                  "ETag"
              ],
              "MaxAgeSeconds": 3000
          }
        ]
        ```
        {{% alert %}}[オブジェクトライフサイクル管理ポリシー](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html)でバケットのデータが期限切れになると、一部 run の履歴が読めなくなる場合があります。{{% /alert %}}
    1. W&BプラットフォームをホストするAWSアカウントに必要なS3権限を付与してください。これにより、クラウドインフラやユーザーブラウザからバケットアクセス用の[事前署名付きURL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を発行できます。

        ```json
        {
          "Version": "2012-10-17",
          "Id": "WandBAccess",
          "Statement": [
            {
              "Sid": "WAndBAccountAccess",
              "Effect": "Allow",
              "Principal": { "AWS": "<aws_principal_and_role_arn>" },
                "Action" : [
                  "s3:GetObject*",
                  "s3:GetEncryptionConfiguration",
                  "s3:ListBucket",
                  "s3:ListBucketMultipartUploads",
                  "s3:ListBucketVersions",
                  "s3:AbortMultipartUpload",
                  "s3:DeleteObject",
                  "s3:PutObject",
                  "s3:GetBucketCORS",
                  "s3:GetBucketLocation",
                  "s3:GetBucketVersioning"
                ],
              "Resource": [
                "arn:aws:s3:::<wandb_bucket>",
                "arn:aws:s3:::<wandb_bucket>/*"
              ]
            }
          ]
        }
        ```

        `<wandb_bucket>` を該当バケット名で、KMSキーARN も記録してください。続いて [W&B 設定]({{< relref path="#configure-byob" lang="ja" >}}) を行います。

        [マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})の場合は `<aws_principal_and_role_arn>` を下記の値に置き換えてください。

        * [マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
        * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`
  
詳細は [AWS 自主管理ホスティングガイド]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}}) をご覧ください。

{{% /tab %}}
{{% tab header="GCP" value="gcp"%}}
詳細は GCP ドキュメント [Create a bucket](https://cloud.google.com/storage/docs/creating-buckets) をご覧ください。
1. GCS バケットの用意

    GCP プロジェクト内で GCS バケットを用意する手順:

    1. 任意のバケット名で GCS バケットを作成。必要に応じてサブパス用フォルダも作成してください。
    1. 暗号化タイプを `Google-managed` に設定。
    1. CORS ポリシーを `gsutil` コマンドで設定します。UI からは設定できません。

       1. ローカルで `cors-policy.json` というファイルを作成。
       1. 下記の CORS ポリシーをファイルにコピーして保存。

           ```json
           [
             {
               "origin": ["*"],
               "responseHeader": ["Content-Type"],
               "exposeHeaders": ["ETag"],
               "method": ["GET", "HEAD", "PUT"],
               "maxAgeSeconds": 3000
             }
           ]
           ```

          {{% alert %}}[オブジェクトライフサイクル管理ポリシー](https://cloud.google.com/storage/docs/lifecycle)でバケットのデータが期限切れになると、一部 run の履歴が読めなくなる場合があります。{{% /alert %}}

      1. `<bucket_name>` を正しいバケット名で置き換え、`gsutil` を実行。

          ```bash
          gsutil cors set cors-policy.json gs://<bucket_name>
          ```

      1. ポリシーが正しいか確認します。`<bucket_name>` を実際のバケット名に置換して実行。

          ```bash
          gsutil cors get gs://<bucket_name>
          ```

1. [マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})の場合は、W&Bプラットフォーム連携用のGCPサービスアカウントに `storage.admin` ロールを付与してください。W&BはこのロールでCORSやバージョニング設定などのバケット属性確認を行います。このロールが無い場合、HTTP 403 エラーとなります。

    * [マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `wandb-integration@wandb-production.iam.gserviceaccount.com`
    * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `deploy@wandb-production.iam.gserviceaccount.com`

    バケット名を控えて、続いて [BYOB向けW&Bの設定]({{< relref path="#configure-byob" lang="ja" >}}) に進みます。
{{% /tab %}}

{{% tab header="Azure" value="azure" %}}
詳細は Azure ドキュメント [Create a blob storage container](https://learn.microsoft.com/en-us/azure/storage/blobs/blob-containers-portal) をご覧ください。
1. Azure Blob Storage コンテナの用意

    インスタンスレベルBYOBの場合、[このTerraformモジュール](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) を使用しない場合、以下の手順で Azure サブスクリプションにバケットを用意します:

    1. 任意のバケット名で作成。必要に応じてW&B用サブパスフォルダも作成します。
    1. バケットにCORSポリシーを構成

        UIからCORSを設定するにはblob storageを開き、`Settings/Resource Sharing (CORS)` までスクロールし、以下を入力してください:

        | パラメータ | 値 |
        | --- | --- |
        | Allowed Origins | `*`  |
        | Allowed Methods | `GET`, `HEAD`, `PUT` |
        | Allowed Headers | `*` |
        | Exposed Headers | `*` |
        | Max Age | `3000` |

        {{% alert %}}[オブジェクトライフサイクル管理ポリシー](https://learn.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-policy-configure?tabs=azure-portal)でバケットのデータが期限切れになると、一部 run の履歴が読めなくなる場合があります。{{% /alert %}}
1. ストレージアカウントアクセスキーを発行し、名前とストレージアカウント名を記録してください。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})利用時は、ストレージアカウント名とアクセスキーをW&Bチームと安全な手段で共有してください。

    チームレベルBYOBの場合、[Terraform](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector) によるAzure Blob Storage バケットと必要なアクセス機構・パーミッションの用意がおすすめです。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})利用ならOIDC issuer URLも指定してください。以下の情報を控えてください:

    * ストレージアカウント名
    * ストレージコンテナ名
    * マネージドIDクライアントID
    * Azure テナントID

{{% /tab %}}
{{% tab header="S3-compatible" value="s3-compatible" %}}
S3互換バケットを作成します。以下の情報を控えてください:
- アクセスキー
- シークレットアクセスキー
- URLエンドポイント
- バケット名
- フォルダパス（該当する場合）
- リージョン

{{% /tab %}}
{{< /tabpane >}}

続いて、[ストレージアドレスを決定]({{< relref path="#determine-the-storage-address" lang="ja" >}})しましょう。

## ストレージアドレスの決定  {#determine-the-storage-address}
このセクションでは、W&B チームと BYOB ストレージバケットを接続する際に使用する記法を説明します。例中の < > 内のプレースホルダはバケット固有の値に置き換えてください。
詳細な手順はタブでご確認ください。

{{< tabpane text=true >}}
{{% tab header="CoreWeave" value="coreweave" %}}
このセクションは **Dedicated Cloud** または **Self-Managed** のチームレベルBYOB用です。インスタンスレベルBYOBまたはマルチテナントクラウドについては、すぐに [W&Bの設定]({{< relref path="#configure-byob" lang="ja" >}}) に進めます。

下記フォーマットでバケットパスを決定してください。<> 内はバケットの値で置き換えます。

**バケットフォーマット**:
```none
cw://<accessKey>:<secretAccessKey>@cwobject.com/<bucketName>?tls=true
```

`cwobject.com` のHTTPSエンドポイントがサポートされます。TLS 1.3は必須です。その他CoreWeaveエンドポイントにご興味があれば [support](mailto:support@wandb.com) にご連絡ください。
{{% /tab %}}
{{% tab header="AWS" value="aws" %}}
**バケットフォーマット**:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>?region=<region>
```
アドレス中、`region`パラメータはW&Bインスタンスとバケットの両方がAWS内かつ `AWS_REGION` が一致する場合を除き、必須です。
{{% /tab %}}
{{% tab header="GCP" value="gcp" %}}
**バケットフォーマット**:
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```
{{% /tab %}}
{{% tab header="Azure" value="azure" %}}
**バケットフォーマット**:
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```
{{% /tab %}}
{{% tab header="S3-compatible" value="s3-compatible" %}}
**バケットフォーマット**:
```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>&tls=true
```
このアドレスでは、`region`パラメータが必須です。

{{% alert %}}
このセクションはAWS S3以外でホストされるS3互換ストレージ用です。例えば、オンプレミスで運用する [MinIO](https://github.com/minio/minio) など。AWS S3の場合は **AWS** タブを参照してください。

S3互換モードを持つクラウドネイティブストレージの場合は、可能な限りクラウド専用のプロトコル指定子を使用しましょう。例: CoreWeaveバケットには `cw://` を使う、など。
{{% /alert %}}
{{% /tab %}}
{{< /tabpane >}}

ストレージアドレスを決定したら、[チームレベルBYOBの設定]({{< relref path="#configure-team-level-byob" lang="ja" >}})に進みます。

## W&B の設定  {#configure-byob}
[バケットの用意]({{< relref path="#provision-your-bucket" lang="ja" >}})と[アドレスの決定](#determine-the-storage-address)が完了したら、[インスタンスレベル]({{< relref path="#instance-level-byob" lang="ja" >}})または[チームレベル]({{< relref path="#team-level-byob" lang="ja" >}})でBYOBを設定できます。

{{% alert color="secondary" %}}
ストレージバケットの設計は慎重に行いましょう。W&B 用にバケットを設定した後に別バケットへデータを移行する場合、複雑な作業と W&B のサポートが必要です。この方針は Dedicated Cloud・Self-Managed・Multi-tenant Cloud のチームストレージ全てに該当します。ご不明点は [support](mailto:support@wandb.com) までご連絡ください。
{{% /alert %}}

### インスタンスレベル BYOB

{{% alert %}}
インスタンスレベルで CoreWeave AI Object Storage を利用する場合は、セルフサービス設定には対応していません。[W&Bサポート](mailto:support@wandb.com) までご連絡ください。
{{% /alert %}}

**専用クラウド**の場合: バケット詳細をW&Bチームに共有し、専用クラウドインスタンスの設定を依頼してください。

**セルフマネージド**の場合: W&BアプリからインスタンスレベルBYOBを設定できます。
1. `admin`権限のユーザーでW&Bにログイン。
1. 右上のユーザーアイコンをクリックし、**System Console** を押す。
1. **Settings** > **System Connections** へ移動。
1. **Bucket Storage** セクションで、**Identity**欄のIDが新しいバケットへアクセスできるよう権限を付与してください。
1. **Provider** を選択。
1. **Bucket Name** を入力。
1. 必要があれば新しいバケットの **Path** を入力。
1. **Save** をクリック。

{{% alert %}}
Self-Managed では、W&Bが管理するTerraformモジュールを活用して、必要なアクセス機構やIAMパーミッション付きでストレージバケットを用意することを推奨します:

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [インスタンスレベルBYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) または [チームレベルBYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
{{% /alert %}}

### チームレベル BYOB

バケットの[ストレージ場所を決定](#determine-the-storage-address)したら、チーム作成時に W&B App で設定可能です。

{{% alert %}}
- チームを作成した後、ストレージは変更できません。
- インスタンスレベル BYOB を設定したい場合は [インスタンスレベル BYOB]({{< relref path="#instance-level-byob" lang="ja" >}}) を参照してください。
- CoreWeaveストレージをチームに設定する場合は、事前に [support](mailto:support@wandb.com) に連絡の上でバケット設定確認とチーム設定の検証を依頼してください（チーム作成後、ストレージ詳細は変更できません）。
{{% /alert %}}

デプロイタイプを選択して手順を確認してください。

{{< tabpane text=true >}}
{{% tab header="Dedicated Cloud / Self-Hosted" value="dedicated" %}}

1. **専用クラウド**: アカウントチームへバケットパスを共有し、インスタンスのサポート対象ファイルストアに追加してもらってから以下の手順へ進んでください。
1. **セルフマネージド**: `GORILLA_SUPPORTED_FILE_STORES` 環境変数にバケットパスを追加し、W&B を再起動してから以下の手順へ進んでください。
1. `admin`権限ユーザーでW&Bにログインし、左上のアイコンから左ナビゲーションを開き **Create a team to collaborate** をクリック。
1. チーム名を指定。
1. **Storage Type** を **External storage** に設定。

    {{% alert %}}インスタンスレベルストレージをチームストレージとして使う場合（内部・外部問わず）、**Storage Type** を **Internal** のままにしてください。インスタンスレベルバケットがBYOB設定でも、外部ストレージを使う場合のみ **External** を選択し、バケット情報を次のステップで設定。{{% /alert %}}

1. **Bucket location** をクリック。
1. 既存バケットを使うにはリストから選択。新しいバケットを追加するには下の **Add bucket** をクリックし、バケット詳細を入力。

    **Cloud provider** をクリックし、**CoreWeave**、**AWS**、**GCP**、**Azure** から選択してください。
    
    Cloud provider がリストに無い場合は、手順1でインスタンスのサポートファイルストアにバケットパスを追加したか確認してください。ストレージプロバイダーがそれでも出ない場合は、[support](mailto:support@wandb.ai)までお問い合わせください。
1. バケット詳細を指定します。
    - **CoreWeave** の場合はバケット名のみ
    - Amazon S3・GCP・S3互換ストレージは、[先ほど決めた](#determine-the-storage-address)バケットパス全体を入力
    - Dedicated または Self-Managed 上の Azure では **Account name** にアカウント名、**Container name** にコンテナ名を入力
    - オプション項目:
      - サブパスの場合は **Path** を入力
      - **AWS**: **KMS key ARN** を入力
      - **Azure**: 必要に応じて **Tenant ID** と **Managed Identity Client ID** を入力
1. **Create team** をクリック

万一アクセス権や設定エラーがあった場合は、ページ下部にエラーや警告が表示されます。なければチームが作成されます。

{{% /tab %}}
{{% tab header="Multi-tenant Cloud" value="multi-tenant" %}}

1. 以前に新しいチーム作成を開始したウィンドウに戻り W&B 組織IDを改めて確認するか、admin権限ユーザーでW&Bにログインし、左ナビゲーションの **Create a team to collaborate** からスタートしてください。
1. チーム名を入力
1. **Storage Type** を **External storage** に設定
1. **Bucket location** をクリック
1. 既存バケットを使う場合はリストから選択、新規追加は下の **Add bucket** からバケット詳細を入力

    **Cloud provider** をクリックし、**CoreWeave**、**AWS**、**GCP**、**Azure** を選択
1. バケット詳細を指定
    - **CoreWeave** の場合はバケット名のみ
    - Amazon S3・GCP・S3互換ストレージは、[先ほど決めた](#determine-the-storage-address)バケットパス全体を入力
    - Dedicated または Self-Managed 上の Azure では **Account name** にアカウント名、**Container name** にコンテナ名を入力
    - オプション:
      - サブパスは **Path** に入力
      - **AWS**: KMS キーARNを入力
      - **Azure**: テナントID・Managed Identity Client IDがある場合は入力
     - **Invite team members** でチームメンバーのメールアドレス（カンマ区切り）を入力して招待も可能。作成後に追加もOK
1. **Create team** をクリック

W&B からエラーや警告が表示される場合は設定内容をご確認ください。なければチームが作成されます。

{{% /tab %}}
{{< /tabpane >}}

## トラブルシューティング
<details open>
<summary>CoreWeave AI Object Storage への接続</summary>

- **接続エラー**
  - W&BインスタンスがCoreWeaveネットワークエンドポイントに接続可能か確認してください。
  - CoreWeaveはバーチャルホスト形式のパスを使います。例: `cw://bucket-name.cwobject.com` は正しく、~`cw://cwobject.com/bucket-name/`~ は誤りです。
  - バケット名にアンダースコア（`_`）やDNS互換でない文字は使えません。
  - バケット名はCoreWeaveの全ロケーションで一意である必要があります。
  - バケット名の先頭に `cw-` や `vip-` は使えません（予約済み）。
- **CORS検証失敗**
  - CORSポリシーが必須です。CoreWeaveはS3互換ですので、CORSの詳細は [Configuring cross-origin resource sharing (CORS)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enabling-cors-examples.html) を参照してください。
  - `AllowedMethods` に `GET`、`PUT`、`HEAD` を必ず含めてください。
  - `ExposeHeaders` に `ETag` を必ず加えてください。
  - W&BフロントエンドのドメインをCORSの `AllowedOrigins` に含めてください。このページ記載のCORS例は `*` ですべて許可しています。
- **LOTAエンドポイント関連**
  - W&B からの LOTA エンドポイント接続はまだサポートされていません。ご要望は [support](mailto:support@wandb.com) まで。
- **アクセスキー・権限エラー**
  - CoreWeave API Access Key が有効期限切れでないか確認。
  - CoreWeave API Access Key・Secret Key に `GetObject`、`PutObject`、`DeleteObject`、`ListBucket` の権限があるか確認（本ページサンプルで十分です）。詳しくは [Create and Manage Access Keys](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys) を参照してください。

</details>
