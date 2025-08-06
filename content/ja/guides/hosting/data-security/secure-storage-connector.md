---
title: 独自バケットの持ち込み（BYOB）
menu:
  default:
    identifier: secure-storage-connector
    parent: data-security
weight: 1
---

## 概要

BYOB（Bring your own bucket）は、W&B Artifactsやその他の機密データを自分のクラウドやオンプレミスインフラストラクチャーに保存できる機能です。[専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) や [マルチテナントクラウド]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) の場合、ご自身のバケットに保存したデータは W&B の管理インフラストラクチャーへコピーされません。

{{% alert %}}
* W&B SDK / CLI / UI とバケット間の通信には、[事前署名付きURL]({{< relref "./presigned-urls.md" >}})が使用されます。
* W&B はガベージコレクションプロセスにより W&B Artifacts を削除します。詳細は [Artifacts の削除]({{< relref "/guides/core/artifacts/manage-data/delete-artifacts.md" >}}) を参照してください。
* バケット設定時にサブパスを指定することで、W&B がバケットルート直下のフォルダにファイルを書き込まないようにできます。これにより、組織のバケットガバナンスポリシーにより柔軟に準拠できます。
{{% /alert %}}

### 中央データベースとバケットに保存されるデータ

BYOB 機能を利用すると、W&B の中央データベースに保存されるデータと、ご自身のバケットに保存されるデータの種類が分かれます。

#### データベースに保存されるもの

- Users、Teams、Artifacts、Experiments、Projects のメタデータ
- Reports
- Experiment logs
- システムメトリクス
- コンソールログ

#### バケットに保存されるもの

- Experimentのファイルやメトリクス
- Artifact ファイル
- メディアファイル
- Runファイル
- Parquet 形式のエクスポート済み履歴メトリクスやシステムイベント

### バケットのスコープ

ストレージバケットのスコープは次の2通りです：

| スコープ         | 説明 |
|------------------|------|
| インスタンスレベル | [専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}})や[自己管理型]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}})で、組織やインスタンス内で必要な権限を持つユーザーはインスタンスのストレージバケットのファイルにアクセスできます。[マルチテナントクラウド]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}})では利用できません。 |
| チームレベル    | W&B Team ごとに設定した場合、そのチームメンバーのみがバケット内のファイルにアクセスできます。チームレベルのストレージバケットは、より厳格なデータアクセス制御やデータ分離を要するチームに最適です。<br><br>例えば複数事業部や異なる顧客案件ごとにチームを分けて、インフラや管理リソースを効率的に利用できます。全てのデプロイメントタイプで利用可能です。チームレベルの BYOB はチーム作成時に設定します。 |

この柔軟な設計により、組織のニーズに応じてさまざまなストレージトポロジーを選択できます。例：

- インスタンスと複数チームで同じバケットを利用できる
- チームごとに異なるバケットを使うことも、インスタンスバケットや複数チームでサブパスを使って1つのバケットを共有することも可能
- チームによって異なるクラウドやロケーションのバケットを利用でき、それぞれの管理者が運用可能

例えば、「Kappa」というチームがデフォルトでインスタンスレベルのストレージバケットを使用している場合、「Omega」という新しいチームを作成し、そのチームのみのチームレベルストレージバケットを設定できます。この場合、Team Omegaが生成したファイルはTeam Kappaから見えませんが、Team Kappaが生成したファイルはTeam Omegaからアクセス可能です。Team Kappa のデータも分離したい場合は、Kappaチームにもチームレベルバケットを設定してください。

### 利用可能なストレージプロバイダー

W&B は次のストレージプロバイダーに接続できます。

- [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage)：AI ワークロード向けに最適化された高速・S3互換オブジェクトストレージ
- [Amazon S3](https://aws.amazon.com/s3/)：業界最高水準を誇るスケーラビリティ、可用性、セキュリティ、パフォーマンスのオブジェクトストレージ
- [Google Cloud Storage](https://cloud.google.com/storage)：大規模な非構造データストレージ
- [Azure Blob Storage](https://azure.microsoft.com/products/storage/blobs)：膨大な量の非構造データ（テキスト、バイナリ、画像など）用のクラウド基本ストレージ
- S3互換ストレージ（例 [MinIO](https://github.com/minio/minio) など）、ご自身のクラウドやオンプレでホスト可能

以下の表は、各W&Bデプロイメントタイプごとのスコープ別 BYOB の利用可否を示しています。

| W&Bデプロイメント種別  | インスタンスレベル | チームレベル | 補足情報 |
|------------------------|------------------|------------|---------|
| 専用クラウド            | &check;         | &check;    | CoreWeave AI Object Storage, Amazon S3, GCP Storage, Microsoft Azure Blob Storage、およびオンプレやクラウドでホストするS3互換ストレージ(MinIO等)が対応 |
| マルチテナントクラウド  | 適用不可         | &check;    | CoreWeave AI Object Storage, Amazon S3, GCP Storageのチームレベル BYOB対応。Microsoft AzureはW&Bがバケットを管理します。 |
| 自己管理型              | &check;         | &check;    | CoreWeave AI Object Storage, Amazon S3, GCP Storage, Microsoft Azure Blob Storage、およびオンプレやクラウドでホストするS3互換ストレージに対応 |

次のセクションから、BYOB のセットアップ手順をご案内します。

## バケットの用意 {#provision-your-bucket}

[利用可能なストレージ]({{< relref "#availability-matrix" >}}) を確認したら、アクセス権や CORS 設定を含むストレージバケットを用意します。続きはタブで選択してください。

{{< tabpane text=true >}}
{{% tab header="CoreWeave" value="coreweave" %}}
<a id="coreweave-requirements"></a>**必要条件**:
- **専用クラウド** または **自己管理型** v0.70.0 以降、または **マルチテナントクラウド**
- バケット・APIアクセスキー・シークレットキー作成権限付きの CoreWeave アカウント（AI Object Storage 有効）
- W&B インスタンスから CoreWeave ネットワークへの接続が必要

詳細は [CoreWeave公式ドキュメント](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/create-bucket) をご覧ください。

1. **マルチテナントクラウド**：バケットポリシーに必要なオーガナイゼーションIDを取得
    1. [W&B App](https://wandb.ai/) にログイン
    1. 左ナビゲーションで **Create a new team** をクリック
    1. 開いたドロワー上部の **Invite team members** の上に表示された組織IDをコピー
    1. このページは開いたまま、後ほど [W&B の設定]({{< relref "#configure-byob" >}}) で使用
1. CoreWeave で好みの名前・アベイラビリティゾーンでバケットを作成。必要に応じて W&B 用サブパスも作成。バケット名、ゾーン、APIキー、シークレットキー、サブパスを記録
1. バケットに次の CORS ポリシーを設定
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
    CoreWeave ストレージは S3互換です。CORS の詳細は [AWSドキュメント](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enabling-cors-examples.html) をご確認ください。
1. **マルチテナントクラウド**：W&B デプロイがバケットへアクセス＆[事前署名URL]({{< relref "./presigned-urls.md" >}})生成に必要な権限を付与するポリシーをバケットに設定。[バケットポリシーリファレンス](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy) 参照

    `<cw-bucket>` はCoreWeaveバケット名、`<wb-org-id>` は上記で取得したW&BオーガナイゼーションIDに置き換えます。

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

`"Sid": "AllowUsersInOrg"` 以降の記述は、W&B 組織内ユーザーの直接バケットアクセスを許可します。不要な場合、この部分は削除可能です。

{{% /tab %}}
{{% tab header="AWS" value="aws" %}}
詳細は [S3 バケット作成方法](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) を参照してください。

1. KMS キーを用意

    W&B では S3 バケットへのデータ暗号化・復号のために KMS キーが必要です。キータイプは `ENCRYPT_DECRYPT` にし、以下のポリシーを割り当てます。

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

    `<Your_Account_Id>` および `<aws_kms_key.key.arn>` に自分の値を指定してください。

    [マルチテナントクラウド]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) または [専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}})利用時、`<aws_principal_and_role_arn>` には以下のいずれかを使います：

    * [マルチテナントクラウド]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}})：`arn:aws:iam::725579432336:role/WandbIntegration`
    * [専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}})：`arn:aws:iam::830241207209:root`

    この設定により AWS アカウントはキーのフルアクセス権を持ち、W&B プラットフォームが必要な権限も付与されます。KMS Key ARN をメモしてください。

1. S3 バケットを用意

    次の手順でS3バケットを用意します。

    1. 任意の名前でS3バケットを作成。必要に応じてW&B用のサブパスも作成。
    1. サーバー側暗号化を有効化し、上記で作成したKMSキーを指定
    1. 下記ポリシーで CORS を設定：

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
        {{% alert %}}バケット内データが [オブジェクトライフサイクル管理ポリシー](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html) により削除された場合、一部runの履歴が読めなくなることがあります。{{% /alert %}}
    1. W&B プラットフォーム用AWSアカウントへ、下記S3権限を付与してください（バケットへ[事前署名URL]({{< relref "./presigned-urls.md" >}})生成などで利用）

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

        `<wandb_bucket>` にご自身のバケット名を、また `<aws_principal_and_role_arn>` には以下のいずれかを指定します。

        * [マルチテナントクラウド]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}})：`arn:aws:iam::725579432336:role/WandbIntegration`
        * [専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}})：`arn:aws:iam::830241207209:root`
  
詳しくは [AWS自己管理型ガイド]({{< relref "/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" >}}) をご覧ください。

{{% /tab %}}
{{% tab header="GCP" value="gcp"%}}
詳細は [GCP 公式ドキュメント](https://cloud.google.com/storage/docs/creating-buckets) を参照。

1. GCS バケットを用意

    GCP プロジェクトで GCS バケットを用意します：

    1. 任意の名前で GCS バケットを作成。必要に応じてサブパスも作成
    1. 暗号化タイプを `Google-managed` に設定
    1. `gsutil` で CORS ポリシーを設定。UI では設定できません

       1. ローカルで `cors-policy.json` ファイルを作成
       1. 下記ポリシー内容をコピー&保存

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

          {{% alert %}}バケットの [オブジェクトライフサイクル管理ポリシー](https://cloud.google.com/storage/docs/lifecycle) でファイルが削除される場合、一部runの履歴が読めなくなることがあります。{{% /alert %}}

      1. `<bucket_name>`を自分のバケット名に置き換えて、`gsutil` を実行

          ```bash
          gsutil cors set cors-policy.json gs://<bucket_name>
          ```

      1. ポリシー設定を確認。`<bucket_name>` を修正の上実行
        
          ```bash
          gsutil cors get gs://<bucket_name>
          ```

1. [マルチテナントクラウド]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}})または[専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}})を使う場合は、W&Bプラットフォームに紐付くGCPサービスアカウントへ `storage.admin` ロール を付与します（バケットのCORS 設定/バージョン管理等の確認用権限）。この権限がないと HTTP 403 エラーになります。

    * [マルチテナントクラウド]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}})：`wandb-integration@wandb-production.iam.gserviceaccount.com`
    * [専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}})：`deploy@wandb-production.iam.gserviceaccount.com`

    バケット名を控え、[W&B のBYOB設定]({{< relref "#configure-byob" >}}) へ進みます。
{{% /tab %}}

{{% tab header="Azure" value="azure" %}}
詳細は [Azure公式ドキュメント](https://learn.microsoft.com/en-us/azure/storage/blobs/blob-containers-portal) を参照してください。

1. Azure Blob Storage コンテナ作成

    インスタンスレベル BYOB で [こちらのTerraformモジュール](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) を利用しない場合、以下手順で Azure Blob Storage バケットを用意：

    1. 任意のバケット名で作成。必要に応じてサブパスも作成
    1. CORS ポリシーをバケットに設定

        UI 設定の場合は、Blob ストレージ > `Settings/Resource Sharing (CORS)` で以下を入力：

        | パラメータ         | 値        |
        | ---               | ---      |
        | Allowed Origins   | `*`      |
        | Allowed Methods   | `GET`, `HEAD`, `PUT` |
        | Allowed Headers   | `*`      |
        | Exposed Headers   | `*`      |
        | Max Age           | `3000`   |

        {{% alert %}}[オブジェクトライフサイクル管理ポリシー](https://learn.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-policy-configure?tabs=azure-portal) によりデータが削除された場合、一部runの履歴が読めなくなることがあります。{{% /alert %}}

1. ストレージアカウントアクセスキーとストレージアカウント名を取得。[専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) では安全な方法で W&B チームと共有

    チームレベル BYOB では、[Terraform モジュール](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector) でバケットと必要な権限・アクセス管理を自動化するのがおすすめです。[専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}})利用時は OIDC Issuer URL も必要。次の情報を控えてください：

    * ストレージアカウント名
    * ストレージコンテナ名
    * マネージドIDクライアントID
    * AzureテナントID

{{% /tab %}}
{{% tab header="S3-compatible" value="s3-compatible" %}}
S3互換バケットを作成し、以下を控えてください：
- アクセスキー
- シークレットアクセスキー
- URL エンドポイント
- バケット名
- 必要ならフォルダーのパス
- リージョン

{{% /tab %}}
{{< /tabpane >}}

次に、[ストレージアドレスを決定]({{< relref "#determine-the-storage-address" >}}) します。

## ストレージアドレスの決定  {#determine-the-storage-address}

このセクションでは、W&B Team を BYOB バケットへ接続するアドレス構文を説明します。例の中の山括弧（`<>`）部分はご自身の情報へ置き換えてください。詳細はタブで選択してください。

{{< tabpane text=true >}}
{{% tab header="CoreWeave" value="coreweave" %}}
このセクションは **専用クラウド** または **自己管理型** でのチームレベル BYOB 用です。インスタンスレベルやマルチテナントクラウドでは直接 [W&B の設定]({{< relref "#configure-byob" >}}) へ進んでください。

**バケット形式**:
```none
cw://<accessKey>:<secretAccessKey>@cwobject.com/<bucketName>?tls=true
```

  `cwobject.com` のHTTPSエンドポイントが利用可能です。TLS1.3必須。他のCoreWeaveエンドポイントご希望の場合は[support](mailto:support@wandb.com)までご連絡ください。
{{% /tab %}}
{{% tab header="AWS" value="aws" %}}
**バケット形式**:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>?region=<region>
```
`region` パラメータは、W&Bインスタンス・ストレージバケット両方がAWS上にあり `AWS_REGION` と一致する場合以外は必須です。
{{% /tab %}}
{{% tab header="GCP" value="gcp" %}}
**バケット形式**:
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```
{{% /tab %}}
{{% tab header="Azure" value="azure" %}}
**バケット形式**:
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```
{{% /tab %}}
{{% tab header="S3-compatible" value="s3-compatible" %}}
**バケット形式**:
```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>&tls=true
```
ここでも `region` パラメータは必須です。

{{% alert %}}
このタブは S3 以外の場所にホストされている S3互換ストレージ向け（例：[MinIO](https://github.com/minio/minio) をオンプレで運用する等）。AWS S3 の場合は **AWS** タブを参照してください。

クラウドネイティブのストレージバケットで S3互換モードが任意の場合、可能な限り固有のプロトコル指定子を利用してください。例えばCoreWeave バケットであれば `cw://` を推奨します。
{{% /alert %}}
{{% /tab %}}
{{< /tabpane >}}

ストレージアドレスが決定したら、[チームレベル BYOB の設定]({{< relref "#configure-team-level-byob" >}}) へ進みます。

## W&B の設定 {#configure-byob}

[バケットの用意]({{< relref "#provision-your-bucket" >}}) と [バケットアドレスの決定](#determine-the-storage-address) ができたら、[インスタンスレベル]({{< relref "#instance-level-byob" >}}) または [チームレベル]({{< relref "#team-level-byob" >}}) の BYOB 設定に進みます。

{{% alert color="secondary" %}}
ストレージバケットの構成を慎重に検討してください。W&B用にバケットを設定後、別のバケットへのデータ移行は非常に困難で、W&B側のサポートが必要になります（専用クラウド・自己管理型・マルチテナントクラウドのチームストレージすべて対象）。ご相談は [support](mailto:support@wandb.com) までご連絡ください。
{{% /alert %}}

### インスタンスレベル BYOB

{{% alert %}}
CoreWeave AI Object Storageのインスタンスレベル BYOBの場合は、この手順ではなく [W&B サポート](mailto:support@wandb.com) へご連絡ください。 サービス化された設定は未対応です。
{{% /alert %}}

**専用クラウド** の場合：バケット情報を W&B チームへ共有し、専用クラウドインスタンスの設定を依頼します。

**自己管理型** の場合、W&B App でインスタンスレベル BYOB を設定できます：

1. `admin` 権限のユーザーでW&Bにログイン
1. 画面上部のユーザーアイコンをクリックし **System Console** を開く
1. **Settings** > **System Connections** へ移動
1. **Bucket Storage** セクションで **Identity** 項目の権限を新バケットへ付与
1. **Provider** を選択
1. **Bucket Name** を入力
1. 必要に応じて **Path** も指定
1. **Save** をクリック

{{% alert %}}
自己管理型の場合、W&B管理のTerraformモジュールでストレージバケット＋必要な権限自動設定が推奨です：

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure：[インスタンスレベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) または [チームレベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
{{% /alert %}}

### チームレベル BYOB

[バケットの場所の決定](#determine-the-storage-address) ができたら、W&B App でチーム作成時に team level BYOB を設定します。

{{% alert %}}
- チーム作成後はストレージを変更できません
- インスタンスレベル BYOB については [インスタンスレベル BYOB]({{< relref "#instance-level-byob" >}}) を参照
- チームに CoreWeave ストレージを使う場合は、バケットがCoreWeave側で正しく構成されているか/S設定が正しいか[サポート](mailto:support@wandb.com)へご連絡ください（作成後修正不可のため）
{{% /alert %}}

デプロイメントタイプを選択して手順に進みます。

{{< tabpane text=true >}}
{{% tab header="Dedicated Cloud / Self-Hosted" value="dedicated" %}}

1. **専用クラウド**: バケットパスをアカウントチームへ必ず連絡し、インスタンスのサポート対象ファイルストアに追加してもらってから次以降の手順を行う
1. **自己管理型**: `GORILLA_SUPPORTED_FILE_STORES` 環境変数へバケットパスを追加しW&B再起動後、以下の操作を行う
1. `admin` 権限のユーザーでW&Bにログインし、左上アイコンからナビゲーションを開く → **Create a team to collaborate** をクリック
1. チーム名を入力
1. **Storage Type** を **External storage** に設定

    {{% alert %}}インスタンスレベルストレージ（内部・外部問わず）をチームストレージとして使う場合、**Storage Type** を **Internal** のままにします。チームごとの外部ストレージ使用時のみ、**External** にしてバケット情報を次ステップで入力してください。{{% /alert %}}

1. **Bucket location** をクリック
1. 既存バケットを使う場合はリストから選択。新規追加はリスト下の **Add bucket** をクリックし、バケット情報を入力。

    **Cloud provider** で **CoreWeave**, **AWS**, **GCP**, **Azure** から選択
    
    Cloud provider が表示されない場合、手順1でサポート対象ファイルストア追加が完了しているか確認し、それでも表示されなければ [サポート](mailto:support@wandb.ai) までご相談ください。

1. バケット詳細を入力
    - **CoreWeave** はバケット名のみ入力
    - Amazon S3、GCP、S3互換の場合は[前項で決定](#determine-the-storage-address)した完全バケットパス
    - Azureの場合は (Dedicated/Self-Managed 限定) **Account name** にアカウント名、**Container name** にBlobコンテナ名
    - 必要に応じて：
      - サブパスがあれば **Path** を指定
      - **AWS**: **KMS key ARN** にKMS暗号化キーのARN
      - **Azure**: 該当時 **Tenant ID** および **Managed Identity Client ID**
1. **Create team** をクリック

エラーまたは不正な設定がW&B側で検出された場合、画面下に警告やエラーが表示されます。問題が無ければチームが作成されます。

{{% /tab %}}
{{% tab header="Multi-tenant Cloud" value="multi-tenant" %}}

1. 以前新チーム作成開始時のブラウザウィンドウでオーガナイゼーションIDを確認。もしくは `admin` 権限ユーザーでログインし、左上アイコンからナビゲーションを開く → **Create a team to collaborate** をクリック
1. チーム名を入力
1. **Storage Type** を **External storage** に設定
1. **Bucket location** をクリック
1. 既存バケットはリストから選択。新規ならリスト下部の **Add bucket** をクリックしてバケット情報を入力

    **Cloud provider** で **CoreWeave**, **AWS**, **GCP**, **Azure** から選択
1. バケット詳細を入力
    - **CoreWeave** はバケット名のみ入力
    - Amazon S3、GCP、S3互換の場合は[前項で決定](#determine-the-storage-address)した完全バケットパス
    - Azureの場合（DedicatedまたはSelf-Managed）**Account name** にアカウント名、**Container name** にBlobコンテナ名
    - 必要に応じて：
      - サブパスがあれば **Path**
      - **AWS**: **KMS key ARN**
      - **Azure**: **Tenant ID**、**Managed Identity Client ID**
     - **Invite team members** で追加する場合はカンマ区切りのメールアドレスリストを入力（あとから追加も可能）
1. **Create team** をクリック

エラーまたは不正な設定時は画面下に警告表示。問題なければ作成完了です。

{{% /tab %}}
{{< /tabpane >}}

## トラブルシューティング

<details open>
<summary>CoreWeave AI Object Storage への接続</summary>

- **接続エラー**
  - W&B インスタンスから CoreWeave ネットワークエンドポイントへ接続できるか確認してください。
  - CoreWeave はバケット名をパスのサブドメイン（virtual-hosted style）として扱います（例：`cw://bucket-name.cwobject.com` が正、 ~`cw://cwobject.com/bucket-name/`~ は誤）。
  - バケット名にアンダースコア（`_`）やDNS互換性のない記号は使えません。
  - バケット名はCoreWeave全体で一意である必要があります。
  - `cw-` または `vip-` で始まるバケット名は予約済みです。
- **CORS認証失敗**
  - CORSポリシーが必須です。CoreWeaveはS3互換・詳細は [AWS CORS構成](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enabling-cors-examples.html) 参照。
  - `AllowedMethods` に `GET` `PUT` `HEAD` を含めてください。
  - `ExposeHeaders` には `ETag` を含めてください。
  - W&B UIドメインが `AllowedOrigins` で許可されている必要があります（このページの例では `*` を推奨）。
- **LOTA エンドポイント問題**
  - W&BからLOTAエンドポイントへ直接は現在非対応です。ご要望は [support](mailto:support@wandb.com) まで。
- **アクセスキー・権限エラー**
  - CoreWeave APIアクセスキーが有効か確認
  - APIアクセスキー/シークレットキーには `GetObject`, `PutObject`, `DeleteObject`, `ListBucket` 権限が必要です（このページの例で網羅）。[Create and Manage Access Keys](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys) も参照

</details>
