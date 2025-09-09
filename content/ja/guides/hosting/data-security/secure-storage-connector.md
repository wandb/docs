---
title: 自分の バケット を持ち込む (BYOB)
menu:
  default:
    identifier: ja-guides-hosting-data-security-secure-storage-connector
    parent: data-security
weight: 1
---

## 概要
BYOB（Bring your own bucket）は、W&B の Artifacts やその他の機微なデータを、あなた自身の クラウド または オンプレミス のインフラストラクチャーに保存できるようにする機能です。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) を利用する場合、あなたのバケットに保存したデータは W&B 管理のインフラストラクチャーには複製されません。

{{% alert %}}
* W&B の SDK / CLI / UI とあなたのバケット間の通信は、[事前署名 URL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を用いて行われます。
* W&W は Artifacts を削除するためにガーベジコレクション プロセスを使用します。詳細は、[Artifacts の削除]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}}) を参照してください。
* バケットの設定時にサブパスを指定することで、W&B がバケットのルート フォルダーにファイルを保存しないようにできます。これは、組織のバケット ガバナンスポリシーにより良く準拠するのに役立ちます。
{{% /alert %}}

### 中央データベースに保存されるデータとバケットに保存されるデータ
BYOB 機能を使用する場合、ある種類のデータは W&B の中央データベースに、その他の種類のデータはあなたのバケットに保存されます。

#### データベース
- Users、Teams、Artifacts、Experiments、Projects のメタデータ
- Reports
- Experiment ログ
- システム メトリクス
- コンソール ログ

#### バケット
- Experiment のファイルおよびメトリクス
- Artifact ファイル
- メディア ファイル
- Run ファイル
- Parquet 形式のエクスポート済み履歴メトリクスとシステムイベント

### バケットのスコープ
ストレージ バケットは次のいずれかのスコープで設定できます。

| Scope          | 説明 |
|----------------|------|
| Instance level | [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) と [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) では、組織またはインスタンス内で必要な権限を持つ任意のユーザーが、インスタンスのストレージ バケットに保存されたファイルにアクセスできます。[Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) には該当しません。 |
| Team level     | W&B の Team を Team レベルのストレージ バケットを使うよう設定すると、その Team のメンバーがそこに保存されたファイルへアクセスできます。Team レベルのストレージは、機密性の高いデータや厳格なコンプライアンス要件を持つ Team に対し、より強力なデータ アクセス制御とデータ分離を提供します。<br><br>Team レベルのストレージは、同一インスタンスを共有する事業部門や部門が、インフラや管理リソースを効率的に活用するのに役立ちます。また、別々の顧客案件のために個別のプロジェクト Team が AI ワークフローを管理できるようにもなります。すべてのデプロイメント形態で利用可能です。Team レベルの BYOB は、Team のセットアップ時に設定します。 |

この柔軟な設計により、組織のニーズに応じたいろいろなストレージ トポロジーを構成できます。例えば:
- 同じバケットをインスタンスと 1 つ以上の Team で共用できます。
- 各 Team は別々のバケットを使用でき、一部の Team はインスタンスのバケットへ書き込むことも、サブパスを使って複数の Team が 1 つのバケットを共有することもできます。
- Team ごとに異なる クラウド インフラ環境やリージョンにバケットをホストし、異なるストレージ管理チームが運用することもできます。

例として、あなたの組織に Kappa という Team があるとします。組織（および Team Kappa）はデフォルトでインスタンス レベルのストレージ バケットを使用しています。次に、Omega という Team を作成します。Team Omega を作成する際、その Team に対して Team レベルのストレージ バケットを設定します。Team Omega によって生成されたファイルは Team Kappa からはアクセスできません。一方、Team Kappa によって作成されたファイルは Team Omega からアクセス可能です。Team Kappa のデータも分離したい場合は、Kappa に対しても Team レベルのストレージ バケットを設定する必要があります。

### 可用性マトリクス
W&B は次のストレージ プロバイダーに接続できます:
- [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage) は、AI ワークロード向けに最適化された高性能な S3 互換オブジェクト ストレージ サービスです。
- [Amazon S3](https://aws.amazon.com/s3/) は、業界最高水準のスケーラビリティ、データ可用性、セキュリティ、パフォーマンスを提供するオブジェクト ストレージ サービスです。
- [Google Cloud Storage](https://cloud.google.com/storage) は、非構造化データを大規模に保存するためのマネージド サービスです。
- [Azure Blob Storage](https://azure.microsoft.com/products/storage/blobs) は、テキスト、バイナリ データ、画像、動画、ログなど大量の非構造化データを保存するための クラウド ベースのオブジェクト ストレージ ソリューションです。
- [MinIO](https://github.com/minio/minio) などの S3 互換ストレージ（自社の クラウド またはオンプレミス インフラにホスト）。

次の表は、W&B の各デプロイメント形態における各スコープでの BYOB の可用性を示します。

| W&B deployment type        | Instance level   | Team level | 追加情報 |
|----------------------------|------------------|------------|----------|
| Dedicated Cloud            | &check;          | &check;    | Instance と Team レベルの BYOB は、CoreWeave AI Object Storage、Amazon S3、GCP Storage、Microsoft Azure Blob Storage、そして自社の クラウド またはオンプレミス インフラにホストされた [MinIO](https://github.com/minio/minio) のような S3 互換ストレージでサポートされています。 |
| Multi-tenant Cloud         | Not Applicable   | &check;    | Team レベルの BYOB は、CoreWeave AI Object Storage、Amazon S3、GCP Storage、Microsoft Azure Blob Storage でサポートされています。 |
| Self-Managed               | &check;          | &check;    | Instance と Team レベルの BYOB は、CoreWeave AI Object Storage、Amazon S3、GCP Storage、Microsoft Azure Blob Storage、そして自社の クラウド またはオンプレミス インフラにホストされた [MinIO](https://github.com/minio/minio) のような S3 互換ストレージでサポートされています。 |

以下のセクションでは、BYOB のセットアップ手順を説明します。

## バケットをプロビジョニングする {#provision-your-bucket}

[可用性の確認]({{< relref path="#availability-matrix" lang="ja" >}}) が済んだら、アクセス ポリシーや CORS を含むストレージ バケットをプロビジョニングします。続行するタブを選択してください。

{{< tabpane text=true >}}
{{% tab header="CoreWeave" value="coreweave" %}}
<a id="coreweave-requirements"></a>**要件**:
- **Multi-tenant Cloud**、または
- **Dedicated Cloud** v0.73.0 以上、または
- v0.33.14+ の Helm チャートでデプロイされた **Self-Managed** v0.73.0 以上
- AI Object Storage が有効化され、バケット、API アクセス キー、シークレット キーを作成する権限を持つ CoreWeave アカウント
- あなたの W&B インスタンスが CoreWeave のネットワーク エンドポイントへ接続できること

詳細は CoreWeave のドキュメントの [Create a CoreWeave AI Object Storage bucket](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/create-bucket) を参照してください。

1. <a id="coreweave-org-id"></a>**Multi-tenant Cloud**: バケット ポリシーに必要な組織 ID を取得します。
    1. [W&B App](https://wandb.ai/) にログインします。
    1. 左側のナビゲーションで **Create a new team** をクリックします。
    1. 開いたドロワーで、**Invite team members** の上に表示されている W&B の組織 ID をコピーします。
    1. このページは開いたままにしておきます。後で [W&B を設定]({{< relref path="#configure-byob" lang="ja" >}}) する際に使用します。
1. <a id="coreweave-customer-namespace"></a>**Dedicated Cloud** / **Self-Managed**: バケット ポリシーに必要なカスタマー 名前空間を取得します。
    1. W&B App でユーザー プロフィール アイコンをクリックし、**System Console** をクリックします。
    1. **Authentication** タブをクリックします。
    1. ページ下部で **Customer Namespace** の値をコピーします。バケット ポリシーの設定に使用します。
    1. System Console を閉じても構いません。
1. CoreWeave で、希望する CoreWeave のアベイラビリティゾーンに、任意の名前でバケットを作成します。必要に応じて、W&B がすべてのファイルで使用するサブパス用のフォルダーを作成します。バケット名、アベイラビリティゾーン、API アクセス キー、シークレット キー、サブパスを控えておきます。
1. バケットに対して次の CORS（オリジン間リソース共有）ポリシーを設定します:
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
    CoreWeave ストレージは S3 互換です。CORS の詳細は、AWS ドキュメントの [Configuring cross-origin resource sharing (CORS)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enabling-cors-examples.html) を参照してください。

1. あなたの W&B デプロイメントがバケットへアクセスし、クラウド インフラ内の AI ワークロードやユーザーのブラウザがバケットへアクセスするために利用する [事前署名 URL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を生成できるように、必要な権限を付与するバケット ポリシーを設定します。CoreWeave のドキュメントの [Bucket Policy Reference](https://docs.coreweave.com/docs/products/storage/object-storage/reference/bucket-policy) を参照してください。

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
          "CW": "arn:aws:iam::wandb:static/<wb-cw-principal>"
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

    `"Sid": "AllowUsersInOrg"` で始まる節は、あなたの組織内のユーザーにバケットへの直接アクセスを許可します。この機能が不要な場合は、ポリシーからこの節を省略できます。
1. バケット ポリシー内のプレースホルダーを置き換えます:
    - `<cw-bucket>`: あなたのバケット名
    - `<cw-wandb-principal>`:
      - **Multi-tenant Cloud**: `arn:aws:iam::wandb:static/wandb-integration-public`
      - **Dedicated Cloud** または **Self-Managed**: `arn:aws:iam::wandb:static/wandb-integration`
    - `<wb-org-id>`:
      - **Multi-tenant Cloud**: [ステップ 1]({{< relref path="#coreweave-org-id" lang="ja" >}}) の組織 ID
      - **Dedicated Cloud** または **Self-Managed**: [ステップ 2]({{< relref path="#coreweave-customer-namespace" lang="ja" >}}) のカスタマー 名前空間
1. **Dedicated Cloud**: 追加の手順完了のために [support](mailto:support@wandb.ai) へ連絡してください。
1. **Self-Managed**: あなたの W&B デプロイメントで環境変数 `GORILLA_SUPPORTED_FILE_STORES` を厳密に `cw://` に設定し、W&B を再起動してください。これを行わないと、Team ストレージの設定時に CoreWeave が選択肢として表示されません。

続いて、[W&B を設定]({{< relref path="#configure-byob" lang="ja" >}}) します。

{{% /tab %}}
{{% tab header="AWS" value="aws" %}}
詳細は AWS ドキュメントの [Create an S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) を参照してください。
1. KMS キーをプロビジョニングします。

    W&B では、S3 バケット上のデータを暗号化・復号するために KMS キーのプロビジョニングが必要です。キーの使用タイプは `ENCRYPT_DECRYPT` でなければなりません。以下のポリシーをキーに割り当てます:

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

    `<Your_Account_Id>` と `<aws_kms_key.key.arn>` を適切に置き換えてください。

    [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、`<aws_principal_and_role_arn>` を次の値に置き換えます:

    * [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
    * [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`

    このポリシーは、あなたの AWS アカウントにキーへのフル アクセスを付与し、さらに W&B プラットフォームをホストしている AWS アカウントに必要な権限を割り当てます。KMS キーの ARN を控えておいてください。

1. S3 バケットをプロビジョニングします。

    あなたの AWS アカウントで S3 バケットをプロビジョニングするには、次の手順に従います:

    1. 任意の名前で S3 バケットを作成します。必要に応じて、すべての W&B ファイルを格納するサブパスとして設定できるフォルダーを作成します。
    1. サーバーサイド暗号化を有効化し、前のステップで作成した KMS キーを使用します。
    1. 次のポリシーで CORS を設定します:

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
        {{% alert %}}バケット内のデータが [オブジェクト ライフサイクル管理ポリシー](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html) により有効期限切れになると、一部の run の履歴が読み取れなくなる可能性があります。{{% /alert %}}
    1. W&B プラットフォームをホストしている AWS アカウントに必要な S3 権限を付与します。この権限は、クラウド インフラ内の AI ワークロードやユーザーのブラウザがバケットへアクセスするために利用する [事前署名 URL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を生成するために必要です。

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

        `<wandb_bucket>` を適切に置き換え、バケット名を控えてください。続いて、[W&B を設定]({{< relref path="#configure-byob" lang="ja" >}}) します。

        [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、`<aws_principal_and_role_arn>` を次の値に置き換えます。

        * [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
        * [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`
  
詳細は、[AWS self-managed hosting ガイド]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}}) を参照してください。

{{% /tab %}}
{{% tab header="GCP" value="gcp"%}}
詳細は GCP ドキュメントの [Create a bucket](https://cloud.google.com/storage/docs/creating-buckets) を参照してください。
1. GCS バケットをプロビジョニングします。

    あなたの GCP プロジェクトで GCS バケットをプロビジョニングするには、次の手順に従います:

    1. 任意の名前で GCS バケットを作成します。必要に応じて、すべての W&B ファイルを格納するサブパスとして設定できるフォルダーを作成します。
    1. 暗号化タイプを `Google-managed` に設定します。
    1. `gsutil` で CORS ポリシーを設定します。これは UI では行えません。

       1. ローカルに `cors-policy.json` というファイルを作成します。
       1. 次の CORS ポリシーをファイルにコピーして保存します。

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

          {{% alert %}}バケット内のデータが [オブジェクト ライフサイクル管理ポリシー](https://cloud.google.com/storage/docs/lifecycle) により有効期限切れになると、一部の run の履歴が読み取れなくなる可能性があります。{{% /alert %}}

      1. `<bucket_name>` を正しいバケット名に置き換え、`gsutil` を実行します。

          ```bash
          gsutil cors set cors-policy.json gs://<bucket_name>
          ```

      1. バケットのポリシーを検証します。`<bucket_name>` を正しいバケット名に置き換えます。
        
          ```bash
          gsutil cors get gs://<bucket_name>
          ```

1. [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、W&B プラットフォームに紐づく GCP サービス アカウントへ `storage.admin` ロールを付与してください。W&B は、このロールを用いてバケットの CORS 設定やオブジェクトのバージョン管理の有効化有無といった属性を確認します。サービス アカウントに `storage.admin` ロールがない場合、これらの確認は HTTP 403 エラーになります。

    * [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合: `wandb-integration@wandb-production.iam.gserviceaccount.com`
    * [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合: `deploy@wandb-production.iam.gserviceaccount.com`

    バケット名を控えておいてください。続いて、[BYOB 用に W&B を設定]({{< relref path="#configure-byob" lang="ja" >}}) します。
{{% /tab %}}

{{% tab header="Azure" value="azure" %}}
詳細は Azure ドキュメントの [Create a blob storage container](https://learn.microsoft.com/en-us/azure/storage/blobs/blob-containers-portal) を参照してください。

**Instance レベルの BYOB**:

1. Azure Blob Storage コンテナーをプロビジョニングします。

    [この Terraform モジュール](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) を使わない場合は、以下の手順であなたの Azure サブスクリプションに Azure Blob Storage コンテナーをプロビジョニングします:

    1. 任意の名前でコンテナーを作成します。必要に応じて、すべての W&B ファイルを格納するサブパスとして設定できるフォルダーを作成します。
    1. コンテナーに CORS ポリシーを設定します。

        UI で CORS ポリシーを設定するには、対象の Blob Storage に移動し、`Settings/Resource Sharing (CORS)` までスクロールして、次のように設定します:

        | パラメータ | 値 |
        | --- | --- |
        | Allowed Origins | `*`  |
        | Allowed Methods | `GET`, `HEAD`, `PUT` |
        | Allowed Headers | `*` |
        | Exposed Headers | `*` |
        | Max Age | `3000` |

        {{% alert %}}バケット内のデータが [オブジェクト ライフサイクル管理ポリシー](https://learn.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-policy-configure?tabs=azure-portal) により有効期限切れになると、一部の run の履歴が読み取れなくなる可能性があります。{{% /alert %}}

1. ストレージ アカウントのアクセス キーを生成し、その名前とストレージ アカウント名を控えてください。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、ストレージ アカウント名とアクセス キーを安全な共有手段で W&B チームに共有してください。

**Team レベルの BYOB**:

W&B は、Azure Blob Storage コンテナーと必要なアクセス機構および権限をまとめてプロビジョニングするために [Terraform](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/secure-storage-connector) の使用を推奨します。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用する場合は、あなたのインスタンスの OIDC issuer URL を提供してください。以下の情報を控えておきます:

* ストレージ アカウント名
* ストレージ コンテナー名
* Managed identity client id
* Azure tenant id

{{% /tab %}}
{{% tab header="S3-compatible" value="s3-compatible" %}}
S3 互換のバケットを作成します。以下を控えておきます:
- アクセス キー
- シークレット アクセス キー
- URL エンドポイント
- バケット名
- フォルダー パス（該当する場合）
- リージョン

{{% /tab %}}
{{< /tabpane >}}

次に、[ストレージ アドレスを決定]({{< relref path="#determine-the-storage-address" lang="ja" >}}) します。

## ストレージ アドレスを決定する  {#determine-the-storage-address}
このセクションでは、W&B の Team を BYOB のストレージ バケットに接続する際の記法を説明します。例では、山括弧（`<>`）内のプレースホルダー値を、あなたのバケットの情報に置き換えてください。
詳細手順はタブを選択してください。

{{< tabpane text=true >}}
{{% tab header="CoreWeave" value="coreweave" %}}
このセクションは、**Dedicated Cloud** または **Self-Managed** における Team レベルの BYOB にのみ該当します。インスタンス レベルの BYOB または Multi-tenant Cloud をお使いの場合は、すでに [W&B の設定]({{< relref path="#configure-byob" lang="ja" >}}) の準備ができています。

以下の形式でバケットのフル パスを決定します。山括弧（`<>`）内のプレースホルダーはバケットの値に置き換えてください。

**バケット形式**:
```none
cw://<accessKey>:<secretAccessKey>@cwobject.com/<bucketName>?tls=true
```

  `cwobject.com` の HTTPS エンドポイントがサポートされています。TLS 1.3 が必須です。その他の CoreWeave エンドポイントに関心がある場合は [support](mailto:support@wandb.com) までご連絡ください。
{{% /tab %}}
{{% tab header="AWS" value="aws" %}}
**バケット形式**:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>?region=<region>
```
アドレス内の `region` パラメータは、あなたの W&B インスタンスとストレージ バケットがともに AWS にデプロイされ、かつ W&B インスタンスの `AWS_REGION` がその S3 バケットのリージョンと一致している場合を除き、必須です。
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
アドレス内の `region` パラメータは必須です。

{{% alert %}}
このセクションは、オンプレミスでホストした [MinIO](https://github.com/minio/minio) など、S3 以外にホストされた S3 互換ストレージ バケット向けです。AWS S3 にホストされたストレージ バケットの場合は、代わりに **AWS** タブを参照してください。

オプションとして S3 互換モードを持つ Cloud ネイティブ ストレージ バケットでは、可能な限り Cloud ネイティブのプロトコル指定子を使用してください。例えば CoreWeave のバケットには `s3://` ではなく `cw://` を使用します。
{{% /alert %}}
{{% /tab %}}
{{< /tabpane >}}

ストレージ アドレスを決定したら、[Team レベルの BYOB を設定]({{< relref path="#configure-team-level-byob" lang="ja" >}}) する準備が整いました。

## W&B を設定する  {#configure-byob}
[バケットをプロビジョニング]({{< relref path="#provision-your-bucket" lang="ja" >}}) し、[アドレスを決定](#determine-the-storage-address) したら、[インスタンス レベル]({{< relref path="#instance-level-byob" lang="ja" >}}) または [Team レベル]({{< relref path="#team-level-byob" lang="ja" >}}) の BYOB を設定できます。

{{% alert color="secondary" %}}
ストレージ バケットのレイアウトは慎重に計画してください。W&B 用のストレージ バケットを設定した後に、そのデータを別のバケットへ移行するのは複雑で、W&B の支援が必要です。これは Dedicated Cloud と Self-Managed のストレージ、ならびに Multi-tenant Cloud の Team レベル ストレージに該当します。ご不明点は [support](mailto:support@wandb.com) までご連絡ください。
{{% /alert %}}

### インスタンス レベルの BYOB

{{% alert %}}
インスタンス レベルでの CoreWeave AI Object Storage については、自己設定手順はまだサポートされていません。代わりに [W&B support](mailto:support@wandb.com) にお問い合わせください。
{{% /alert %}}

**Dedicated Cloud** の場合: バケットの詳細を W&B チームに共有し、専用クラウドのインスタンス側で設定してもらいます。

**Self-Managed** の場合: W&B App からインスタンス レベルの BYOB を設定できます。
1. `admin` ロールのユーザーで W&B にログインします。
1. 右上のユーザー アイコンをクリックし、**System Console** をクリックします。
1. **Settings** > **System Connections** に移動します。
1. **Bucket Storage** セクションで、**Identity** フィールドの ID に新しいバケットへのアクセス権が付与されていることを確認します。
1. **Provider** を選択します。
1. **Bucket Name** を入力します。
1. 必要に応じて、新しいバケットで使用する **Path** を入力します。
1. **Save** をクリックします。

{{% alert %}}
Self-Managed の場合、W&B が管理する Terraform モジュールを使用して、ストレージ バケットと、それに必要なアクセス機構や関連する IAM 権限をまとめてプロビジョニングすることを推奨します:

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [Instance レベルの BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) または [Team レベルの BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/secure-storage-connector)
{{% /alert %}}

### Team レベルの BYOB

W&B App で Team の作成時に、Team レベルの BYOB を設定できます。次の 2 つの選択肢があります:
- **Use an existing bucket**: 先にバケットの [ストレージ ロケーションを決定](#determine-the-storage-address) しておく必要があります。
- **Create a new bucket**（Multi-tenant Cloud のみ）: Team の作成時に、W&B があなたのクラウド プロバイダーにバケットを自動作成できます。CoreWeave、AWS、GCP、Azure をサポートします。

{{% alert %}}
- Team を作成した後は、そのストレージを変更できません。
- インスタンス レベルの BYOB については、[インスタンス レベルの BYOB]({{< relref path="#instance-level-byob" lang="ja" >}}) を参照してください。
- Team で CoreWeave ストレージを設定する予定がある場合は、[CoreWeave の要件](#coreweave-requirements) を確認し、Team 作成後はストレージ詳細を変更できないため、CoreWeave でのバケット設定が正しいことの確認および Team の設定の検証のために、[support](mailto:support@wandb.com) へご連絡ください。
{{% /alert %}}

あなたのデプロイメント形態を選択して続行してください。

{{< tabpane text=true >}}
{{% tab header="Dedicated Cloud / Self-Hosted" value="dedicated" %}}

1. **Dedicated Cloud**: Team でこのストレージ バケットを使う手順の前に、インスタンスの supported file stores に追加できるよう、バケット パスをアカウント チームへ必ず提供してください。
1. **Self-Managed**: Team でこのストレージ バケットを使う手順の前に、バケット パスを `GORILLA_SUPPORTED_FILE_STORES` 環境変数に必ず追加し、W&B を再起動してください。
1. `admin` ロールのユーザーで W&B にログインし、左上のアイコンをクリックして左ナビゲーションを開き、**Create a team to collaborate** をクリックします。
1. Team 名を入力します。
1. **Storage Type** を **External storage** に設定します。

    {{% alert %}}Team のストレージとしてインスタンス レベルのストレージを利用する（内部・外部を問わず）場合は、インスタンス レベルのバケットが BYOB で構成されていても、**Storage Type** は **Internal** のままにしてください。Team で別の外部ストレージを使用する場合は、**Storage Type** を **External** に設定し、次のステップでバケット詳細を構成します。{{% /alert %}}

1. **Bucket location** をクリックします。
1. 既存のバケットを使用する場合は、リストから選択します。新しいバケットを追加するには、下部の **Add bucket** をクリックし、バケットの詳細を入力します。

    **Cloud provider** をクリックし、**CoreWeave**、**AWS**、**GCP**、**Azure** のいずれかを選択します。
    
    クラウド プロバイダーがリストに表示されない場合は、手順 1 に従って、インスタンスの supported file stores にバケット パスを追加したか確認してください。それでもストレージ プロバイダーが表示されない場合は、[support](mailto:support@wandb.ai) へお問い合わせください。
1. バケットの詳細を指定します。
    - **CoreWeave** の場合は、バケット名のみを入力します。
    - Amazon S3、GCP、または S3 互換ストレージの場合は、先に [決定した](#determine-the-storage-address) フル バケット パスを入力します。
    - W&B の Dedicated または Self-Managed 上の Azure の場合、**Account name** に Azure アカウント、**Container name** に Azure Blob Storage のコンテナー名を入力します。
    - 必要に応じて、追加の接続設定を指定します:
      - 該当する場合は、**Path** にバケットのサブパスを設定します。
      - **CoreWeave**: 追加の接続設定は不要です。
      - **AWS**: 暗号化用に **KMS key ARN** に KMS 暗号化キーの ARN を設定します。
      - **GCP**: 追加の接続設定は不要です。
      - **Azure**: **Tenant ID** と **Managed Identity Client ID** の値を指定します。`GORILLA_SUPPORTED_FILE_STORES` で接続文字列を構成していない限り、これらのフィールドは必須です。
1. **Create team** をクリックします。

W&B がバケットへのアクセス エラーや無効な設定を検出した場合、ページ下部にエラーや警告が表示されます。問題がなければ Team が作成されます。

{{% /tab %}}
{{% tab header="Multi-tenant Cloud" value="multi-tenant" %}}

1. 先ほど新しい Team の作成を開始して組織 ID を確認したブラウザ ウィンドウに戻るか、`admin` ロールのユーザーで W&B にログインし、左上のアイコンをクリックして左ナビゲーションを開き、**Create a team to collaborate** をクリックします。
1. Team 名を入力します。
1. **Storage Type** を **External storage** に設定します。
1. **Bucket location** をクリックします。
1. 既存のバケットを使用する場合は、リストから選択します。
1. 新しいバケットを作成するには、下部の **Add bucket** をクリックし、次を行います:
    1. **Cloud provider** をクリックし、**CoreWeave**、**AWS**、**GCP**、**Azure** のいずれかを選択します。
    1. バケットの詳細を入力します:
        - **Name**: Azure 以外のすべてのプロバイダーではバケット名を入力します。Azure の場合はコンテナー名です。
        - **Path**（任意）: バケット内で使用するサブパスを入力します。
    1. 選択したクラウド プロバイダー向けの追加の接続設定を指定します:
        - CoreWeave: 追加の設定は不要です。
        - AWS: 暗号化のために **KMS key ARN** を任意で指定します。
        - GCP: 追加の設定は不要です。
        - Azure: 
            - **Account name**（必須）: Azure ストレージ アカウント名
            - **Tenant ID**（任意）: Azure Active Directory のテナント ID
            - **Managed Identity Client ID**（任意）: マネージド ID 認証のクライアント ID
    
    {{% alert %}}**Create team** をクリックすると、指定した構成で W&B があなたのクラウド プロバイダーにバケットを自動作成します。{{% /alert %}}

1. Team へメンバーを招待します。**Invite team members** に、カンマ区切りでメール アドレスのリストを指定します。あるいは、Team 作成後にメンバーを招待することもできます。
1. **Create team** をクリックします。

W&B がバケットへのアクセス エラーや無効な設定を検出した場合、ページ下部にエラーや警告が表示されます。問題がなければ Team が作成されます。

{{% /tab %}}
{{< /tabpane >}}

## トラブルシューティング
<details open>
<summary>CoreWeave AI Object Storage への接続</summary>

- **接続エラー**
  - あなたの W&B インスタンスが CoreWeave のネットワーク エンドポイントへ接続できることを確認してください。
  - CoreWeave は仮想ホスト形式のパスを使用します。つまり、バケット名はパスの先頭にサブドメインとして含まれます。例: `cw://bucket-name.cwobject.com` は正しく、`cw://cwobject.com/bucket-name/` は誤りです。
  - バケット名にアンダースコア（`_`）や DNS のルールに互換性のない文字を含めてはいけません。
  - バケット名は CoreWeave のロケーション間でグローバルに一意でなければなりません。
  - バケット名は予約済みプレフィックスである `cw-` または `vip-` で始めてはいけません。
- **CORS 検証の失敗**
  - CORS ポリシーは必須です。CoreWeave は S3 互換です。CORS の詳細は AWS ドキュメントの [Configuring cross-origin resource sharing (CORS)](https://docs.aws.amazon.com/AmazonS3/latest/userguide/enabling-cors-examples.html) を参照してください。
  - `AllowedMethods` には `GET`、`PUT`、`HEAD` を含める必要があります。
  - `ExposeHeaders` には `ETag.
  - W&B のフロントエンド ドメインは、CORS ポリシーの `AllowedOrigins` に含める必要があります。このページで提示している CORS ポリシーの例では、`*` によりすべてのドメインを許可しています。
- **LOTA エンドポイントに関する問題**
  - W&B から LOTA エンドポイントへの接続は、まだサポートされていません。関心がある場合は [support](mailto:support@wandb.com) までご連絡ください。
- **アクセス キーと権限エラー**
  - CoreWeave の API アクセス キーが期限切れになっていないことを確認してください。
  - CoreWeave の API アクセス キーとシークレット キーに、`GetObject`、`PutObject`、`DeleteObject`、`ListBucket` の十分な権限があることを確認してください。このページの例はこの要件を満たしています。CoreWeave ドキュメントの [Create and Manage Access Keys](https://docs.coreweave.com/docs/products/storage/object-storage/how-to/manage-access-keys) を参照してください。

</details>