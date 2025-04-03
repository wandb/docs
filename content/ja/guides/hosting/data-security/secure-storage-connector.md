---
title: Bring your own bucket (BYOB)
menu:
  default:
    identifier: ja-guides-hosting-data-security-secure-storage-connector
    parent: data-security
weight: 1
---

Bring your own bucket (BYOB) を使用すると、W&B の Artifacts やその他の関連する機密データを、お客様の クラウド または オンプレミス の インフラストラクチャー に保存できます。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、お客様の バケット に保存する データ は、W&B が管理する インフラストラクチャー にコピーされません。

{{% alert %}}
* W&B SDK / CLI / UI とお客様の バケット 間の通信は、[事前署名付き URL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を使用して行われます。
* W&B は、W&B Artifacts を削除するためにガベージコレクション プロセス を使用します。詳細については、[Artifacts の削除]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}}) を参照してください。
* バケット を構成する際にサブパスを指定して、W&B が バケット のルートにあるフォルダーにファイルを保存しないようにすることができます。これは、組織の バケット ガバナンス ポリシーへの準拠を向上させるのに役立ちます。
{{% /alert %}}

## 中央データベースに保存されるデータと バケット に保存されるデータ

BYOB 機能を使用する場合、特定の種類の データ は W&B 中央データベースに保存され、他の種類の データ はお客様の バケット に保存されます。

### データベース

- ユーザー 、 Teams、Artifacts、Experiments、および Projects の メタデータ
- Reports
- Experiment ログ
- システム メトリクス

## バケット

- Experiment ファイルと メトリクス
- Artifact ファイル
- メディア ファイル
- Run ファイル

## 設定オプション
ストレージ バケット を構成できる スコープ は、*インスタンス レベル* または *Team レベル* の 2 つです。

- インスタンス レベル: 組織内で関連する 権限 を持つ ユーザー は、インスタンス レベルのストレージ バケット に保存されているファイルに アクセス できます。
- Team レベル: W&B Team の メンバー は、Team レベルで構成された バケット に保存されているファイルに アクセス できます。Team レベルのストレージ バケット を使用すると、機密性の高い データ や厳格なコンプライアンス要件を持つ Teams に対して、より優れた データ アクセス制御と データ 分離が可能になります。

インスタンス レベルで バケット を構成することも、組織内の 1 つまたは複数の Teams に対して個別に構成することもできます。

たとえば、組織に Kappa という Team があるとします。組織 (および Team Kappa) は、デフォルトでインスタンス レベルのストレージ バケット を使用します。次に、Omega という Team を作成します。Team Omega を作成するときに、その Team の Team レベルのストレージ バケット を構成します。Team Omega によって生成されたファイルは、Team Kappa からは アクセス できません。ただし、Team Kappa によって作成されたファイルは、Team Omega から アクセス できます。Team Kappa の データを分離する場合は、Team レベルのストレージ バケット を構成する必要があります。

{{% alert %}}
Team レベルのストレージ バケット は、特に異なる事業部門や部署が 1 つの インスタンス を共有して インフラストラクチャー と管理リソースを効率的に利用する場合、[自己管理]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス にも同じメリットをもたらします。これは、個別の顧客エンゲージメントのために AI ワークフロー を管理する個別の プロジェクト Teams を持つ企業にも当てはまります。
{{% /alert %}}

## 可用性マトリックス
次の表は、さまざまな W&B サーバー デプロイメント タイプ における BYOB の可用性を示しています。`X` は、その機能が特定の デプロイメント タイプ で利用できることを意味します。

| W&B サーバー デプロイメント タイプ | インスタンス レベル | Team レベル | 追加情報 |
|----------------------------|--------------------|----------------|------------------------|
| 専用クラウド | X | X | インスタンス および Team レベルの BYOB は、Amazon Web Services、Google Cloud Platform、および Microsoft Azure で利用できます。Team レベルの BYOB の場合、同じ クラウド または別の クラウド 内の クラウド ネイティブ ストレージ バケット、または クラウド または オンプレミス の インフラストラクチャー でホストされている [MinIO](https://github.com/minio/minio) などの S3 互換のセキュア ストレージに接続できます。 |
| SaaS Cloud | 適用外 | X | Team レベルの BYOB は、Amazon Web Services と Google Cloud Platform でのみ利用できます。W&B は、Microsoft Azure のデフォルトのストレージ バケット と唯一のストレージ バケット を完全に管理します。 |
| 自己管理 | X | X | インスタンス レベルの BYOB は、インスタンス がお客様によって完全に管理されているため、デフォルトです。自己管理 インスタンス が クラウド にある場合、Team レベルの BYOB 用に、同じ クラウド または別の クラウド 内の クラウド ネイティブ ストレージ バケット に接続できます。インスタンス または Team レベルの BYOB のいずれかに対して、[MinIO](https://github.com/minio/minio) などの S3 互換のセキュア ストレージを使用することもできます。 |

{{% alert color="secondary" %}}
専用クラウド または 自己管理 インスタンス の インスタンス または Team レベルのストレージ バケット 、または SaaS Cloud アカウントの Team レベルのストレージ バケット を構成すると、これらの スコープ のストレージ バケット を変更または再構成することはできません。これには、別の バケット に データを移行したり、メイン プロダクト ストレージ内の関連する参照を再マップしたりすることも含まれます。W&B では、インスタンス または Team レベルの スコープ のいずれかを構成する前に、ストレージ バケット のレイアウトを慎重に計画することを推奨します。ご不明な点がございましたら、W&B Team までお問い合わせください。
{{% /alert %}}

## Team レベルの BYOB 向けのクロス クラウド または S3 互換ストレージ

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [自己管理]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス の Team レベルの BYOB 用に、別の クラウド 内の クラウド ネイティブ ストレージ バケット 、または [MinIO](https://github.com/minio/minio) などの S3 互換ストレージ バケット に接続できます。

クロス クラウド または S3 互換ストレージの使用を有効にするには、W&B インスタンス の `GORILLA_SUPPORTED_FILE_STORES` 環境 変数を使用して、次のいずれかの形式で、関連する アクセス キー を含むストレージ バケット を指定します。

<details>
<summary>専用クラウド または 自己管理 インスタンス で Team レベルの BYOB 用に S3 互換ストレージを構成する</summary>

次の形式でパスを指定します。
```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>?tls=true
```
`region` パラメータ は必須です。ただし、W&B インスタンス が AWS にあり、W&B インスタンス ノード で構成された `AWS_REGION` が S3 互換ストレージ用に構成された リージョン と一致する場合は除きます。

</details>
<details>
<summary>専用クラウド または 自己管理 インスタンス で Team レベルの BYOB 用にクロス クラウド ネイティブ ストレージを構成する</summary>

W&B インスタンス とストレージ バケット の場所固有の形式でパスを指定します。

GCP または Azure の W&B インスタンス から AWS の バケット へ:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>
```

GCP または AWS の W&B インスタンス から Azure の バケット へ:
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```

AWS または Azure の W&B インスタンス から GCP の バケット へ:
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```

</details>

{{% alert %}}
Team レベルの BYOB 用の S3 互換ストレージへの接続は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では利用できません。また、Team レベルの BYOB 用の AWS バケット への接続は、その インスタンス が GCP にあるため、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではクロス クラウド です。そのクロス クラウド 接続は、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) および [自己管理]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス で以前に概説した アクセス キー と環境 変数 ベースのメカニズムを使用しません。
{{% /alert %}}

詳細については、W&B サポート (support@wandb.com) までお問い合わせください。

## W&B プラットフォーム と同じ クラウド 内の クラウド ストレージ

ユースケース に基づいて、Team または インスタンス レベルでストレージ バケット を構成します。ストレージ バケット のプロビジョニングまたは構成方法は、Azure の アクセス メカニズムを除き、構成されているレベルに関係なく同じです。

{{% alert %}}
W&B では、必要な アクセス メカニズムと関連する IAM 権限 とともにストレージ バケット をプロビジョニングするために、W&B が管理する Terraform モジュール を使用することを推奨します。

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [インスタンス レベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) または [Team レベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="AWS" value="aws" %}}
1. KMS キー をプロビジョニングします。

    W&B では、S3 バケット 上の データを 暗号化および復号化するために、KMS キー をプロビジョニングする必要があります。キー の使用タイプは `ENCRYPT_DECRYPT` である必要があります。次の ポリシー を キー に割り当てます。

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

    `<Your_Account_Id>` と `<aws_kms_key.key.arn>` を適宜置き換えます。

    [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、`<aws_principal_and_role_arn>` を対応する 値 に置き換えます。

    * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
    * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`

    この ポリシー は、AWS アカウント に キー へのフル アクセス を許可し、W&B プラットフォーム をホストする AWS アカウント に必要な 権限 も割り当てます。KMS キー ARN の記録を保持します。

2. S3 バケット をプロビジョニングします。

    次の手順に従って、AWS アカウント で S3 バケット をプロビジョニングします。

    1. 任意の名前で S3 バケット を作成します。オプションで、すべての W&B ファイルを保存するためのサブパスとして構成できるフォルダーを作成します。
    2. バケット の バージョン管理 を有効にします。
    3. 前のステップの KMS キー を使用して、サーバー 側の 暗号化 を有効にします。
    4. 次の ポリシー で CORS を構成します。

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
                "MaxAgeSeconds": 3600
            }
        ]
        ```

    5. W&B プラットフォーム をホストする AWS アカウント に必要な S3 権限 を付与します。これには、お客様の クラウド インフラストラクチャー または ユーザー の ブラウザー 内の AI ワークロード が バケット への アクセス に使用する [事前署名付き URL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を生成するための 権限 が必要です。

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

        `<wandb_bucket>` を適宜置き換え、 バケット 名の記録を保持します。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、インスタンス レベルの BYOB の場合に備えて、 バケット 名を W&B Team と共有します。任意の デプロイメント タイプ の Team レベルの BYOB の場合は、[Team の作成中に バケット を構成します]({{< relref path="#configure-byob-in-wb" lang="ja" >}})。

        [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、`<aws_principal_and_role_arn>` を対応する 値 に置き換えます。

        * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
        * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`
  
  詳細については、[AWS 自己管理 ホスティング ガイド]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}}) を参照してください。
{{% /tab %}}

{{% tab header="GCP" value="gcp"%}}
1. GCS バケット をプロビジョニングします。

    次の手順に従って、GCP プロジェクト で GCS バケット をプロビジョニングします。

    1. 任意の名前で GCS バケット を作成します。オプションで、すべての W&B ファイルを保存するためのサブパスとして構成できるフォルダーを作成します。
    2. ソフト削除を有効にします。
    3. オブジェクト の バージョン管理 を有効にします。
    4. 暗号化 タイプ を `Google-managed` に設定します。
    5. `gsutil` で CORS ポリシー を設定します。これは UI では実行できません。

      1. `cors-policy.json` というファイルをローカルに作成します。
      2. 次の CORS ポリシー をファイルにコピーして保存します。

          ```json
          [
          {
            "origin": ["*"],
            "responseHeader": ["Content-Type"],
            "exposeHeaders": ["ETag"],
            "method": ["GET", "HEAD", "PUT"],
            "maxAgeSeconds": 3600
          }
          ]
          ```

      3. `<bucket_name>` を正しい バケット 名に置き換え、`gsutil` を実行します。

          ```bash
          gsutil cors set cors-policy.json gs://<bucket_name>
          ```

      4. バケット の ポリシー を確認します。`<bucket_name>` を正しい バケット 名に置き換えます。
        
          ```bash
          gsutil cors get gs://<bucket_name>
          ```

2. [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、W&B プラットフォーム にリンクされている GCP サービス アカウント に `Storage Admin` ロール を付与します。

    * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、アカウント は `wandb-integration@wandb-production.iam.gserviceaccount.com` です。
    * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合、アカウント は `deploy@wandb-production.iam.gserviceaccount.com` です。

    バケット 名の記録を保持します。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、インスタンス レベルの BYOB の場合に備えて、 バケット 名を W&B Team と共有します。任意の デプロイメント タイプ の Team レベルの BYOB の場合は、[Team の作成中に バケット を構成します]({{< relref path="#configure-byob-in-wb" lang="ja" >}})。
{{% /tab %}}

{{% tab header="Azure" value="azure"%}}
1. Azure Blob Storage をプロビジョニングします。

    インスタンス レベルの BYOB の場合、[この Terraform モジュール](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) を使用していない場合は、次の手順に従って Azure サブスクリプション で Azure Blob Storage バケット をプロビジョニングします。

    * 任意の名前で バケット を作成します。オプションで、すべての W&B ファイルを保存するためのサブパスとして構成できるフォルダーを作成します。
    * BLOB とコンテナー のソフト削除を有効にします。
    * バージョン管理 を有効にします。
    * バケット で CORS ポリシー を構成します。

      UI から CORS ポリシー を設定するには、BLOB ストレージに移動し、`[設定] -> [リソース共有 (CORS)]` までスクロールして、次のように設定します。

      | パラメータ | 値 |
      | --- | --- |
      | 許可される オリジン | `*` |
      | 許可される メソッド | `GET`、`HEAD`、`PUT` |
      | 許可される ヘッダー | `*` |
      | 公開される ヘッダー | `*` |
      | 最大年齢 | `3600` |

2. ストレージ アカウント の アクセス キー を生成し、ストレージ アカウント 名とともに記録を保持します。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、安全な共有メカニズムを使用して、ストレージ アカウント 名と アクセス キー を W&B Team と共有します。

    Team レベルの BYOB の場合、W&B では、必要な アクセス メカニズムと 権限 とともに Azure Blob Storage バケット をプロビジョニングするために、[Terraform](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector) を使用することを推奨します。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用する場合は、インスタンス の OIDC 発行者 URL を指定します。[Team の作成中に バケット を構成する]({{< relref path="#configure-byob-in-wb" lang="ja" >}}) ために必要な詳細をメモしておきます。

    * ストレージ アカウント 名
    * ストレージ コンテナー 名
    * マネージド ID クライアント ID
    * Azure テナント ID
{{% /tab %}}
{{< /tabpane >}}

## W&B で BYOB を構成する

{{< tabpane text=true >}}

{{% tab header="Team レベル" value="team" %}}
{{% alert %}}
[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [自己管理]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス で Team レベルの BYOB 用に、別の クラウド 内の クラウド ネイティブ ストレージ バケット 、または [MinIO](https://github.com/minio/minio) などの S3 互換ストレージ バケット に接続する場合は、[Team レベルの BYOB 向けのクロス クラウド または S3 互換ストレージ]({{< relref path="#cross-cloud-or-s3-compatible-storage-for-team-level-byob" lang="ja" >}}) を参照してください。そのような場合は、以下の手順を使用して Team 用に構成する前に、W&B インスタンス の `GORILLA_SUPPORTED_FILE_STORES` 環境 変数を使用してストレージ バケット を指定する必要があります。
{{% /alert %}}

{{% alert %}}
[セキュア ストレージ コネクターの動作を示す ビデオ](https://www.youtube.com/watch?v=uda6jIx6n5o) (9 分) をご覧ください。
{{% /alert %}}

W&B Team を作成するときに Team レベルでストレージ バケット を構成するには:

1. [Team 名] フィールドに Team の名前を入力します。
2. [ストレージ タイプ] オプションで [外部ストレージ] を選択します。
3. ドロップダウンから [新しい バケット ] を選択するか、既存の バケット を選択します。

    複数の W&B Teams が同じ クラウド ストレージ バケット を使用できます。これを有効にするには、ドロップダウンから既存の クラウド ストレージ バケット を選択します。

4. [クラウド プロバイダー] ドロップダウンから、 クラウド プロバイダーを選択します。
5. [名前] フィールドにストレージ バケット の名前を入力します。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または Azure 上の [自己管理]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) インスタンス がある場合は、[アカウント 名] フィールドと [コンテナー 名] フィールドに値を入力します。
6. (オプション) オプションの [パス] フィールドに バケット サブパスを入力します。W&B が バケット のルートにあるフォルダーにファイルを保存しないようにする場合は、これを行います。
7. (AWS バケット を使用している場合はオプション) [KMS キー ARN] フィールドに KMS 暗号化 キー の ARN を入力します。
8. (Azure バケット を使用している場合はオプション) [テナント ID] フィールドと [マネージド ID クライアント ID] フィールドに値を入力します。
9. ([SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではオプション) Team の作成時に Team メンバー を招待します。
10. [Team を作成] ボタンを押します。

{{< img src="/images/hosting/prod_setup_secure_storage.png" alt="" >}}

バケット への アクセス に問題がある場合、または バケット の 設定 が無効な場合、ページの下部に エラー または 警告 が表示されます。
{{% /tab %}}

{{% tab header="インスタンス レベル" value="instance"%}}
専用クラウド または 自己管理 インスタンス の インスタンス レベルの BYOB を構成するには、W&B サポート (support@wandb.com) までお問い合わせください。
{{% /tab %}}
{{< /tabpane >}}
