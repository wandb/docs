---
title: Bring your own bucket (BYOB)
menu:
  default:
    identifier: ja-guides-hosting-data-security-secure-storage-connector
    parent: data-security
weight: 1
---

Bring your own bucket (BYOB) を使用すると、W&B Artifacts およびその他の関連する機密データを、お客様自身のクラウドまたはオンプレミス のインフラストラクチャーに保存できます。[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、お客様の バケット に保存するデータは、W&B が管理するインフラストラクチャーにコピーされません。

{{% alert %}}
* W&B SDK / CLI / UI とお客様の バケット 間の通信は、[事前署名付き URL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を使用して行われます。
* W&B は、W&B Artifacts を削除するために、ガベージコレクション プロセス を使用します。詳細については、[Artifacts の削除]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}}) を参照してください。
* バケット を構成する際にサブパスを指定して、W&B が バケット のルートにあるフォルダーにファイルを保存しないようにすることができます。これは、組織の バケット ガバナンス ポリシーへの準拠を向上させるのに役立ちます。
{{% /alert %}}

## 中央データベースと バケット に保存されるデータ

BYOB 機能を使用する場合、特定の種類のデータは W&B 中央データベースに保存され、他の種類のデータはユーザー の バケット に保存されます。

### データベース

- ユーザー 、 Teams、Artifacts、Experiments、Projects の メタデータ
- Reports
- Experiment ログ
- システム メトリクス

## バケット

- Experiment ファイルと メトリクス
- Artifact ファイル
- メディア ファイル
- Run ファイル

## 構成オプション
ストレージ バケット の構成範囲は、*インスタンス レベル* または *Team レベル* の 2 つです。

- インスタンス レベル: 組織内で関連する権限を持つ ユーザー は、インスタンス レベルのストレージ バケット に保存されているファイルに アクセス できます。
- Team レベル: W&B Team のメンバーは、Team レベルで構成された バケット に保存されているファイルに アクセス できます。Team レベルのストレージ バケット により、非常に機密性の高いデータや厳格なコンプライアンス要件を持つチームに対して、より高度なデータ アクセス 制御とデータ分離が可能になります。

インスタンス レベルで バケット を構成したり、組織内の 1 つまたは複数のチームに対して個別に構成したりできます。

たとえば、組織に Kappa というチームがあるとします。組織 (および Team Kappa) は、デフォルトでインスタンス レベルのストレージ バケット を使用します。次に、Omega というチームを作成します。Team Omega を作成するときに、そのチームの Team レベルのストレージ バケット を構成します。Team Omega によって生成されたファイルは、Team Kappa は アクセス できません。ただし、Team Kappa によって作成されたファイルは、Team Omega は アクセス できます。Team Kappa のデータを分離する場合は、Team Kappa 用にも Team レベルのストレージ バケット を構成する必要があります。

{{% alert %}}
Team レベルのストレージ バケット は、特に異なる事業部門や部署がインフラストラクチャーと管理リソースを効率的に利用するためにインスタンスを共有する場合、[自己管理]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスに同じメリットをもたらします。これは、個別の顧客エンゲージメントのために AI ワークフローを管理する個別の プロジェクト チームを持つ企業にも当てはまります。
{{% /alert %}}

## 可用性マトリックス
次の表は、さまざまな W&B サーバー デプロイメント タイプにおける BYOB の可用性を示しています。`X` は、特定のデプロイメント タイプでその機能が利用可能であることを意味します。

| W&B サーバー デプロイメント タイプ | インスタンス レベル | Team レベル | 追加情報 |
|----------------------------|--------------------|----------------|------------------------|
| Dedicated cloud | X | X | インスタンスと Team レベルの両方の BYOB は、Amazon Web Services、Google Cloud Platform、および Microsoft Azure で使用できます。Team レベルの BYOB の場合、同じクラウドまたは別のクラウドのクラウドネイティブ ストレージ バケット、あるいはクラウドまたはオンプレミス のインフラストラクチャーでホストされている [MinIO](https://github.com/minio/minio) のような S3 互換のセキュア ストレージに接続できます。 |
| SaaS Cloud | 該当なし | X | Team レベルの BYOB は、Amazon Web Services と Google Cloud Platform でのみ使用できます。W&B は、Microsoft Azure のデフォルトおよび唯一のストレージ バケット を完全に管理します。 |
| Self-managed | X | X | インスタンス レベルの BYOB は、インスタンスがユーザー によって完全に管理されているため、デフォルトです。自己管理インスタンスがクラウドにある場合は、Team レベルの BYOB 用に、同じクラウドまたは別のクラウドのクラウドネイティブ ストレージ バケット に接続できます。インスタンス レベルまたは Team レベルの BYOB のいずれかに対して、[MinIO](https://github.com/minio/minio) のような S3 互換のセキュア ストレージを使用することもできます。 |

{{% alert color="secondary" %}}
Dedicated cloud または Self-managed インスタンスのインスタンスまたは Team レベルのストレージ バケット 、あるいは SaaS Cloud アカウントの Team レベルのストレージ バケット を構成すると、これらの範囲のストレージ バケット を変更または再構成することはできません。これには、別の バケット にデータを移行したり、メイン プロダクション ストレージ内の関連する参照を再マッピングしたりすることも含まれます。W&B では、インスタンスまたは Team レベルの範囲のいずれかを構成する前に、ストレージ バケット のレイアウトを慎重に計画することをお勧めします。ご不明な点がございましたら、W&B チームにお問い合わせください。
{{% /alert %}}

## Team レベルの BYOB のクロスクラウドまたは S3 互換ストレージ

[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスの Team レベルの BYOB 用に、別のクラウドのクラウドネイティブ ストレージ バケット 、または [MinIO](https://github.com/minio/minio) のような S3 互換ストレージ バケット に接続できます。

クロスクラウドまたは S3 互換ストレージの使用を有効にするには、W&B インスタンスの `GORILLA_SUPPORTED_FILE_STORES` 環境変数を使用して、次のいずれかの形式で、関連する アクセス キーを含むストレージ バケット を指定します。

<details>
<summary>Dedicated cloud または Self-managed インスタンスで Team レベルの BYOB 用に S3 互換ストレージを構成する</summary>

次の形式を使用してパスを指定します。
```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>?tls=true
```
W&B インスタンスが AWS にあり、W&B インスタンス ノードで構成された `AWS_REGION` が S3 互換ストレージ用に構成されたリージョンと一致する場合を除き、`region` パラメータは必須です。

</details>
<details>
<summary>Dedicated cloud または Self-managed インスタンスで Team レベルの BYOB 用にクロスクラウド ネイティブ ストレージを構成する</summary>

W&B インスタンスとストレージ バケット の場所に合わせて特定の形式でパスを指定します。

GCP または Azure の W&B インスタンスから AWS の バケット へ:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>
```

GCP または AWS の W&B インスタンスから Azure の バケット へ:
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```

AWS または Azure の W&B インスタンスから GCP の バケット へ:
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```

</details>

{{% alert %}}
Team レベルの BYOB の S3 互換ストレージへの接続は、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では使用できません。また、Team レベルの BYOB の AWS バケット への接続は、そのインスタンスが GCP にあるため、[SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではクロスクラウドです。そのクロスクラウド接続は、[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) および [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンス用に以前に概説した アクセス キーと環境変数に基づくメカニズムを使用しません。
{{% /alert %}}

詳細については、W&B サポート (support@wandb.com) までお問い合わせください。

## W&B プラットフォームと同じクラウド内のクラウド ストレージ

ユースケース に基づいて、Team またはインスタンス レベルでストレージ バケット を構成します。Azure の アクセス メカニズムを除き、ストレージ バケット のプロビジョニングまたは構成方法は、構成されるレベルに関係なく同じです。

{{% alert %}}
W&B では、W&B が管理する Terraform モジュールを使用して、必要な アクセス メカニズムと関連する IAM 権限とともにストレージ バケット をプロビジョニングすることをお勧めします。

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [インスタンス レベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) または [Team レベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="AWS" value="aws" %}}
1. KMS キーをプロビジョニングする

    W&B では、S3 バケット 上のデータを暗号化および復号化するために、KMS キーをプロビジョニングする必要があります。キーの使用タイプは `ENCRYPT_DECRYPT` である必要があります。次のポリシーをキーに割り当てます。

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

    `<Your_Account_Id>` および `<aws_kms_key.key.arn>` を適宜置き換えます。

    [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、`<aws_principal_and_role_arn>` を対応する値に置き換えます。

    * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
    * [Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`

    このポリシーは、AWS アカウントにキーへのフル アクセス権を付与し、W&B プラットフォームをホストする AWS アカウントに必要な権限も割り当てます。KMS キー ARN の記録を保管してください。

2. S3 バケット をプロビジョニングする

    次の手順に従って、AWS アカウントで S3 バケット をプロビジョニングします。

    1. 任意の名前で S3 バケット を作成します。オプションで、すべての W&B ファイルを保存するためのサブパスとして構成できるフォルダーを作成します。
    2. バケット の バージョン管理を有効にします。
    3. 前のステップの KMS キーを使用して、サーバー側の暗号化を有効にします。
    4. 次のポリシーで CORS を構成します。

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

    5. W&B プラットフォームをホストする AWS アカウントに必要な S3 権限を付与します。これには、クラウド インフラストラクチャーまたは ユーザー ブラウザー内の AI ワークロードが バケット への アクセス に使用する [事前署名付き URL]({{< relref path="./presigned-urls.md" lang="ja" >}}) を生成するための権限が必要です。

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

        `<wandb_bucket>` を適宜置き換え、 バケット 名の記録を保管してください。[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、インスタンス レベルの BYOB の場合に W&B チームと バケット 名を共有します。任意のデプロイメント タイプで Team レベルの BYOB の場合は、[チームの作成中に バケット を構成します]({{< relref path="#configure-byob-in-wb" lang="ja" >}})。

        [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、`<aws_principal_and_role_arn>` を対応する値に置き換えます。

        * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
        * [Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`
  
  詳細については、[AWS 自己管理ホスティング ガイド]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}}) を参照してください。
{{% /tab %}}

{{% tab header="GCP" value="gcp"%}}
1. GCS バケット をプロビジョニングする

    次の手順に従って、GCP プロジェクトで GCS バケット をプロビジョニングします。

    1. 任意の名前で GCS バケット を作成します。オプションで、すべての W&B ファイルを保存するためのサブパスとして構成できるフォルダーを作成します。
    2. ソフト削除を有効にします。
    3. オブジェクト の バージョン管理を有効にします。
    4. 暗号化タイプを「Google 管理」に設定します。
    5. `gsutil` で CORS ポリシーを設定します。これは UI では実行できません。

      1. ローカルに `cors-policy.json` というファイルを作成します。
      2. 次の CORS ポリシーをファイルにコピーして保存します。

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

      4. バケット のポリシーを確認します。`<bucket_name>` を正しい バケット 名に置き換えます。
        
          ```bash
          gsutil cors get gs://<bucket_name>
          ```

2. [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、W&B プラットフォームにリンクされている GCP サービス アカウントに `Storage Admin` ロールを付与します。

    * [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、アカウントは `wandb-integration@wandb-production.iam.gserviceaccount.com` です
    * [Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合、アカウントは `deploy@wandb-production.iam.gserviceaccount.com` です

    バケット 名の記録を保管してください。[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、インスタンス レベルの BYOB の場合に W&B チームと バケット 名を共有します。任意のデプロイメント タイプで Team レベルの BYOB の場合は、[チームの作成中に バケット を構成します]({{< relref path="#configure-byob-in-wb" lang="ja" >}})。
{{% /tab %}}

{{% tab header="Azure" value="azure"%}}
1. Azure Blob Storage をプロビジョニングする

    インスタンス レベルの BYOB の場合、[この Terraform モジュール](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) を使用していない場合は、次の手順に従って、Azure サブスクリプションで Azure Blob Storage バケット をプロビジョニングします。

    * 任意の名前で バケット を作成します。オプションで、すべての W&B ファイルを保存するためのサブパスとして構成できるフォルダーを作成します。
    * BLOB とコンテナーのソフト削除を有効にします。
    * バージョン管理を有効にします。
    * バケット で CORS ポリシーを構成します

      UI から CORS ポリシーを設定するには、BLOB ストレージに移動し、下にスクロールして「設定/リソース共有 (CORS)」に移動し、次の設定を行います。

      | パラメータ | 値 |
      | --- | --- |
      | 許可されるオリジン | `*`  |
      | 許可されるメソッド | `GET` 、`HEAD` 、`PUT` |
      | 許可されるヘッダー | `*` |
      | 公開されるヘッダー | `*` |
      | 最大経過時間 | `3600` |

2. ストレージ アカウント アクセス キーを生成し、ストレージ アカウント名とともに記録しておきます。[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、セキュアな共有メカニズムを使用して、ストレージ アカウント名と アクセス キーを W&B チームと共有します。

    Team レベルの BYOB の場合、W&B では、[Terraform](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector) を使用して、必要な アクセス メカニズムと権限とともに Azure Blob Storage バケット をプロビジョニングすることをお勧めします。[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用する場合は、インスタンスの OIDC 発行者 URL を指定します。[チームの作成中に バケット を構成する]({{< relref path="#configure-byob-in-wb" lang="ja" >}}) ために必要な詳細をメモしておきます。

    * ストレージ アカウント名
    * ストレージ コンテナー名
    * 管理対象 ID クライアント ID
    * Azure テナント ID
{{% /tab %}}
{{< /tabpane >}}

## W&B で BYOB を構成する

{{< tabpane text=true >}}

{{% tab header="Team レベル" value="team" %}}
{{% alert %}}
[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスの Team レベルの BYOB 用に、別のクラウドのクラウドネイティブ ストレージ バケット 、または [MinIO](https://github.com/minio/minio) のような S3 互換ストレージ バケット に接続する場合は、[Team レベルの BYOB のクロスクラウドまたは S3 互換ストレージ]({{< relref path="#cross-cloud-or-s3-compatible-storage-for-team-level-byob" lang="ja" >}}) を参照してください。このような場合、以下の手順を使用してチーム用に構成する前に、W&B インスタンスの `GORILLA_SUPPORTED_FILE_STORES` 環境変数を使用してストレージ バケット を指定する必要があります。
{{% /alert %}}

W&B Team を作成するときに Team レベルでストレージ バケット を構成するには:

1. **チーム名**フィールドにチームの名前を入力します。
2. **ストレージ タイプ**オプションで**外部ストレージ**を選択します。
3. ドロップダウンから**新しい バケット **を選択するか、既存の バケット を選択します。

    複数の W&B Team が同じクラウド ストレージ バケット を使用できます。これを有効にするには、ドロップダウンから既存のクラウド ストレージ バケット を選択します。

4. **クラウド プロバイダー**ドロップダウンから、クラウド プロバイダーを選択します。
5. **名前**フィールドにストレージ バケット の名前を入力します。[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または Azure 上の [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスがある場合は、**アカウント名**フィールドと**コンテナー名**フィールドに値を入力します。
6. (オプション)**パス**フィールドに バケット サブパスを入力します。W&B が バケット のルートにあるフォルダーにファイルを保存しないようにする場合は、これを行います。
7. (AWS バケット を使用している場合はオプション)**KMS キー ARN**フィールドに KMS 暗号化キーの ARN を入力します。
8. (Azure バケット を使用している場合はオプション)**テナント ID**フィールドと**管理対象 ID クライアント ID**フィールドに値を入力します。
9. ([SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) ではオプション) オプションで、チームを作成するときにチームメンバーを招待します。
10. **チームの作成**ボタンを押します。

{{< img src="/images/hosting/prod_setup_secure_storage.png" alt="" >}}

バケット への アクセス に問題がある場合、または バケット の設定が無効な場合は、ページの下部にエラーまたは警告が表示されます。
{{% /tab %}}

{{% tab header="インスタンス レベル" value="instance"%}}
Dedicated cloud または Self-managed インスタンスのインスタンス レベルの BYOB を構成するには、W&B サポート (support@wandb.com) までお問い合わせください。
{{% /tab %}}
{{< /tabpane >}}
