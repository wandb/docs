---
title: 独自の バケット を持ち込む (BYOB)
menu:
  default:
    identifier: ja-guides-hosting-data-security-secure-storage-connector
    parent: data-security
weight: 1
---

自分のバケットを持ち込む (BYOB) を使用することで、W&B Artifacts やその他の機密性の高いデータを、自分のクラウドやオンプレミスのインフラストラクチャーに保存することができます。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})または[SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})の場合、バケットに保存したデータは W&B が管理するインフラストラクチャーにコピーされません。

{{% alert %}}
* W&B SDK / CLI / UI とあなたのバケットとの通信は、[事前署名URL]({{< relref path="./presigned-urls.md" lang="ja" >}})を使用して行われます。
* W&B はガベージコレクションプロセスを使用して W&B Artifacts を削除します。詳細は[Artifacts の削除]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})をご覧ください。
* バケットを設定する際にサブパスを指定することができ、これにより W&B がバケットのルートフォルダにファイルを保存しないようにできます。これにより、組織のバケットガバナンスポリシーによりよく適合するのに役立つかもしれません。
{{% /alert %}}

## 中央データベースとバケットに保存されるデータ

BYOB 機能を使用すると、データの一部は W&B の中央データベースに保存され、その他のデータは自分のバケットに保存されます。

### データベース

- ユーザー、チーム、Artifacts、Experiments、および Projects のメタデータ
- Reports
- 実験ログ
- システムメトリクス

## バケット

- 実験ファイルとメトリクス
- Artifact ファイル
- メディアファイル
- Run ファイル

## 設定オプション
ストレージバケットを設定するには、*インスタンスレベル*または*チームレベル*の二つのスコープがあります。

- インスタンスレベル: 組織内で関連権限を持つユーザーがインスタンスレベルのストレージバケットに保存されたファイルにアクセスできます。
- チームレベル: W&B チームのメンバーは、チームレベルで設定されたバケットに保存されたファイルにアクセスできます。チームレベルのストレージバケットは、機密性の高いデータや厳しいコンプライアンス要件を持つチームに対してより厳格なデータ アクセス制御とデータの分離を提供します。

バケットをインスタンスレベルで設定し、組織内の1つ以上のチームでそれぞれ設定することができます。

たとえば、組織内に Kappa というチームがあるとします。あなたの組織 (およびチーム Kappa) は、デフォルトでインスタンスレベルのストレージバケットを使用します。次に Omega というチームを作成します。チーム Omega を作成する際に、そのチームのためのチームレベルのストレージバケットを設定することができます。チーム Omega によって生成されたファイルはチーム Kappa によってアクセスすることはできません。ただし、チーム Kappa によって作成されたファイルはチーム Omega によってアクセスすることができます。チーム Kappa のデータを分離したい場合は、彼らのためにチームレベルのストレージバケットを設定する必要があります。

{{% alert %}}
チームレベルのストレージバケットは、特に異なる事業部門や部門がインスタンスを共有してインフラストラクチャーや管理資源を効果的に利用するため、[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスにも同じ恩恵を提供します。これは、別々のプロジェクトチームが別々の顧客向けの AI ワークフローを管理する企業にも適用されます。
{{% /alert %}}

## 利用可能性マトリックス
以下の表は、異なる W&B サーバーデプロイメントタイプでの BYOB の利用可能性を示しています。 `X`は、そのデプロイメントタイプで機能が利用可能であることを示します。

| W&B サーバーデプロイメントタイプ | インスタンスレベル | チームレベル | 追加情報 |
|----------------------------|--------------------|----------------|------------------------|
| 専用クラウド | X | X | Amazon Web Services、Google Cloud Platform、Microsoft Azure のいずれでもインスタンスとチームレベルの BYOB が利用可能です。チームレベルの BYOB については、クラウドネイティブのストレージバケットに同じクラウドまたは別のクラウド、またはあなたのクラウドやオンプレミスのインフラストラクチャーにホストされた [MinIO](https://github.com/minio/minio) のような S3 互換の安全なストレージにも接続できます。 |
| SaaS クラウド | 該当なし | X | チームレベルの BYOB は Amazon Web Services および Google Cloud Platform のみで利用可能です。W&B は Microsoft Azure のデフォルトで唯一のストレージバケットを完全に管理しています。 |
| セルフマネージド | X | X | インスタンスレベルの BYOB がデフォルトです。なぜなら、インスタンスは完全にあなたによって管理が行われるためです。もしあなたのセルフマネージドインスタンスがクラウドにある場合、チームレベルの BYOB のために同じまたは別のクラウドにクラウドネイティブのストレージバケットに接続できます。また、インスタンスまたはチームレベルの BYOB のいずれかに [MinIO](https://github.com/minio/minio) のような S3 互換の安全なストレージを使用することもできます。|

{{% alert color="secondary" %}}
専用クラウドまたはセルフマネージドインスタンスの場合、インスタンスレベルまたはチームレベルのストレージバケットを一度設定すると、そのスコープのストレージバケットを変更したり再設定したりすることはできません。それにはデータを別のバケットに移行したり、主な製品ストレージの関連する参照をリマップしたりすることができないことも含まれます。W&B では、インスタンスおよびチームレベルのいずれかのスコープのストレージバケットを設定する前に、ストレージバケットのレイアウトを注意深く計画することをお勧めします。質問がある場合は、W&B チームにご連絡ください。
{{% /alert %}}

## チームレベル BYOB のためのクロスクラウドまたは S3 互換ストレージ

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスにおいて、別のクラウドのクラウドネイティブのストレージバケットまたは [MinIO](https://github.com/minio/minio) のような S3 互換のストレージバケットに接続して、チームレベル BYOB を利用することができます。

クロスクラウドまたは S3 互換のストレージの利用を有効にするには、次のフォーマットのいずれかを使用してストレージバケットを指定し、W&B インスタンス用の`GORILLA_SUPPORTED_FILE_STORES` 環境変数を設定します。

<details>
<summary>専用クラウドまたはセルフマネージドインスタンスでチームレベル BYOB のために S3 互換のストレージを設定する</summary>

次のフォーマットを使用してパスを指定します：

```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>?tls=true
```
`region`パラメータは必須です。ただし、W&B インスタンスが AWS 内にあり、W&B インスタンスノードで設定されている`AWS_REGION` が S3 互換ストレージ用に設定されたリージョンと一致している場合を除きます。

</details>
<details>
<summary>専用クラウドまたはセルフマネージドインスタンスでチームレベル BYOB のためにクロスクラウドネイティブストレージを設定する</summary>

W&B インスタンスとストレージバケットの場所に特有のフォーマットでパスを指定します：

GCP または Azure にある W&B インスタンスから AWS にあるバケットへ：
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>
```

GCP または AWS にある W&B インスタンスから Azure にあるバケットへ：
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```

AWS または Azure にある W&B インスタンスから GCP にあるバケットへ：
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```

</details>

{{% alert %}}
チームレベル BYOB 用の S3 互換ストレージへの接続は [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では利用できません。また、[SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、チームレベル BYOB 用の AWS バケットへの接続はクロスクラウドで行われます。これは、そのインスタンスが GCP に存在するためです。そのクロスクラウドの接続は、以前説明した専用クラウドやセルフマネージドインスタンスのようにアクセキーおよび環境変数ベースのメカニズムを使用しません。
{{% /alert %}}

詳しくは support@wandb.com までお問い合わせください。

## W&B プラットフォームと同じクラウドのクラウドストレージ

ユースケースに基づいて、チームまたはインスタンスレベルにストレージバケットを設定します。ストレージバケットが調達または設定される方法は、それがどのレベルで設定されているかにかかわらず同じです。ただし、Azure のアクセスメカニズムに不一致があります。 

{{% alert %}}
W&B は、ストレージバケットをプロビジョニングするために W&B が管理する Terraform モジュールを使用して、必要なアクセスメカニズムと関連する IAM 権限と共に使用することをお勧めします：

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [インスタンスレベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) または [チームレベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="AWS" value="aws" %}}
1. KMS キーのプロビジョニング

    W&B では、S3 バケット上のデータを暗号化および復号化するために KMS キーをプロビジョニングする必要があります。キーの使用タイプは `ENCRYPT_DECRYPT` である必要があります。キーに次のポリシーを割り当てます：

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

    `<Your_Account_Id>` および `<aws_kms_key.key.arn>` を適宜置き換えてください。

    [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、`<aws_principal_and_role_arn>` を次の値で置き換えてください：

    * [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合: `arn:aws:iam::725579432336:role/WandbIntegration`
    * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合: `arn:aws:iam::830241207209:root`

    このポリシーは、AWS アカウントに対してキーへの完全なアクセス権を与え、W&B プラットフォームをホスティングする AWS アカウントに必要な権限も割り当てます。KMS キーの ARN を記録してください。

2. S3 バケットのプロビジョニング

    AWS アカウントに S3 バケットをプロビジョニングするために、以下の手順に従ってください：

    1. お好みの名前で S3 バケットを作成します。必要に応じてフォルダーを作成し、サブパスとして設定して W&B のすべてのファイルを保存します。
    2. バケットのバージョニングを有効にします。
    3. 前のステップで生成した KMS キーを使用して、サーバー側の暗号化を有効にします。
    4. 以下のポリシーで CORS を設定します：

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

    5. W&B プラットフォームのホスティング先の AWS アカウントに必要な S3 権限を付与します。これにより、AI ワークロードがクラウドインフラストラクチャーやユーザーブラウザーでバケットにアクセスするための[事前署名された URL]({{< relref path="./presigned-urls.md" lang="ja" >}})が生成されます。

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

        `<wandb_bucket>` を適宜置き換え、バケット名を記録してください。もし専用クラウドを使用している場合、インスタンスレベルの BYOB の場合は W&B チームとバケット名を共有してください。どのデプロイメントタイプにおいてもチームレベルの BYOB の場合、[チーム作成時にバケットを設定してください]({{< relref path="#configure-byob-in-wb" lang="ja" >}})。

        [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合、`<aws_principal_and_role_arn>` を次の値で置き換えてください。

        * [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}): `arn:aws:iam::725579432336:role/WandbIntegration`
        * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}): `arn:aws:iam::830241207209:root`
  
  詳細については、[AWS セルフマネージドホスティングガイド]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}})をご覧ください。
{{% /tab %}}

{{% tab header="GCP" value="gcp"%}}
1. GCS バケットのプロビジョニング

    GCP プロジェクト内で GCS バケットをプロビジョニングするために、以下の手順に従います：

    1. 好みの名前で GCS バケットを作成します。必要に応じてフォルダーを作成し、サブパスとして設定して W&B のすべてのファイルを保存します。
    2. ソフト削除を有効にします。
    3. オブジェクトのバージョニングを有効にします。
    4. 暗号化タイプを `Google-managed` に設定します。
    5. UI では不可能なので、`gsutil` を用いて CORS ポリシーを設定します。

      1. `cors-policy.json` という名前のファイルをローカルに作成します。
      2. ファイルに以下の CORS ポリシーをコピーして保存します。

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

      3. `<bucket_name>` を正しいバケット名で置き換えて `gsutil` を実行します。

          ```bash
          gsutil cors set cors-policy.json gs://<bucket_name>
          ```

      4. バケットのポリシーを確認します。`<bucket_name>` を正しいバケット名で置き換えます。
        
          ```bash
          gsutil cors get gs://<bucket_name>
          ```

2. [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合、W&B プラットフォームにリンクされた GCP サービスアカウントに `Storage Admin` ロールを付与します：

    * [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合、アカウントは: `wandb-integration@wandb-production.iam.gserviceaccount.com`
    * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合、アカウントは: `deploy@wandb-production.iam.gserviceaccount.com`

    バケット名を記録してください。専用クラウドを使用している場合、インスタンスレベルの BYOB ではバケット名を W&B チームと共有してください。どのデプロイメントタイプにおいてもチームレベルの BYOB の場合、[チーム作成時にバケットを設定してください]({{< relref path="#configure-byob-in-wb" lang="ja" >}})。
{{% /tab %}}

{{% tab header="Azure" value="azure"%}}
1. Azure Blob Storage のプロビジョニング

    インスタンスレベルの BYOB の場合、[この Terraform モジュール](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob)を使用していない場合は、以下のステップに従って Azure サブスクリプション内で Azure Blob Storage バケットをプロビジョニングします：

    * 好みの名前でバケットを作成します。必要に応じてフォルダーを作成し、サブパスとして設定して W&B のすべてのファイルを保存します。
    * ブロブとコンテナのソフト削除を有効にします。
    * バージョニングを有効にします。
    * バケットに CORS ポリシーを設定します。

      CORS ポリシーを UI を通じて設定するには、 Blob ストレージに移動し、`Settings/Resource Sharing (CORS)` までスクロールダウンして、以下を設定します：

      | パラメータ | 値 |
      | --- | --- |
      | Allowed Origins | `*`  |
      | Allowed Methods | `GET`, `HEAD`, `PUT` |
      | Allowed Headers | `*` |
      | Exposed Headers | `*` |
      | Max Age | `3600` |

2. ストレージ アカウント アクセス キーを生成し、ストレージ アカウント名と共に記録します。専用クラウドを使用している場合、ストレージ アカウント名とアクセス キーを安全な共有方法で W&B チームと共有します。

    チームレベルの BYOB の場合、W&B は[Terraform](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)を使用して、必要なアクセスメカニズムと権限と共に Azure Blob Storage バケットをプロビジョニングすることをお勧めします。専用クラウドを使用する場合、インスタンスのための OIDC 発行者 URL を提供します。バケットを設定する際に必要な詳細をメモしてください： 

    * ストレージ アカウント名
    * ストレージ コンテナ名
    * マネージドアイデンティティ クライアント ID
    * Azure テナント ID
{{% /tab %}}
{{< /tabpane >}}

## W&B での BYOB の設定

{{< tabpane text=true >}}

{{% tab header="チームレベル" value="team" %}}
{{% alert %}}
別のクラウドのクラウドネイティブのストレージバケットに接続している場合、または[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスでチームレベル BYOB のために [MinIO](https://github.com/minio/minio) のような S3 互換のストレージバケットに接続している場合、[チームレベル BYOB のためのクロスクラウドまたは S3 互換ストレージ]({{< relref path="#cross-cloud-or-s3-compatible-storage-for-team-level-byob" lang="ja" >}}) を参照してください。そのような場合、W&B インスタンス用の `GORILLA_SUPPORTED_FILE_STORES` 環境変数を使用してストレージバケットを指定し、それをチーム用に設定する前に以下の手順に従ってください。
{{% /alert %}}

{{% alert %}}
[セキュアストレージコネクターが動作しているビデオデモンストレーション](https://www.youtube.com/watch?v=uda6jIx6n5o) (9 分) を視聴してください。
{{% /alert %}}

W&B チーム作成時にチームレベルでストレージバケットを設定するには：

1. **チーム名** フィールドにチーム名を入力します。 
2. **ストレージタイプ** オプションとして **外部ストレージ** を選択します。
3. ドロップダウンから **新しいバケット** を選択するか、既存のバケットを選択します。

    複数の W&B チームが同じクラウドストレージバケットを使用できます。これを有効にするには、ドロップダウンから既存のクラウドストレージバケットを選択してください。

4. **クラウドプロバイダー** ドロップダウンからクラウドプロバイダーを選択します。
5. **名前** フィールドにストレージバケットの名前を入力します。Azure で [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスを持っている場合は、**アカウント名** および **コンテナ名** の値を入力してください。
6. (オプション) バケットのサブパスをオプションの **パス** フィールドに入力します。そうすることで、W&B がバケットのルートにあるフォルダにファイルを保存しないようにすることができます。
7. (AWS バケットを使用する場合はオプション) **KMSキーARN** フィールドに暗号化キーの ARN を入力します。
8. (Azure バケットを使用する場合はオプション) **テナントID** および **マネージド アイデンティティ クライアント ID** の値を入力します。
9. ([SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})でオプション) チームの作成時にオプションでチームメンバーを招待します。
10. **チームを作成** ボタンを押します。

{{< img src="/images/hosting/prod_setup_secure_storage.png" alt="" >}}

バケットにアクセスする際や設定が無効な場合、ページの下部にエラーや警告が表示されます。
{{% /tab %}}

{{% tab header="インスタンスレベル" value="instance"%}}
専用クラウドまたはセルフマネージドインスタンスのインスタンスレベル BYOB を設定するには、support@wandb.com まで W&B サポートにお問い合わせください。
{{% /tab %}}
{{< /tabpane >}}