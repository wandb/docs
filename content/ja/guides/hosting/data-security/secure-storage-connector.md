---
title: Bring your own bucket (BYOB)
menu:
  default:
    identifier: ja-guides-hosting-data-security-secure-storage-connector
    parent: data-security
weight: 1
---

自分のバケットを持ち込む (BYOB) という機能を利用すると、W&B Artifacts やその他関連する機密データを自分のクラウドまたはオンプレミスのインフラストラクチャーに保存できます。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [SaaSクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})の場合、バケットに保存したデータは W&B 管理のインフラストラクチャーにはコピーされません。

{{% alert %}}
* W&B SDK / CLI / UI とあなたのバケット間の通信は、[事前署名付き URL]({{< relref path="./presigned-urls.md" lang="ja" >}})を使用して行われます。
* W&B はガベージコレクションプロセスを使って W&B Artifacts を削除します。詳しくは[Artifacts の削除]({{< relref path="/guides/core/artifacts/manage-data/delete-artifacts.md" lang="ja" >}})をご覧ください。
* バケットを設定するときにサブパスを指定することができ、W&B がバケットのルートフォルダーにファイルを保存しないようにできます。これにより、組織のバケット管理ポリシーにより確実に適合できます。
{{% /alert %}}

## 中央データベースとバケットに保存されるデータ

BYOB 機能を使用する場合、ある種のデータは W&B の中央データベースに保存され、その他の種類は自分のバケットに保存されます。

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
ストレージバケットを設定できる範囲は、*インスタンスレベル* または *チームレベル* の2つです。

- インスタンスレベル: あなたの組織内の関連する権限を持つユーザーが、インスタンスレベルのストレージバケットに保存されているファイルにアクセス可能。
- チームレベル: W&B チームのメンバーはチームレベルで設定されたバケットに保存されているファイルにアクセスできる。チームレベルのストレージバケットは、機密データや厳しい規制要件を持つチームに対して、より良いデータアクセス管理とデータ分離を実現します。

あなたの組織内で、インスタンスレベルと1つ以上のチームに対して個別にバケットを設定することができます。

例えば、組織に Kappa というチームがあるとします。この組織（およびチーム Kappa）はデフォルトでインスタンスレベルのストレージバケットを使用しています。次に、Omega というチームを作成します。Team Omega を作成する際に、そのチーム用のチームレベルのストレージバケットを設定します。チーム Omega により生成されたファイルはチーム Kappa によってアクセスされません。ただし、チーム Kappa により作成されたファイルはチーム Omega によってアクセス可能となります。チーム Kappa のデータを分離したい場合は、独自のチームレベルのストレージバケットを設定する必要があります。

{{% alert %}}
チームレベルのストレージバケットは、[セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンスにおいても同様のメリットをもたらします。特に異なるビジネスユニットや部門がインスタンスを共有してインフラストラクチャーと管理リソースを効率的に活用する際に有用です。また、別々の顧客との取引のために AI ワークフローを管理するプロジェクトチームが分かれている企業に対しても適用されます。
{{% /alert %}}

## アベイラビリティマトリックス
以下の表は、異なる W&B サーバーのデプロイメントタイプでの BYOB の利用可能性を示しています。`X` は特定のデプロイメントタイプで機能が利用可能であることを意味します。

| W&B サーバーのデプロイメントタイプ | インスタンスレベル | チームレベル | 追加情報 |
|----------------------------|--------------------|----------------|------------------------|
| 専用クラウド | X | X | インスタンスレベルとチームレベルの両方の BYOB は Amazon Web Services、Google Cloud Platform、Microsoft Azure で利用可能です。チームレベルの BYOB については、同じクラウドまたは別のクラウドのクラウドネイティブストレージバケット、または S3 互換のセキュアストレージ（例：MinIO）に接続できます。 |
| SaaS クラウド | 該当なし | X | チームレベルの BYOB は Amazon Web Services と Google Cloud Platform でのみ利用可能です。W&B が Microsoft Azure のデフォルトかつ唯一のストレージバケットを完全に管理しています。 |
| セルフマネージド | X | X | インスタンスレベルの BYOB はデフォルトです。インスタンスはあなたによって完全に管理されているためクラウドにセルフマネージドインスタンスがある場合、同じまたは別のクラウドのクラウドネイティブストレージバケットに接続してチームレベルの BYOB を実現できます。インスタンスまたはチームレベルの BYOB のいずれかに MinIO などの S3 互換セキュアストレージを使用することも可能です。 |

{{% alert color="secondary" %}}
専用クラウドやセルフマネージドインスタンス、または SaaS クラウドアカウントでは、インスタンスまたはチームレベルのストレージバケットを一度設定すると、そのスコープに対してストレージバケットを変更または再設定することはできません。この制約には別のバケットへのデータ移行や、メインプロダクトストレージの関連参照の再マッピングが含まれます。 W&B はインスタンスまたはチームレベルのスコープのためにストレージバケットのレイアウトを慎重に計画することを推奨します。質問がある場合は、W&B チームに連絡してください。
{{% /alert %}}

## チームレベルの BYOB のための異なるクラウドまたは S3 互換ストレージ

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンス内でチームレベル BYOB のために他のクラウドのクラウドネイティブストレージバケットまたは S3 互換ストレージバケット（例：MinIO）に接続することができます。

異なるクラウドまたは S3 互換ストレージの利用を可能にするために、`GORILLA_SUPPORTED_FILE_STORES` 環境変数を用いて、下記の形式のいずれかでストレージバケットを指定してください。

<details>
<summary>専用クラウドまたはセルフマネージドインスタンスでのチームレベル BYOB のための S3 互換ストレージの設定</summary>

以下の形式でパスを指定してください。
```text
s3://<accessKey>:<secretAccessKey>@<url_endpoint>/<bucketName>?region=<region>?tls=true
```
`region` パラメータは必須ですが、W&B インスタンスが AWS 内にあり、W&B インスタンスノードで設定されている `AWS_REGION` が S3 互換ストレージに設定されたリージョンと一致している場合は例外です。

</details>
<details>
<summary>専用クラウドまたはセルフマネージドインスタンスでのチームレベル BYOB のための異なるクラウドネイティブストレージの設定</summary>

W&Bインスタンスとストレージバケットの場所に特定の形式でパスを指定してください。

GCP または Azure 内の W&B インスタンスから AWS のバケットへ:
```text
s3://<accessKey>:<secretAccessKey>@<s3_regional_url_endpoint>/<bucketName>
```

GCP または AWS 内の W&B インスタンスから Azure のバケットへ:
```text
az://:<urlEncodedAccessKey>@<storageAccountName>/<containerName>
```

AWS または Azure 内の W&B インスタンスから GCP のバケットへ:
```text
gs://<serviceAccountEmail>:<urlEncodedPrivateKey>@<bucketName>
```

</details>

{{% alert %}}
チームレベル BYOB のための S3 互換ストレージとの接続は [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) では利用できません。また、チームレベル BYOB のための AWS バケットとの接続は、[SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})内のクロスクラウドです。そのインスタンスは GCP 内にあります。そのクロスクラウドの接続は、専用クラウドまたはセルフマネージドインスタンスで前述したアクセスポイントと環境変数に基づくメカニズムを使用していません。
{{% /alert %}}

詳細については、support@wandb.com から W&B サポートに連絡してください。

## W&B プラットフォームと同じクラウドでのクラウドストレージ

ユースケースに基づいて、チームまたはインスタンスレベルでストレージバケットを設定します。ストレージバケットのプロビジョニングや設定方法は、設定するレベルに関係なく同じですが、Azure でのアクセスメカニズムを除きます。

{{% alert %}}
W&B は、必要なアクセスメカニズムおよび関連するIAM権限と共に、ストレージバケットをプロビジョニングするために W&B が管理する Terraform モジュールを使用することを推奨します。

* [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector)
* [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector)
* Azure - [インスタンスレベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) または [チームレベル BYOB](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector)
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="AWS" value="aws" %}}
1. KMS キーをプロビジョニングする

    W&B は、S3 バケット上のデータを暗号化および復号化するために KMS キーのプロビジョニングを要求します。キー使用タイプは `ENCRYPT_DECRYPT` である必要があります。キーに次のポリシーを割り当てます。

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

    `<Your_Account_Id>` と `<aws_kms_key.key.arn>` をそれぞれ置き換えてください。

    [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、`<aws_principal_and_role_arn>` を以下の対応する値で置き換えてください。

    * [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合: `arn:aws:iam::725579432336:role/WandbIntegration`
    * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合: `arn:aws:iam::830241207209:root`

    このポリシーは、AWS アカウントにキーへのフルアクセス権限を付与し、W&B プラットフォームをホストしている AWS アカウントに必要な権限も割り当てます。KMS キーの ARN を記録してください。

2. S3 バケットをプロビジョニングする

    次の手順に従って、AWS アカウント内に S3 バケットをプロビジョニングします。

    1. 任意の名前で S3 バケットを作成します。必要に応じて、すべての W&B ファイルを保存するためにサブパスとして設定できるフォルダを作成します。
    2. バージョン管理を有効にします。
    3. サーバーサイド暗号化を有効にします。前のステップからの KMS キーを使用します。
    4. 次のポリシーを使用して CORS を設定します。

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

    5. [事前署名付き URL]({{< relref path="./presigned-urls.md" lang="ja" >}})の生成に必要な S3 の権限を、W&B プラットフォームをホストしている AWS アカウントに付与します。これらの権限はあなたのクラウドインフラストラクチャー内の AI ワークロードまたはユーザーブラウザがバケットにアクセスするのに必要です。

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

        `<wandb_bucket>` を適宜置き換えて、バケット名を記録してください。 [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合、インスタンスレベル BYOB の場合 W&B チームにバケット名を共有してください。 どのデプロイメントタイプでもチームレベル BYOB の場合は、[チーム作成時にバケットを設定してください]({{< relref path="#configure-byob-in-wb" lang="ja" >}})。

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) を使用している場合、`<aws_principal_and_role_arn>` を対応する値で置き換えます。

*SaaS クラウド*([SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) の場合): `arn:aws:iam::725579432336:role/WandbIntegration`
*専用クラウド*([専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) の場合): `arn:aws:iam::830241207209:root`

詳細については、[AWS セルフマネージドホスティングガイド]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}})を参照してください。
{{% /tab %}}

{{% tab header="GCP" value="gcp"%}}
1. GCS バケットをプロビジョニングする

    次の手順に従って、GCP プロジェクト内に GCS バケットをプロビジョニングします。

    1. 任意の名前で GCS バケットを作成します。オプションで、サブパスとして設定できるフォルダを作成して、すべての W&B ファイルを格納します。
    2. ソフトデリーションを有効にします。
    3. オブジェクトバージョニングを有効にします。
    4. 暗号化タイプを `Google-managed` に設定します。
    5. `gsutil` を使用して CORS ポリシーを設定します。UI では設定できません。

      1. ローカルに `cors-policy.json` というファイルを作成します。
      2. 以下の CORS ポリシーをファイルにコピーして保存してください。

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

      3. `<bucket_name>` を正しいバケット名に置き換えて `gsutil` を実行します。

          ```bash
          gsutil cors set cors-policy.json gs://<bucket_name>
          ```

      4. バケットのポリシーを確認してください。`<bucket_name>` を正しいバケット名に置き換えます。
        
          ```bash
          gsutil cors get gs://<bucket_name>
          ```

2. [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合、W&B プラットフォームにリンクされた GCP のサービスアカウントに `ストレージ管理者` 役割を付与します。

    * [SaaS クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) アカウント: `wandb-integration@wandb-production.iam.gserviceaccount.com`
    * [専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) アカウント: `deploy@wandb-production.iam.gserviceaccount.com`

    バケット名を記録しておいてください。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合は、インスタンスレベル BYOB の場合、バケット名を W&B チームに共有してください。いずれのデプロイメントタイプでもチームレベル BYOB の場合は、[チーム作成時にバケットを設定します]({{< relref path="#configure-byob-in-wb" lang="ja" >}})。
{{% /tab %}}

{{% tab header="Azure" value="azure"%}}
1. Azure Blob Storage をプロビジョニングする

    インスタンスレベルの BYOB の場合、[この Terraform モジュール](https://github.com/wandb/terraform-azurerm-wandb/tree/main/examples/byob) を使用していない場合は、以下の手順に従って、Azure サブスクリプションに Azure Blob Storage バケットをプロビジョニングします。

    * 任意の名前でバケットを作成します。必要に応じて、すべての W&B ファイルを保存するためのサブパスとして設定できるフォルダを作成します。
    * BLOB とコンテナーのソフトデリーションを有効にします。
    * バージョン管理を有効にします。
    * バケットの CORS ポリシーを設定します。

      UI を介して CORS ポリシーを設定するには、BLOB ストレージに移動し、`設定/リソース共有 (CORS)`までスクロールし、次のように設定します。

      | パラメータ | 値 |
      | --- | --- |
      | 許可されたオリジン | `*`  |
      | 許可されたメソッド | `GET`, `HEAD`, `PUT` |
      | 許可されたヘッダー | `*` |
      | 公開されたヘッダー | `*` |
      | 最大年齢 | `3600` |

2. ストレージアカウントのアクセスキーを生成し、そのストレージアカウント名とともに記録してください。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) を使用している場合、安全な共有メカニズムを用いて、ストレージアカウント名とアクセスキーを W&B チームと共有してください。

    チームレベルの BYOB の場合、W&B は [Terraform](https://github.com/wandb/terraform-azurerm-wandb/tree/main/modules/secure_storage_connector) を使用して、必要なアクセスメカニズムと権限を伴う Azure Blob Storage バケットをプロビジョニングすることを推奨します。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})を使用している場合は、インスタンスの OIDC 発行者 URL を提供します。次の詳細をメモしておき、チーム作成時にバケットを設定するために使用します:

    * ストレージアカウント名
    * ストレージコンテナ名
    * 管理されたアイデンティティ クライアント ID
    * Azure テナントID
{{% /tab %}}
{{< /tabpane >}}

## W&B での BYOB の設定

{{< tabpane text=true >}}

{{% tab header="チームレベル" value="team" %}}
{{% alert %}}
異なるクラウドのクラウドネイティブストレージバケットに接続するか、[MinIO](https://github.com/minio/minio) のような S3 互換ストレージバケットに専用クラウドまたはセルフマネージドインスタンス内のチームレベル BYOB に使用している場合、チームレベル BYOB の異なるクラウドまたは S3 互換ストレージに[S3 互換ストレージ](./#cross-cloud-or-s3-compatible-storage-for-team-level-byob) を参照してください。そんな場合、W&B インスタンス用の `GORILLA_SUPPORTED_FILE_STORES` 環境変数を使用してストレージバケットを指定し、その後に次の手順に従って、チーム用に設定してください。
{{% /alert %}}

W&B チーム作成時にチームレベルでストレージバケットを設定するには:

1. **チーム名** フィールドにチーム名を入力してください。
2. **ストレージタイプ** オプションで **外部ストレージ** を選択します。
3. ドロップダウンから **新しいバケット** を選択するか、既存のバケットを選択します。

    複数の W&B チームが同じクラウドストレージバケットを使用することができます。これを有効にするには、既存のクラウドストレージバケットをドロップダウンから選択します。

4. **クラウドプロバイダー** のドロップダウンからクラウドプロバイダーを選択します。
5. **名前** フィールドにストレージバケットの名前を入力してください。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスの場合、アカウント名とコンテナ名のフィールドにそれぞれの値を入力してください。
6. （オプション）バケツルートのフォルダに W&B がファイルを保存しない場合は必須）サブパスバケットのオプションフィールドに入力してください。
7. （オプション、AWS バケット使用時のみ適用）**KMS キー ARN** フィールドに KMS 暗号化キーの ARN を入力してください。
8. （オプション、Azure バケット使用時のみ適用）**テナント ID** フィールドと **管理されたアイデンティティクライアント ID** フィールドにそれぞれの値を入力します。
9. （オプション、*SaaS クラウド*のみに適用）チームメンバーをチームに招待することもできます。
10. **チーム作成** ボタンを押してください。

{{< img src="/images/hosting/prod_setup_secure_storage.png" alt="" >}}

バケットのアクセスに問題がある場合や、バケットが無効な設定になっている場合は、ページの下部にエラーまたは警告が表示されます。
{{% /tab %}}

{{% tab header="インスタンスレベル" value="instance"%}}
インスタンスレベルの BYOB を調整するには、W&B サポート (support@wandb.com) に連絡してください。専用クラウドまたはセルフマネージドインスタンスの場合は、W&B サポートチームに連絡する必要があります。
{{% /tab %}}
{{< /tabpane >}}