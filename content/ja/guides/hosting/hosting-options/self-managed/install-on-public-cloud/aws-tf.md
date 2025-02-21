---
title: Deploy W&B Platform on AWS
description: AWS 上での W&B サーバー のホスティング。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-aws-tf
    parent: install-on-public-cloud
weight: 10
---

{{% alert %}}
Weights & Biases ( W&B )では、 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドの デプロイメント オプションをお勧めします。 W&B のフルマネージドサービスは、シンプルで安全に使用でき、設定は最小限で済みます。
{{% /alert %}}

W&B は、 [W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) を使用して、 AWS 上に プラットフォーム を デプロイメント することをお勧めします。

開始する前に、 Terraform で利用可能な [リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択して、 [State File](https://developer.hashicorp.com/terraform/language/state) を保存することを W&B はお勧めします。

State File は、すべての コンポーネント を再作成せずに、アップグレードを展開したり、 デプロイメント に変更を加えたりするために必要なリソースです。

Terraform Module は、次の必須 コンポーネント を デプロイメント します。

- ロードバランサー
- AWS Identity & Access Management ( IAM )
- AWS Key Management System ( KMS )
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager ( ACM )
- Amazon Elastic Load Balancing ( ALB )
- Amazon Secrets Manager

その他の デプロイメント オプションには、次のオプション コンポーネント も含めることができます。

- Redis 用 Elastic Cache
- SQS

## 前提条件の権限

Terraform を実行する アカウント は、 イントロダクション で説明されているすべての コンポーネント を作成でき、 **IAM ポリシー** と **IAM ロール** を作成し、ロールをリソースに割り当てる権限が必要です。

## 一般的なステップ

このトピックのステップは、このドキュメントで説明されているどの デプロイメント オプションにも共通です。

1. 開発 環境 を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
   - W&B は、 バージョン 管理のために Git リポジトリを作成することをお勧めします。
2. `terraform.tfvars` ファイルを作成します。

   `tvfars` ファイルの内容は、インストール タイプ に応じてカスタマイズできますが、推奨される最小限のものは、以下の例のようになります。

   ```bash
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   ```

   `namespace` 変数は、 Terraform によって作成されたすべてのリソースのプレフィックスとなる文字列であるため、 デプロイメント する前に `tvfars` ファイルで変数を定義してください。

   `subdomain` と `domain` の組み合わせで、 W&B が設定される FQDN が形成されます。 上記の例では、 W&B の FQDN は `wandb-aws.wandb.ml` となり、 FQDN レコードが作成される DNS `zone_id` となります。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` の両方も設定が必要です。 モジュールでは、これは必須の入力です。 実行例では、すべてのソースから W&B インストールへの アクセス が許可されています。

3. ファイル `versions.tf` を作成します。

   このファイルには、 AWS に W&B を デプロイメント するために必要な Terraform と Terraform プロバイダー の バージョン が含まれます。

   ```bash
   provider "aws" {
     region = "eu-central-1"

     default_tags {
       tags = {
         GithubRepo = "terraform-aws-wandb"
         GithubOrg  = "wandb"
         Enviroment = "Example"
         Example    = "PublicDnsExternal"
       }
     }
   }
   ```

   AWS プロバイダー を設定するには、 [Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

   オプションですが、強く推奨されるのは、このドキュメントの冒頭で言及した [リモートバックエンド構成](https://developer.hashicorp.com/terraform/language/backend) を追加することです。

4. ファイル `variables.tf` を作成します。

   `terraform.tfvars` で設定されたすべてのオプションについて、 Terraform は対応する変数宣言を必要とします。

   ```
   variable "namespace" {
     type        = string
     description = "リソースに使用される名前プレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "インスタンスへの アクセス に使用される ドメイン 名。"
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UI に アクセス するための サブドメイン 。"
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biases サブドメイン を作成する ドメイン 。"
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-server への アクセス を許可された CIDR。"
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server への アクセス を許可された CIDR。"
    nullable    = false
    type        = list(string)
   }
   ```

## 推奨される デプロイメント オプション

これは、すべての `Mandatory` コンポーネント を作成し、 `Kubernetes Cluster` に最新 バージョン の `W&B` をインストールする最も簡単な デプロイメント オプション構成です。

1. `main.tf` を作成します。

   `General Steps` でファイルを作成した同じ ディレクトリー に、次の内容でファイル `main.tf` を作成します。

   ```
   module "wandb_infra" {
     source  = "wandb/wandb/aws"
     version = "~>2.0"

     namespace   = var.namespace
     domain_name = var.domain_name
     subdomain   = var.subdomain
     zone_id     = var.zone_id

     allowed_inbound_cidr           = var.allowed_inbound_cidr
     allowed_inbound_ipv6_cidr      = var.allowed_inbound_ipv6_cidr

     public_access                  = true
     external_dns                   = true
     kubernetes_public_access       = true
     kubernetes_public_access_cidrs = ["0.0.0.0/0"]
   }

   data "aws_eks_cluster" "app_cluster" {
     name = module.wandb_infra.cluster_id
   }

   data "aws_eks_cluster_auth" "app_cluster" {
     name = module.wandb_infra.cluster_id
   }

   provider "kubernetes" {
     host                   = data.aws_eks_cluster.app_cluster.endpoint
     cluster_ca_certificate = base64decode(data.aws_eks_cluster.app_cluster.certificate_authority.0.data)
     token                  = data.aws_eks_cluster_auth.app_cluster.token
   }

   module "wandb_app" {
     source  = "wandb/wandb/kubernetes"
     version = "~>1.0"

     license                    = var.license
     host                       = module.wandb_infra.url
     bucket                     = "s3://${module.wandb_infra.bucket_name}"
     bucket_aws_region          = module.wandb_infra.bucket_region
     bucket_queue               = "internal://"
     database_connection_string = "mysql://${module.wandb_infra.database_connection_string}"

     # TF attempts to deploy while the work group is
     # still spinning up if you do not wait
     depends_on = [module.wandb_infra]
   }

   output "bucket_name" {
     value = module.wandb_infra.bucket_name
   }

   output "url" {
     value = module.wandb_infra.url
   }
   ```

2. W&B を デプロイメント します。

   W&B を デプロイメント するには、次の コマンド を実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS を有効にする

別の デプロイメント オプションでは、 `Redis` を使用して SQL クエリをキャッシュし、 実験 の メトリクス をロードする際の アプリケーション の応答を高速化します。

キャッシュを有効にするには、 [推奨される デプロイメント ]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明されているのと同じ `main.tf` ファイルにオプション `create_elasticache_subnet = true` を追加する必要があります。

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "~>2.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
	**create_elasticache_subnet = true**
}
[...]
```

## メッセージブローカー (キュー) を有効にする

デプロイメント オプション 3 は、外部 `message broker` を有効にすることです。 W&B には ブローカー が組み込まれているため、これはオプションです。 このオプションは、 パフォーマンス の向上をもたらしません。

メッセージブローカー を提供する AWS リソースは `SQS` です。これを有効にするには、 [推奨される デプロイメント ]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明されているのと同じ `main.tf` にオプション `use_internal_queue = false` を追加する必要があります。

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "~>2.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
  **use_internal_queue = false**

[...]
}
```

## その他の デプロイメント オプション

3 つすべての デプロイメント オプションを組み合わせて、すべての構成を同じファイルに追加できます。
[Terraform Module](https://github.com/wandb/terraform-aws-wandb) は、標準オプションと `Deployment - Recommended` にある最小限の構成とともに組み合わせることができるいくつかのオプションを提供します。

## 手動構成

Amazon S3 バケットを W&B のファイル ストレージ バックエンドとして使用するには、次のことを行う必要があります。

* [Amazon S3 バケットと バケット 通知を作成する]({{< relref path="#create-an-s3-bucket-and-bucket-notifications" lang="ja" >}})
* [SQS キューを作成する]({{< relref path="#create-an-sqs-queue" lang="ja" >}})
* [W&B を実行している ノード に権限を付与する]({{< relref path="#grant-permissions-to-node-that-runs-wb" lang="ja" >}})

バケットからのオブジェクト作成通知を受信するように構成された SQS キューとともに、 バケット を作成する必要があります。 インスタンスには、このキューから読み取るための権限が必要です。

### S3 バケット と バケット 通知を作成する

Amazon S3 バケット を作成し、 バケット 通知を有効にするには、次の手順に従います。

1. AWS コンソール で Amazon S3 に移動します。
2. [ バケット の作成]を選択します。
3. [詳細設定]で、[ イベント ]セクションの[通知の追加]を選択します。
4. 以前に構成した SQS キューに送信されるように、すべてのオブジェクト作成 イベント を構成します。

{{< img src="/images/hosting/s3-notification.png" alt="エンタープライズファイルストレージ設定" >}}

CORS アクセス を有効にします。 CORS 構成は次のようになります。

```markup
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<CORSRule>
    <AllowedOrigin>http://YOUR-W&B-SERVER-IP</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
</CORSRule>
</CORSConfiguration>
```

### SQS キューを作成する

SQS キューを作成するには、次の手順に従います。

1. AWS コンソール で Amazon SQS に移動します。
2. [キューの作成]を選択します。
3. [詳細]セクションで、[標準]キュー タイプ を選択します。
4. アクセス ポリシー セクションで、次の プリンシパル に権限を追加します。
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

オプションで、[ アクセス ポリシー ]セクションに高度な アクセス ポリシー を追加します。 たとえば、ステートメント を使用して Amazon SQS に アクセス するためのポリシーは次のようになります。

```json
{
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Principal" : "*",
        "Action" : ["sqs:SendMessage"],
        "Resource" : "<sqs-queue-arn>",
        "Condition" : {
          "ArnEquals" : { "aws:SourceArn" : "<s3-bucket-arn>" }
        }
      }
    ]
}
```

### W&B を実行する ノード に権限を付与する

W&B サーバー が実行されている ノード は、 Amazon S3 および Amazon SQS への アクセス を許可するように構成する必要があります。 選択した サーバー デプロイメント の タイプ に応じて、次のポリシー ステートメント を ノード ロールに追加する必要がある場合があります。

```json
{
   "Statement":[
      {
         "Sid":"",
         "Effect":"Allow",
         "Action":"s3:*",
         "Resource":"arn:aws:s3:::<WANDB_BUCKET>"
      },
      {
         "Sid":"",
         "Effect":"Allow",
         "Action":[
            "sqs:*"
         ],
         "Resource":"arn:aws:sqs:<REGION>:<ACCOUNT>:<WANDB_QUEUE>"
      }
   ]
}
```

### W&B サーバー を構成する
最後に、 W&B サーバー を構成します。

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定 ページ に移動します。
2. ***外部ファイル ストレージ バックエンドを使用する* オプションを有効にします。
3. Amazon S3 バケット 、 リージョン 、および Amazon SQS キューに関する情報を次の形式で提供します。
* **ファイル ストレージ バケット **: `s3://<bucket-name>`
* **ファイル ストレージ リージョン ( AWS のみ)**: `<region>`
* **通知サブスクリプション**: `sqs://<queue-name>`

{{< img src="/images/hosting/configure_file_store.png" alt="" >}}

4. [設定の更新]を選択して、新しい設定を適用します。

## W&B バージョン をアップグレードする

W&B を更新するには、ここで説明する手順に従ってください。

1. `wandb_app` モジュールの構成に `wandb_version` を追加します。 アップグレードする W&B の バージョン を指定します。 たとえば、次の行は W&B バージョン `0.48.1` を指定します。

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  {{% alert %}}
  または、 `wandb_version` を `terraform.tfvars` に追加し、同じ名前の変数を作成し、リテラル値を使用する代わりに、 `var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 構成を更新したら、 [推奨される デプロイメント セクション]({{< relref path="#recommended-deployment-option" lang="ja" >}}) で説明されている手順を完了します。

## オペレーター ベース の AWS Terraform モジュール への移行

このセクションでは、 [terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) モジュールを使用して、 _pre-operator_ 環境から _post-operator_ 環境にアップグレードするために必要な手順について詳しく説明します。

{{% alert %}}
Kubernetes [オペレーター](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) パターンへの移行は、 W&B アーキテクチャー に必要です。 アーキテクチャー シフトの詳細な説明については、 [このセクション]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#reasons-for-the-architecture-shift" lang="ja" >}}) を参照してください。
{{% /alert %}}

### 移行前後のアーキテクチャー

以前は、 W&B アーキテクチャー では以下を使用していました。

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

インフラストラクチャー を制御します。

{{< img src="/images/hosting/pre-operator-infra.svg" alt="移行前-インフラストラクチャー" >}}

また、このモジュールを使用して W&B サーバー を デプロイメント します。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

{{< img src="/images/hosting/pre-operator-k8s.svg" alt="移行前-k8s" >}}

移行後、アーキテクチャー では以下を使用します。

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

インフラストラクチャー のインストールと W&B サーバー の Kubernetes クラスター へのインストール の両方を管理するため、 `post-operator.tf` で `module "wandb_app"` は不要になります。

{{< img src="/images/hosting/post-operator-k8s.svg" alt="移行後-k8s" >}}

このアーキテクチャー シフトにより、 SRE/ インフラストラクチャー チームによる手動の Terraform 操作を必要とせずに、追加機能 ( OpenTelemetry 、 Prometheus 、 HPA 、 Kafka 、 イメージ 更新など) を有効にできます。

W&B Pre-Operator の基本インストールを開始するには、 `post-operator.tf` に `.disabled` ファイル拡張子があり、 `pre-operator.tf` がアクティブ ( `.disabled` 拡張子がない) であることを確認します。 これらのファイルは、 [こちら](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration) にあります。

### 前提条件

移行プロセスを開始する前に、次の前提条件が満たされていることを確認してください。

- **Egress**: デプロイメント を airgapped にすることはできません。 **_Release Channel_** の最新仕様を取得するには、 [deploy.wandb.ai](https://deploy.wandb.ai) への アクセス が必要です。
- **AWS 認証情報**: AWS リソースと対話するように構成された適切な AWS 認証情報。
- **Terraform のインストール**: 最新 バージョン の Terraform がシステムにインストールされている必要があります。
- **Route53 ホスト ゾーン**: アプリケーション が提供される ドメイン に対応する既存の Route53 ホスト ゾーン 。
- **Pre-Operator Terraform ファイル**: `pre-operator.tf` および関連する変数ファイル ( `pre-operator.tfvars` など) が正しく設定されていることを確認します。

### Pre-Operator の設定

次の Terraform コマンド を実行して、 Pre-Operator 設定の構成を初期化して適用します。

```bash
terraform init -upgrade
terraform apply -var-file=./pre-operator.tfvars
```

`pre-operator.tf` は次のようになります。

```ini
namespace     = "operator-upgrade"
domain_name   = "sandbox-aws.wandb.ml"
zone_id       = "Z032246913CW32RVRY0WU"
subdomain     = "operator-upgrade"
wandb_license = "ey..."
wandb_version = "0.51.2"
```

`pre-operator.tf` 構成は、2 つのモジュールを呼び出します。

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

このモジュールは、 インフラストラクチャー をスピンアップします。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

このモジュールは、 アプリケーション を デプロイメント します。

### Post-Operator の設定

`pre-operator.tf` に `.disabled` 拡張子があり、 `post-operator.tf` がアクティブであることを確認します。

`post-operator.tfvars` には、追加の変数が含まれています。

```ini
...
# wandb_version = "0.51.2" は、 Release Channel 経由で管理されるか、 ユーザー 仕様で設定されます。

# アップグレードに必要な オペレーター 変数:
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

次の コマンド を実行して、 Post-Operator 構成を初期化して適用します。

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

計画と適用 の手順により、次のリソースが更新されます。

```yaml
actions:
  create:
    - aws_efs_backup_policy.storage_class
    - aws_efs_file_system.storage_class
    - aws_efs_mount_target.storage_class["0"]
    - aws_efs_mount_target.storage_class["1"]
    - aws_eks_addon.efs
    - aws_iam_openid_connect_provider.eks
    - aws_iam_policy.secrets_manager
    - aws_iam_role_policy_attachment.ebs_csi
    - aws_iam_role_policy_attachment.eks_efs
    - aws_iam_role_policy_attachment.node_secrets_manager
    - aws_security_group.storage_class_nfs
    - aws_security_group_rule.nfs_ingress
    - random_pet.efs
    - aws_s3_bucket_acl.file_storage
    - aws_s3_bucket_cors_configuration.file_storage
    - aws_s3_bucket_ownership_controls.file_storage
    - aws_s3_bucket_server_side_encryption_configuration.file_storage
    - helm_release.operator
    - helm_release.wandb
    - aws_cloudwatch_log_group.this[0]
    - aws_iam_policy.default
    - aws_iam_role.default
    - aws_iam_role_policy_attachment.default
    - helm_release.external_dns
    - aws_default_network_acl.this[0]
    - aws_default_route_table.default[0]
    - aws_iam_policy.default
    - aws_iam_role.default
    - aws_iam_role_policy_attachment.default
    - helm_release.aws_load_balancer_controller

  update_in_place:
    - aws_iam_policy.node_IMDSv2
    - aws_iam_policy.node_cloudwatch
    - aws_iam_policy.node_kms
    - aws_iam_policy.node_s3
    - aws_iam_policy.node_sqs
    - aws_eks_cluster.this[0]
    - aws_elasticache_replication_group.default
    - aws_rds_cluster.this[0]
    - aws_rds_cluster_instance.this["1"]
    - aws_default_security_group.this[0]
    - aws_subnet.private[0]
    - aws_subnet.private[1]
    - aws_subnet.public[0]
    - aws_subnet.public[1]
    - aws_launch_template.workers["primary"]

  destroy:
    - kubernetes_config_map.config_map
    - kubernetes_deployment.wandb
    - kubernetes_priority_class.priority
    - kubernetes_secret.secret
    - kubernetes_service.prometheus
    - kubernetes_service.service
    - random_id.snapshot_identifier[0]

  replace:
    - aws_autoscaling_attachment.autoscaling_attachment["primary"]
    - aws_route53_record.alb
    - aws_eks_node_group.workers["primary"]
```

次のようになります。

{{< img src="/images/hosting/post-operator-apply.png" alt="移行後-適用" >}}

`post-operator.tf` には、単一の

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### 移行後の構成の変更点:

1. **必要な プロバイダー の更新**: プロバイダー の互換性のために、 `required_providers.aws.version` を `3.6` から `4.0` に変更します。
2. **DNS およびロードバランサー の構成**: Ingress を介して DNS レコードと AWS ロードバランサー の設定を管理するために、 `enable_dummy_dns` と `enable_operator_alb` を統合します。
3. **ライセンス と サイズ の構成**: 新しい運用要件に合わせて、 `license` パラメータと `size` パラメータを `wandb_infra` モジュールに直接転送します。
4. **カスタム ドメイン の処理**: 必要に応じて、 `kube-system` 名前空間内の 外部 DNS ポッド ログ を確認して DNS の問題をトラブルシューティングするために、 `custom_domain_filter` を使用します。
5. **Helm プロバイダー の構成**: Kubernetes リソースを効果的に管理するために、 Helm プロバイダー を有効にして構成します。

```hcl
provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.app_cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.app_cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.app_cluster.token
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      args        = ["eks", "get-token", "--cluster-name", data.aws_eks_cluster.app_cluster.name]
      command     = "aws"
    }
  }
}
```

この包括的な設定により、オペレーター モデルによって有効になる新しい効率と機能を活用して、 Pre-Operator 構成から Post-Operator 構成へのスムーズな移行が保証されます。
