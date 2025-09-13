---
title: W&B プラットフォームを AWS 上にデプロイ
description: AWS で W&B サーバーをホストする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-aws-tf
    parent: install-on-public-cloud
weight: 10
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B 専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) といったフルマネージドなデプロイメント オプションを推奨します。W&B のフルマネージド サービスは、設定が最小限または不要で、シンプルかつセキュアに利用できます。
{{% /alert %}}

W&B は、AWS 上にプラットフォームをデプロイする際に [W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) の利用を推奨します。

開始前に、Terraform の [State File](https://developer.hashicorp.com/terraform/language/state) を保存するために、Terraform で利用可能な [remote backends](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択することをおすすめします。

State ファイルは、すべてのコンポーネントを作り直すことなく、アップグレードやデプロイメントの変更を行うために必要なリソースです。

Terraform Module は、以下の「必須」コンポーネントをデプロイします:

- Load Balancer
- AWS Identity & Access Management (IAM)
- AWS Key Management System (KMS)
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager (ACM)
- Amazon Elastic Load Balancing (ALB)
- Amazon Secrets Manager

その他のデプロイメント オプションでは、以下のオプション コンポーネントも含められます:

- Elastic Cache for Redis
- SQS

## 事前必要な権限

Terraform を実行するアカウントには、イントロダクションに記載したすべてのコンポーネントを作成できること、ならびに **IAM Policies** と **IAM Roles** を作成し、ロールをリソースに割り当てる権限が必要です。

## 一般的な手順

ここで説明する手順は、このドキュメントで扱うすべてのデプロイメント オプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストール
   - バージョン管理のために Git リポジトリを作成することを W&B は推奨します。
2. `terraform.tfvars` ファイルを作成します。

   `tvfars` ファイルの内容はインストール形態に応じてカスタマイズできますが、最小限の推奨値は以下の例のようになります。

   ```bash
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   eks_cluster_version        = "1.29"
   ```

   デプロイ前に `tvfars` ファイルで変数を定義してください。`namespace` 変数は、Terraform が作成するすべてのリソースのプレフィックスに使われる文字列です。

   `subdomain` と `domain` の組み合わせが、W&B を設定する FQDN を構成します。上記の例では、W&B の FQDN は `wandb-aws.wandb.ml` となり、この FQDN レコードが作成される DNS の `zone_id` を指定します。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` も設定が必要です。このモジュールでは必須入力です。以下の例では、W&B インストールへの アクセス を任意の送信元から許可しています。

3. `versions.tf` ファイルを作成します。

   このファイルには、AWS に W&B をデプロイするのに必要な Terraform および Terraform provider の バージョン を記述します。

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

   AWS プロバイダーの設定については [Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

   任意ですが強く推奨されます。ドキュメント冒頭で触れた [remote backend の設定](https://developer.hashicorp.com/terraform/language/backend) を追加してください。

4. `variables.tf` ファイルを作成します。

   `terraform.tfvars` で設定する各オプションに対応する変数宣言が Terraform には必要です。

   ```
   variable "namespace" {
     type        = string
     description = "Name prefix used for resources"
   }

   variable "domain_name" {
     type        = string
     description = "Domain name used to access instance."
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Subdomain for accessing the Weights & Biases UI."
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Domain for creating the Weights & Biases subdomain on."
   }

   variable "allowed_inbound_cidr" {
    description = "CIDRs allowed to access wandb-server."
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "CIDRs allowed to access wandb-server."
    nullable    = false
    type        = list(string)
   }

   variable "eks_cluster_version" {
    description = "EKS cluster kubernetes version"
    nullable    = false
    type        = string
   }
   ```

## 推奨デプロイメント オプション

これは最もシンプルな構成で、すべての「必須」コンポーネントを作成し、`Kubernetes Cluster` に最新の `W&B` をインストールします。

1. `main.tf` を作成します

   「一般的な手順」で作成したファイルと同じディレクトリーに、以下の内容で `main.tf` を作成します:

   ```
   module "wandb_infra" {
     source  = "wandb/wandb/aws"
     version = "~>7.0"

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
     eks_cluster_version            = var.eks_cluster_version
   }

    data "aws_eks_cluster" "eks_cluster_id" {
      name = module.wandb_infra.cluster_name
    }

    data "aws_eks_cluster_auth" "eks_cluster_auth" {
      name = module.wandb_infra.cluster_name
    }

    provider "kubernetes" {
      host                   = data.aws_eks_cluster.eks_cluster_id.endpoint
      cluster_ca_certificate = base64decode(data.aws_eks_cluster.eks_cluster_id.certificate_authority.0.data)
      token                  = data.aws_eks_cluster_auth.eks_cluster_auth.token
    }


    provider "helm" {
      kubernetes {
        host                   = data.aws_eks_cluster.eks_cluster_id.endpoint
        cluster_ca_certificate = base64decode(data.aws_eks_cluster.eks_cluster_id.certificate_authority.0.data)
        token                  = data.aws_eks_cluster_auth.eks_cluster_auth.token
      }
    }

    output "url" {
      value = module.wandb_infra.url
    }

    output "bucket" {
      value = module.wandb_infra.bucket_name
    }
   ```

2. W&B をデプロイします

   W&B をデプロイするには、以下の コマンド を実行します:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## Redis を有効化

別のデプロイメント オプションとして、`Redis` を使って SQL クエリをキャッシュし、実験の メトリクス を読み込む際のアプリケーション応答を高速化できます。

キャッシュを有効にするには、[推奨デプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明したのと同じ `main.tf` にオプション `create_elasticache_subnet = true` を追加します。

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "~>7.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
	**create_elasticache_subnet = true**
}
[...]
```

## メッセージ ブローカー（キュー）を有効化

デプロイメント オプション 3 は、外部の `message broker` を有効にする構成です。W&B にはブローカーが同梱されているため任意であり、このオプションによる性能向上はありません。

メッセージ ブローカーを提供する AWS リソースは `SQS` です。有効化するには、[推奨デプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明したのと同じ `main.tf` に `use_internal_queue = false` オプションを追加します。

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "~>7.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
  **use_internal_queue = false**

[...]
}
```

## その他のデプロイメント オプション

3 つのデプロイメント オプションは、同じファイルにすべての設定を追加することで組み合わせ可能です。
[Terraform Module](https://github.com/wandb/terraform-aws-wandb) には、標準オプションや「Deployment - Recommended」の最小構成と組み合わせられる複数のオプションが用意されています。

## マニュアル設定

W&B のファイル ストレージ バックエンドとして Amazon S3 バケットを使用するには、以下が必要です:

* [Amazon S3 Bucket と Bucket Notifications を作成]({{< relref path="#create-an-s3-bucket-and-bucket-notifications" lang="ja" >}})
* [SQS Queue を作成]({{< relref path="#create-an-sqs-queue" lang="ja" >}})
* [W&B を実行するノードに権限を付与]({{< relref path="#grant-permissions-to-node-that-runs-wb" lang="ja" >}})

バケットを作成し、そのバケットからのオブジェクト作成通知を受け取るように設定された SQS キューも作成します。インスタンスには、このキューから読み取る権限が必要です。

### S3 バケットと Bucket Notifications の作成

以下の手順に従って Amazon S3 バケットを作成し、バケット通知を有効にします。

1. AWS コンソールで Amazon S3 に移動します。
2. **Create bucket** を選択します。
3. **Advanced settings** 内の **Events** セクションで **Add notification** を選択します。
4. すべてのオブジェクト作成イベントが、事前に設定した SQS Queue に送信されるように構成します。

{{< img src="/images/hosting/s3-notification.png" alt="エンタープライズ ファイル ストレージ 設定" >}}

CORS アクセス を有効にします。CORS 設定は以下のようになります:

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

### SQS Queue の作成

以下の手順で SQS Queue を作成します:

1. AWS コンソールで Amazon SQS に移動します。
2. **Create queue** を選択します。
3. **Details** セクションで **Standard** キュータイプを選択します。
4. Access policy セクションで、以下のプリンシパルに対する権限を追加します:
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

任意で、**Access Policy** セクションに高度な アクセス ポリシーを追加できます。たとえば、以下はステートメントで Amazon SQS に アクセス するためのポリシー例です:

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

### W&B を実行するノードに権限を付与

W&B サーバー が動作するノードには、Amazon S3 と Amazon SQS への アクセス を許可する設定が必要です。選択したサーバー デプロイメントの種類に応じて、ノード ロールに以下のポリシー ステートメントを追加する必要がある場合があります:

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

### W&B サーバーを設定
最後に、W&B Server を設定します。

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定 ページに移動します。
2. ***外部ファイル ストレージ バックエンドを使用する* オプション** を有効にします。
3. 次の形式で Amazon S3 バケット、リージョン、Amazon SQS キューの情報を入力します:
* **File Storage Bucket**: `s3://<bucket-name>`
* **File Storage Region (AWS only)**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

{{< img src="/images/hosting/configure_file_store.png" alt="AWS ファイル ストレージ 設定" >}}

4. **Update settings** を選択して新しい 設定 を適用します。

## W&B の バージョン をアップグレード

以下の手順に従って W&B をアップデートします:

1. `wandb_app` モジュールの 設定 に `wandb_version` を追加し、アップグレード先の W&B の バージョン を指定します。たとえば、次の行は W&B バージョン `0.48.1` を指定しています:

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  {{% alert %}}
  もしくは、`terraform.tfvars` に `wandb_version` を追加し、同名の変数を作成して、リテラル値の代わりに `var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 設定 を更新したら、[推奨デプロイメント セクション]({{< relref path="#recommended-deployment-option" lang="ja" >}}) に記載の手順を実施します。

## operator ベースの AWS Terraform モジュールへ移行

このセクションでは、[terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) モジュールを使用して、_pre-operator_ 環境から _post-operator_ 環境へアップグレードするための手順を説明します。

{{% alert %}}
W&B の アーキテクチャー には、Kubernetes の [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) パターンへの移行が必要です。詳細は、[アーキテクチャー変更の説明]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#reasons-for-the-architecture-shift" lang="ja" >}}) を参照してください。
{{% /alert %}}


### 変更前後の アーキテクチャー

以前の W&B の アーキテクチャー では、次のようにして:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

インフラストラクチャー を制御し、

{{< img src="/images/hosting/pre-operator-infra.svg" alt="pre-operator-infra" >}}

W&B Server のデプロイには次のモジュールを使用していました:

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

{{< img src="/images/hosting/pre-operator-k8s.svg" alt="pre-operator-k8s" >}}

移行後の アーキテクチャー では、次のようにします:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

これにより、Kubernetes クラスターへの インフラストラクチャー のインストールと W&B Server の両方を管理できるため、`post-operator.tf` での `module "wandb_app"` が不要になります。

{{< img src="/images/hosting/post-operator-k8s.svg" alt="post-operator-k8s" >}}

この アーキテクチャー変更 により、SRE/インフラ チームが手動の Terraform オペレーションを行わずに、OpenTelemetry、Prometheus、HPA、Kafka、イメージ更新などの追加機能を利用できるようになります。

W&B Pre-Operator のベース インストールから開始するには、`post-operator.tf` に `.disabled` というファイル拡張子が付いており、`pre-operator.tf` が有効（`.disabled` 拡張子が付いていない）であることを確認してください。これらのファイルは[ここ](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration)にあります。

### 前提条件

移行を開始する前に、以下の前提条件を満たしていることを確認してください:

- **Egress**: デプロイメントはエアギャップでは動作しません。**_Release Channel_** の最新仕様を取得するために [deploy.wandb.ai](https://deploy.wandb.ai) への アクセス が必要です。
- **AWS Credentials**: AWS リソースに対して適切な AWS クレデンシャルが設定されていること。
- **Terraform Installed**: 最新 バージョン の Terraform がシステムにインストールされていること。
- **Route53 Hosted Zone**: アプリケーションを配信するドメインに対応する既存の Route53 ホストゾーンがあること。
- **Pre-Operator Terraform Files**: `pre-operator.tf` と `pre-operator.tfvars` などの関連変数ファイルが正しく設定されていること。

### Pre-Operator セットアップ

Pre-Operator セットアップの構成を初期化・適用するには、以下の Terraform コマンドを実行します:

```bash
terraform init -upgrade
terraform apply -var-file=./pre-operator.tfvars
```

`pre-operator.tf` は次のようになります:

```ini
namespace     = "operator-upgrade"
domain_name   = "sandbox-aws.wandb.ml"
zone_id       = "Z032246913CW32RVRY0WU"
subdomain     = "operator-upgrade"
wandb_license = "ey..."
wandb_version = "0.51.2"
```

`pre-operator.tf` の構成では、2 つのモジュールを呼び出します:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

このモジュールは インフラストラクチャー を起動します。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

このモジュールは アプリケーション をデプロイします。

### Post-Operator セットアップ

`pre-operator.tf` に `.disabled` 拡張子が付いており、`post-operator.tf` が有効であることを確認します。

`post-operator.tfvars` には以下の追加変数が含まれます:

```ini
...
# wandb_version = "0.51.2" は、Release Channel 経由で管理されるか、User Spec で設定されます。

# アップグレードに必要な Operator 変数:
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

以下の コマンド を実行して Post-Operator 構成を初期化・適用します:

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

plan と apply のステップでは、以下のリソースが更新されます:

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

次のような出力が表示されます:

{{< img src="/images/hosting/post-operator-apply.png" alt="post-operator-apply" >}}

`post-operator.tf` には、次の 1 つだけが含まれている点に注意してください:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### post-operator 構成での変更点:

1. **Required Providers の更新**: プロバイダー互換性のために `required_providers.aws.version` を `3.6` から `4.0` に変更します。
2. **DNS と Load Balancer の構成**: `enable_dummy_dns` と `enable_operator_alb` を統合し、Ingress を通じて DNS レコードと AWS Load Balancer の設定を管理します。
3. **ライセンスとサイズの構成**: 新しい運用要件に合わせて、`license` と `size` パラメータを `wandb_infra` モジュールに直接渡します。
4. **カスタム ドメインの取り扱い**: 必要に応じて、`kube-system` ネームスペース内の External DNS ポッドの ログ を確認して DNS の問題をトラブルシュートするために `custom_domain_filter` を使用します。
5. **Helm プロバイダーの構成**: Kubernetes リソースを効果的に管理できるよう、Helm プロバイダーを有効化・構成します:

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

この包括的なセットアップにより、Pre-Operator から Post-Operator 構成へのスムーズな移行が可能となり、operator モデルによってもたらされる新たな効率性と機能を活用できます。