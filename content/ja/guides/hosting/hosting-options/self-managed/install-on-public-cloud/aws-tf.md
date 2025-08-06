---
title: AWS に W&B プラットフォームをデプロイする
description: AWS で W&B サーバー をホスティングする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-aws-tf
    parent: install-on-public-cloud
weight: 10
---

{{% alert %}}
W&B では、[W&B マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) など、完全に管理されたデプロイメントオプションを推奨しています。W&B のフルマネージドサービスはシンプルかつ安全にご利用いただけ、ほとんどあるいは全く設定を必要としません。
{{% /alert %}}

W&B では、AWS 上にプラットフォームをデプロイする際に [W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) のご利用を推奨しています。

開始する前に、Terraform の [リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択し、[State File](https://developer.hashicorp.com/terraform/language/state) を保存することを推奨します。

State File は、すべてのコンポーネントを再作成することなく、デプロイメントのアップグレードや変更を行うために必要なリソースです。

Terraform Module では、以下の「必須」コンポーネントがデプロイされます：

- ロードバランサー
- AWS Identity & Access Management (IAM)
- AWS Key Management System (KMS)
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager (ACM)
- Amazon Elastic Load Balancing (ALB)
- Amazon Secrets Manager

その他のデプロイメントオプションでは、以下の「オプション」コンポーネントも含めることができます：

- Elastic Cache for Redis
- SQS

## 事前に必要な権限

Terraform を実行するアカウントには、イントロダクションに記載されているすべてのコンポーネントの作成権限、**IAM ポリシー**や **IAM ロール**の作成およびリソースへのロール割り当て権限が必要です。

## 一般的な手順

このトピックで説明する手順は、このドキュメントでカバーされているすべてのデプロイメントオプションに共通するものです。

1. 開発環境を準備する。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストール
   - バージョン管理のために Git レポジトリの作成を推奨します。
2. `terraform.tfvars` ファイルを作成します。

   `tfvars` ファイルの内容はインストールタイプに合わせてカスタマイズできますが、最低限必要なのは下記の例のようになります。

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

   デプロイ前に `tfvars` ファイルで変数を定義してください。`namespace` 変数が、Terraform で作成されるすべてのリソースにプレフィックスとして付与されます。

   `subdomain` と `domain` の組み合わせで W&B が設定される FQDN が決まります。上記の例の場合、W&B の FQDN は `wandb-aws.wandb.ml` となり、その FQDN のレコードが作成される DNS の `zone_id` になります。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` も必ず設定が必要です。モジュール内では必須入力となっており、上記の例では W&B インストールへのすべてのアクセスを許可しています。

3. `versions.tf` ファイルを作成します

   このファイルには、AWS 上で W&B をデプロイするために必要な Terraform および Terraform プロバイダーのバージョンを記述します。

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

   [Terraform 公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照して AWS プロバイダーの設定を行ってください。

   また、前述した [リモートバックエンドの設定](https://developer.hashicorp.com/terraform/language/backend) も追加することを強く推奨します。

4. `variables.tf` ファイルを作成します

   `terraform.tfvars` で設定したすべてのオプションには、それぞれの変数宣言が必要です。

   ```
   variable "namespace" {
     type        = string
     description = "リソースに使う名前のプレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "インスタンスへアクセスするためのドメイン名"
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UI へアクセスするためのサブドメイン"
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biases のサブドメインの作成先ドメイン"
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-server へのアクセスを許可する CIDR"
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server へのアクセスを許可する CIDR (IPv6)"
    nullable    = false
    type        = list(string)
   }

   variable "eks_cluster_version" {
    description = "EKS クラスターの Kubernetes バージョン"
    nullable    = false
    type        = string
   }
   ```

## 推奨デプロイメントオプション

これは、すべての「必須」コンポーネントを作成し、`Kubernetes クラスター`に最新バージョンの W&B をインストールする最もシンプルな構成例です。

1. `main.tf` を作成します

   `一般的な手順` で作成したファイルと同じディレクトリに、下記内容の `main.tf` ファイルを作成してください。

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

   W&B をデプロイするには、下記のコマンドを実行してください：

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 有効化

別のデプロイメントオプションでは、`Redis` を利用して SQL クエリをキャッシュし、実験のメトリクス読み込み時のアプリケーションレスポンスを高速化します。

キャッシュを有効にするには、[推奨デプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明した `main.tf` ファイルに `create_elasticache_subnet = true` オプションを追加します。

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

## メッセージブローカー（キュー）の有効化

3 番目のデプロイメントオプションは、外部の `message broker` を有効にするものです。W&B には組み込みのブローカーがあるため、このオプションは必須ではありません。また、パフォーマンス向上をもたらすものでもありません。

AWS でメッセージブローカーを提供するリソースは `SQS` で、これを有効にするには、[推奨デプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションの `main.tf` にオプション `use_internal_queue = false` を追加してください。

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

## その他のデプロイメントオプション

3 つのデプロイメントオプションすべてを同じファイルにまとめて組み合わせることも可能です。[Terraform Module](https://github.com/wandb/terraform-aws-wandb) には、標準オプションや `Deployment - Recommended` の最小構成と併用できる多くのオプションが用意されています。

## 手動設定

W&B のファイルストレージバックエンドとして Amazon S3 バケットを利用する場合、下記の作業が必要です：

* [Amazon S3 バケットおよびバケット通知の作成]({{< relref path="#create-an-s3-bucket-and-bucket-notifications" lang="ja" >}})
* [SQS キューの作成]({{< relref path="#create-an-sqs-queue" lang="ja" >}})
* [W&B を実行しているノードへの権限付与]({{< relref path="#grant-permissions-to-node-that-runs-wb" lang="ja" >}})

バケットの作成と、そのバケットからオブジェクト作成通知を受信するよう設定された SQS キューの作成が必要です。インスタンスにはこのキューから読み込みを行う権限が必要です。

### S3 バケットおよびバケット通知の作成

下記手順で Amazon S3 バケットを作成し、バケット通知を有効にします。

1. AWS コンソールで Amazon S3 へ移動します。
2. **Create bucket** を選択します。
3. **Advanced settings** 内の **Events** セクションで **Add notification** を選びます。
4. すべてのオブジェクト作成イベントが、先ほど設定した SQS キューに送信されるよう設定します。

{{< img src="/images/hosting/s3-notification.png" alt="Enterprise file storage settings" >}}

CORS アクセスを有効にしてください。CORS 設定は以下のようになります：

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

### SQS キューの作成

以下の手順で SQS キューを作成してください：

1. AWS コンソールで Amazon SQS へ移動します。
2. **Create queue** を選択します。
3. **Details** セクションで、**Standard** キュータイプを選びます。
4. Access policy セクションで、以下のプリンシパルに権限を追加します：
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

さらに必要に応じて、**Access Policy** セクションで詳細なアクセス権ポリシーを追加できます。例えば、Amazon SQS へのアクセスを許可するポリシー例は以下の通りです：

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

### W&B を実行しているノードへの権限付与

W&B サーバーの実行ノードには、Amazon S3 および Amazon SQS へのアクセスが許可されている必要があります。サーバーデプロイメントのタイプによっては、下記のようなポリシーステートメントをノードのロールに追加する必要があります：

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

### W&B サーバーの設定
最後に、W&B Server を設定します。

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B の設定ページへアクセスします。
2. ***外部ファイルストレージバックエンドを利用* オプションを有効にします。
3. 次の形式で Amazon S3 バケット、リージョン、Amazon SQS キューに関する情報を入力します：
* **File Storage Bucket**: `s3://<bucket-name>`
* **File Storage Region (AWS のみ)**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

{{< img src="/images/hosting/configure_file_store.png" alt="AWS file storage configuration" >}}

4. **Update settings** を選択し、新しい設定を反映させます。

## W&B バージョンのアップグレード

W&B をアップデートする手順は以下の通りです：

1. 設定で `wandb_app` モジュールの `wandb_version` を追加してください。アップグレードしたい W&B のバージョンを指定します。例えば、下記は W&B バージョン `0.48.1` を利用する例です：

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  {{% alert %}}
  もしくは、`wandb_version` を `terraform.tfvars` に追加し、同名の変数を作成した上で、リテラル値の代わりに `var.wandb_version` を使用することも可能です。
  {{% /alert %}}

2. 設定を更新したら、[推奨デプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明した手順を再度実施してください。

## AWS Terraform モジュールの operator ベースへの移行

このセクションでは、[terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) モジュールを利用して、_pre-operator_ 環境から _post-operator_ 環境へアップグレードするための手順を説明します。

{{% alert %}}
W&B のアーキテクチャー移行に伴い、Kubernetes の [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) パターンが必須となります。 詳細は [アーキテクチャーシフトの解説]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#reasons-for-the-architecture-shift" lang="ja" >}}) をご覧ください。
{{% /alert %}}


### 移行前後のアーキテクチャー

従来、W&B のアーキテクチャーでは次のように設定していました：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

このモジュールでインフラストラクチャーを管理していました：

{{< img src="/images/hosting/pre-operator-infra.svg" alt="pre-operator-infra" >}}

さらに下記モジュールを利用して W&B Server をデプロイ：

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

{{< img src="/images/hosting/pre-operator-k8s.svg" alt="pre-operator-k8s" >}}

移行後は、アーキテクチャーが次のようになります：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

このモジュールが、インフラストラクチャーの構築と Kubernetes クラスターへの W&B Server のインストールをまとめて管理するため、`post-operator.tf` で `module "wandb_app"` は不要になります。

{{< img src="/images/hosting/post-operator-k8s.svg" alt="post-operator-k8s" >}}

このアーキテクチャー変更により、OpenTelemetry、Prometheus、HPA、Kafka、イメージアップデートなどの追加機能を、SRE やインフラ担当者が手動で Terraform 操作を行わなくても利用できるようになります。

W&B Pre-Operator のベースインストールを始める場合、`post-operator.tf` の拡張子を `.disabled` にし、`pre-operator.tf` を有効（拡張子の変更なし）にします。これらのファイルは [こちら](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration) から確認できます。

### 前提条件

移行プロセスを開始する前に、以下を必ずご用意ください：

- **Egress**: デプロイメントは完全に閉じたネットワーク（airgapped）では実行できません。[deploy.wandb.ai](https://deploy.wandb.ai) へのアクセスが必要です（**_Release Channel_** 用スペックの取得に必要）。
- **AWS 認証情報**: 適切な AWS クレデンシャルが設定されていること。
- **Terraform のインストール**: 最新バージョンの Terraform がシステムにインストールされていること。
- **Route53 ホストゾーン**: アプリケーションを提供するドメインに対応する Route53 ホストゾーンが存在すること。
- **Pre-Operator 用 Terraform ファイル**: `pre-operator.tf` および `pre-operator.tfvars` などの変数ファイルが適切にセットアップされていること。

### Pre-Operator 構成のセットアップ

Terraform コマンドを使い、Pre-Operator の構成を初期化・適用してください：

```bash
terraform init -upgrade
terraform apply -var-file=./pre-operator.tfvars
```

`pre-operator.tf` 例：

```ini
namespace     = "operator-upgrade"
domain_name   = "sandbox-aws.wandb.ml"
zone_id       = "Z032246913CW32RVRY0WU"
subdomain     = "operator-upgrade"
wandb_license = "ey..."
wandb_version = "0.51.2"
```

この `pre-operator.tf` の設定は、2 つのモジュールを呼び出します：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

このモジュールでインフラ構築。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

このモジュールでアプリケーションをデプロイします。

### Post-Operator セットアップ

`pre-operator.tf` を `.disabled` 拡張子にし、`post-operator.tf` を有効化してください。

`post-operator.tfvars` には追加変数が含まれます：

```ini
...
# wandb_version = "0.51.2" は Release Channel または User Spec 内で管理されます。

# アップグレード用オペレータ必須変数：
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

次のコマンドで Post-Operator 設定を初期化・適用します：

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

plan/apply 後、下記リソースが更新されます：

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

適用時の画面例は下記の通りです：

{{< img src="/images/hosting/post-operator-apply.png" alt="post-operator-apply" >}}

`post-operator.tf` では、唯一必要なのは：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### post-operator 構成での主な変更点：

1. **プロバイダーの要件更新**: `required_providers.aws.version` を `3.6` から `4.0` へ変更し互換性を確保。
2. **DNS とロードバランサー設定**: `enable_dummy_dns` と `enable_operator_alb` を統合し、Ingress を使った DNS レコードや AWS Load Balancer 設定を管理。
3. **ライセンスとサイズの指定**: `license` および `size` パラメータを `wandb_infra` モジュールへ直接渡し、新仕様に対応。
4. **カスタムドメイン処理**: DNS トラブル対応時には `custom_domain_filter` でフィルタし、`kube-system` ネームスペースの外部 DNS pod のログをチェック可能。
5. **Helm プロバイダー設定**: Kubernetes リソースを効果的に管理できるよう、Helm プロバイダーを有効化・設定：

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

この包括的な構成により、Pre-Operator から Post-Operator へのスムーズな移行と、新しい operator モデルによる高効率な利活用が可能となります。