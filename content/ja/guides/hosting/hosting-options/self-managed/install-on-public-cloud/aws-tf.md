---
title: AWS に W&B プラットフォーム をデプロイする
description: AWS で W&B サーバー をホスティングする
menu:
  default:
    identifier: aws-tf
    parent: install-on-public-cloud
weight: 10
---

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) および [W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) のようなフルマネージドのデプロイメントオプションを推奨しています。W&B のフルマネージドサービスはシンプルかつセキュアであり、ほとんど設定を必要とせずご利用いただけます。
{{% /alert %}}

W&B では AWS 上でプラットフォームをデプロイするために [W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) の利用を推奨しています。

開始前に、Terraform の [State File](https://developer.hashicorp.com/terraform/language/state) を保存するための [remote backends](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択することをお勧めします。

State File は、全てのコンポーネントを作り直すことなく、アップグレードやデプロイメントの変更を行うために必要なリソースです。

この Terraform モジュールでは、以下の `必須` コンポーネントがデプロイされます:

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

他のデプロイメントオプションでは、以下のオプショナルな機能も追加できます:

- Elastic Cache for Redis
- SQS

## 事前に必要な権限

Terraform を実行するアカウントには、イントロダクションで記載した全てのコンポーネントを作成できる権限、および **IAM Policies** と **IAM Roles** を作成し、リソースへアタッチする権限が必要です。

## 一般的な手順

ここで紹介する手順は、本ドキュメントで取り上げている全てのデプロイメントオプションに共通です。

1. 開発環境の準備
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストール
   - バージョン管理のため、Gitリポジトリの作成を推奨します。
2. `terraform.tfvars` ファイルの作成

   `tvfars` ファイルの内容はインストールタイプに応じてカスタマイズできますが、最低限推奨される設定例は下記の通りです。

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

   デプロイ前に `tvfars` ファイル内で変数を必ず定義してください。 `namespace` 変数は Terraform で作成されるリソース名のプレフィックスとなる文字列です。

   `subdomain` と `domain` を組み合わせることで、W&B が設定される FQDN となります。上記例だと W&B の FQDN は `wandb-aws.wandb.ml` となり、その DNS レコードが作成される `zone_id` です。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` の設定も必要です。このモジュールでは必須入力項目となっています。例では、全てのソースから W&B へのアクセスを許可しています。

3. `versions.tf` ファイルの作成

   このファイルには、AWSでW&Bをデプロイするために必要なTerraform本体とプロバイダーのバージョンを記載します。

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

   AWS プロバイダーの設定は [Terraform公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) をご確認ください。

   オプションですが、先述の通り [remote backend configuration](https://developer.hashicorp.com/terraform/language/backend) の追加も強く推奨します。

4. `variables.tf` ファイルの作成

   `terraform.tfvars` で設定したそれぞれのオプションについて、対応する変数宣言が必要です。

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
    description = "wandb-server へのアクセス許可 CIDRs"
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server へのアクセス許可 CIDRs"
    nullable    = false
    type        = list(string)
   }

   variable "eks_cluster_version" {
    description = "EKS クラスターの Kubernetes バージョン"
    nullable    = false
    type        = string
   }
   ```

## 推奨されるデプロイメントオプション

これは最もシンプルな推奨デプロイメント構成で、全ての `必須` コンポーネントを作成し、`Kubernetes クラスター` 上へ最新の `W&B` をインストールします。

1. `main.tf` の作成

   「一般的な手順」で作成したファイルと同じディレクトリーに、下記内容の `main.tf` ファイルを作成します。

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

2. W&B のデプロイ

   W&B をデプロイするには、下記コマンドを実行してください。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS の有効化

もう一つのオプションとして、`Redis` を利用して SQL クエリをキャッシュし、実験のメトリクスをロードする際のアプリケーションレスポンスを高速化できます。

キャッシュ機能を有効にするには、上記[推奨デプロイメント]({{< relref "#recommended-deployment-option" >}}) セクションで説明したのと同じ `main.tf` ファイルに `create_elasticache_subnet = true` オプションを追加します。

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

## メッセージブローカー (キュー) の有効化

デプロイメントオプション3は、外部 `message broker` の有効化です。W&B には組み込みブローカーがあるため、これはオプションです。パフォーマンス向上が目的ではありません。

AWSで message broker を提供するリソースは `SQS` です。有効にするには、[推奨デプロイメント]({{< relref "#recommended-deployment-option" >}}) セクションの `main.tf` に `use_internal_queue = false` を追加してください。

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

全てのデプロイメントオプションをまとめてひとつのファイルで設定することも可能です。[Terraform Module](https://github.com/wandb/terraform-aws-wandb) では複数のオプションが提供されており、標準オプションや `Deployment - Recommended` の最小構成と組み合わせて利用できます。

## 手動設定

Amazon S3 バケットを W&B のファイルストレージバックエンドとして利用する場合、以下の手順が必要です。

* [Amazon S3 バケットとバケット通知の作成]({{< relref "#create-an-s3-bucket-and-bucket-notifications" >}})
* [SQS キューの作成]({{< relref "#create-an-sqs-queue" >}})
* [W&B を実行するノードへの権限付与]({{< relref "#grant-permissions-to-node-that-runs-wb" >}})

バケットを作成し、そのバケットからオブジェクト作成通知を受け取るよう SQS キューを設定する必要があります。さらに、インスタンスにはこのキューを読み取る権限が必要です。

### S3 バケットとバケット通知の作成

下記手順に従って、Amazon S3 バケットを作成し、通知を有効化します。

1. AWS コンソールで Amazon S3 へ移動。
2. **Create bucket** を選択。
3. **Advanced settings** 内の **Events** セクションで **Add notification** を選択。
4. 全てのオブジェクト作成イベントを、事前に設定した SQS Queue へ送信するよう設定します。

{{< img src="/images/hosting/s3-notification.png" alt="Enterprise file storage settings" >}}

CORS アクセスを有効にしてください。CORS 設定は下記のようになります。

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

下記手順に従って、SQS Queue を作成します。

1. AWS コンソールで Amazon SQS へ移動。
2. **Create queue** を選択。
3. **Details** セクションで **Standard** キュータイプを選択。
4. **Access policy** セクションで、下記権限を次の Principal に付与します。
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

必要に応じて、**Access Policy** セクションで追加のアクセス制御ポリシーを設定できます。例として、Amazon SQS へのアクセス用ポリシーは下記になります。

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

### W&B を実行するノードへの権限付与

W&B サーバーが動作しているノードには、Amazon S3 および Amazon SQS へのアクセスを許可する設定が必要です。デプロイメントのタイプにより、下記ポリシーステートメントをノードのロールに追加してください。

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
最後に、W&B サーバーを設定します。

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定画面へアクセスします。
2. ***Use an external file storage backend* オプションを有効化
3. 以下の形式で、Amazon S3 バケット名・リージョン・Amazon SQS キュー情報を入力します。
* **File Storage Bucket**: `s3://<bucket-name>`
* **File Storage Region (AWS のみ)**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

{{< img src="/images/hosting/configure_file_store.png" alt="AWS file storage configuration" >}}

4. **Update settings** を選択して設定を適用します。

## W&B バージョンのアップグレード

W&B をアップデートするには、下記手順に従ってください。

1. `wandb_version` を `wandb_app` モジュールの設定に追加します。アップグレードしたい W&B のバージョンを指定してください。例ではバージョン `0.48.1` を指定：

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  {{% alert %}}
  別案として、`wandb_version` を `terraform.tfvars` に追加し、同名変数を作成。リテラル値の代わりに `var.wandb_version` として利用することもできます。
  {{% /alert %}}

2. 設定を更新したら、[推奨デプロイメント]({{< relref "#recommended-deployment-option" >}}) セクションの手順を行ってください。

## オペレーター型 AWS Terraform モジュールへの移行

このセクションでは、[terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) モジュールを使って、_pre-operator_ 環境から _post-operator_ 環境へのアップグレード手順を詳しく解説します。

{{% alert %}}
Kubernetes [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) パターンへの移行は W&B のアーキテクチャー上不可欠です。詳細は [アーキテクチャー変更の理由]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/#reasons-for-the-architecture-shift" >}}) をご参照ください。
{{% /alert %}}

### 変更前後のアーキテクチャー

これまでの W&B アーキテクチャーでは：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

でインフラを管理していました。

{{< img src="/images/hosting/pre-operator-infra.svg" alt="pre-operator-infra" >}}

そして、以下のモジュールで W&B サーバーをデプロイ：

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

{{< img src="/images/hosting/pre-operator-k8s.svg" alt="pre-operator-k8s" >}}

移行後はアーキテクチャーが変更され：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

インフラ構築と Kubernetes クラスターへの W&B サーバーのデプロイを一括管理。`post-operator.tf` 内に `module "wandb_app"` は不要となります。

{{< img src="/images/hosting/post-operator-k8s.svg" alt="post-operator-k8s" >}}

このアーキテクチャー変更により（OpenTelemetry や Prometheus、HPA、Kafka、イメージアップデートなど）今まで手作業が必要だった機能が SRE/インフラストラクチャーチームによる Terraform オペレーション無しで利用可能になります。

W&B Pre-Operator のベースインストールを行うには、まず `post-operator.tf` に `.disabled` 拡張子を付け、`pre-operator.tf` が有効（`.disabled` がないファイル）になっていることを確認してください。これらのファイルは[こちら](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration)で確認できます。

### 前提条件

マイグレーションを始める前に、以下の前提条件を満たしてください。

- **Egress**: デプロイ先はエアギャップ化不可。最新の **_Release Channel_** スペック取得のため [deploy.wandb.ai](https://deploy.wandb.ai) へアクセスできる必要があります。
- **AWS認証情報**: AWS リソース操作が可能な認証情報が正しく設定されていること。
- **Terraformインストール済み**: お使いの端末に最新バージョンの Terraform がインストールされていること。
- **Route53 Hosted Zone**: アプリケーション公開用ドメインに対応する Route53 ホストゾーンが存在していること。
- **Pre-Operator用Terraformファイル**: `pre-operator.tf` および `pre-operator.tfvars` などの変数ファイルが正しく設定されていること。

### Pre-Operator セットアップ

Pre-Operator セットアップのため、下記のTerraformコマンドを実行します：

```bash
terraform init -upgrade
terraform apply -var-file=./pre-operator.tfvars
```

`pre-operator.tf` の例：

```ini
namespace     = "operator-upgrade"
domain_name   = "sandbox-aws.wandb.ml"
zone_id       = "Z032246913CW32RVRY0WU"
subdomain     = "operator-upgrade"
wandb_license = "ey..."
wandb_version = "0.51.2"
```

`pre-operator.tf` では2つのモジュールを呼び出します：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

このモジュールでインフラを構築。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

このモジュールでアプリケーションをデプロイ。

### Post-Operator セットアップ

`pre-operator.tf` には `.disabled` 拡張子を付け、`post-operator.tf` を有効化してください。

`post-operator.tfvars` には、追加の変数が必要です：

```ini
...
# wandb_version = "0.51.2" は Release Channel または User Spec で管理されます。

# Upgrade 用オペレーター変数（必須）:
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

Post-Operator 設定を初期化・実行するには、次のコマンドを実行します。

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

plan/apply ステップでは、下記リソースが更新されます。

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

このような画面が表示されます：

{{< img src="/images/hosting/post-operator-apply.png" alt="post-operator-apply" >}}

`post-operator.tf` では、以下の1行のみが存在します。

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### post-operator 設定での主な変更点

1. **必要プロバイダーのバージョンアップデート**: `required_providers.aws.version` を `3.6` から `4.0` へ変更し互換性を持たせます。
2. **DNSおよびロードバランサー設定**: `enable_dummy_dns` と `enable_operator_alb` を組み込み、Ingress 経由で DNS レコードと AWS ロードバランサーの設定を管理します。
3. **ライセンスとサイズの設定**: `license` と `size` のパラメータを `wandb_infra` モジュールへ直接渡して新しい運用要件に対応します。
4. **独自ドメイン処理**: DNS トラブルシュートが必要な場合、`custom_domain_filter` を利用し、`kube-system` namespace の External DNS pod ログを確認します。
5. **Helm provider 設定**: Kubernetes リソース管理のため Helm provider を有効化し、設定例は下記の通り。

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

この包括的なセットアップにより、Pre-Operator から Post-Operator へのスムーズな移行ができ、オペレーター型モデルでの新たな効率と機能を最大限に活用できます。