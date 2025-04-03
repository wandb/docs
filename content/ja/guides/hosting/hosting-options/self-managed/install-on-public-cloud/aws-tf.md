---
title: Deploy W&B Platform on AWS
description: AWS 上で W&B サーバー をホストする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-aws-tf
    parent: install-on-public-cloud
weight: 10
---

{{% alert %}}
Weights & Biases (W&B) では、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) など、フルマネージドのデプロイメントオプションをお勧めします。W&B のフルマネージドサービスは、シンプルで安全に使用でき、最小限の設定で済みます。
{{% /alert %}}

W&B は、[W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) を使用して、AWS にプラットフォームをデプロイすることをお勧めします。

開始する前に、W&B は、Terraform で利用可能な [リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択して、[State File](https://developer.hashicorp.com/terraform/language/state) を保存することをお勧めします。

State File は、すべてのコンポーネントを再作成せずに、アップグレードを展開したり、デプロイメントに変更を加えたりするために必要なリソースです。

Terraform Module は、次の `必須` コンポーネントをデプロイします。

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

その他のデプロイメントオプションには、次のオプションコンポーネントを含めることもできます。

- Redis 用 Elastic Cache
- SQS

## 前提条件のアクセス許可

Terraform を実行するアカウントは、イントロダクションで説明されているすべてのコンポーネントを作成でき、**IAM Policies** と **IAM Roles** を作成し、リソースにロールを割り当てるアクセス許可が必要です。

## 一般的な手順

このトピックの手順は、このドキュメントで説明されているすべてのデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
   - W&B は、バージョン管理のために Git リポジトリーを作成することをお勧めします。
2. `terraform.tfvars` ファイルを作成します。

   `tvfars` ファイルの内容は、インストールタイプに応じてカスタマイズできますが、最小限の推奨設定は以下の例のようになります。

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

   `namespace` 変数は、Terraform によって作成されたすべてのリソースのプレフィックスとなる文字列であるため、デプロイする前に `tvfars` ファイルで変数を定義してください。

   `subdomain` と `domain` の組み合わせで、W&B が設定される FQDN が形成されます。上記の例では、W&B FQDN は `wandb-aws.wandb.ml` になり、FQDN レコードが作成される DNS `zone_id` になります。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` の両方も設定が必要です。モジュールでは、これは必須入力です。上記の例では、すべてのソースからの W&B インストールへのアクセスを許可しています。

3. ファイル `versions.tf` を作成します。

   このファイルには、AWS に W&B をデプロイするために必要な Terraform および Terraform プロバイダーのバージョンが含まれます。

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

   AWS プロバイダーの設定については、[Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

   オプションですが、強く推奨されるのは、このドキュメントの冒頭で説明した [リモートバックエンド構成](https://developer.hashicorp.com/terraform/language/backend) を追加することです。

4. ファイル `variables.tf` を作成します。

   `terraform.tfvars` で設定されたすべてのオプションについて、Terraform は対応する変数宣言を必要とします。

   ```
   variable "namespace" {
     type        = string
     description = "リソースに使用される名前のプレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "インスタンスへのアクセスに使用されるドメイン名。"
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UI にアクセスするためのサブドメイン。"
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biases サブドメインを作成するドメイン。"
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-server へのアクセスが許可されている CIDR。"
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server へのアクセスが許可されている CIDR。"
    nullable    = false
    type        = list(string)
   }

   variable "eks_cluster_version" {
    description = "EKS クラスター kubernetes バージョン"
    nullable    = false
    type        = string
   }
   ```

## 推奨されるデプロイメントオプション

これは、すべての `必須` コンポーネントを作成し、`Kubernetes Cluster` に最新バージョンの `W&B` をインストールする、最も簡単なデプロイメントオプション構成です。

1. `main.tf` を作成します。

   「一般的な手順」でファイルを作成したのと同じディレクトリーに、次の内容で `main.tf` ファイルを作成します。

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

2. W&B をデプロイします。

   W&B をデプロイするには、次のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS を有効にする

別のデプロイメントオプションでは、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスをロードする際のアプリケーションの応答を高速化します。

キャッシュを有効にするには、[推奨されるデプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明されているのと同じ `main.tf` ファイルにオプション `create_elasticache_subnet = true` を追加する必要があります。

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

## メッセージブローカー（キュー）を有効にする

デプロイメントオプション 3 は、外部 `message broker` を有効にすることで構成されています。これは、W&B にブローカーが埋め込まれているため、オプションです。このオプションは、パフォーマンスの向上をもたらしません。

メッセージブローカーを提供する AWS リソースは `SQS` であり、これを有効にするには、[推奨されるデプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明されているのと同じ `main.tf` にオプション `use_internal_queue = false` を追加する必要があります。

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

3 つのデプロイメントオプションすべてを組み合わせて、すべての構成を同じファイルに追加できます。
[Terraform Module](https://github.com/wandb/terraform-aws-wandb) は、標準オプションと `Deployment - Recommended` にある最小構成とともに組み合わせることができるいくつかのオプションを提供します。

## 手動構成

Amazon S3 バケットを W&B のファイルストレージバックエンドとして使用するには、次の操作を行う必要があります。

* [Amazon S3 バケットとバケット通知の作成]({{< relref path="#create-an-s3-bucket-and-bucket-notifications" lang="ja" >}})
* [SQS キューの作成]({{< relref path="#create-an-sqs-queue" lang="ja" >}})
* [W&B を実行するノードへのアクセス許可の付与]({{< relref path="#grant-permissions-to-node-that-runs-wb" lang="ja" >}})

バケットと、そのバケットからオブジェクト作成通知を受信する SQS キューを構成する必要があります。インスタンスには、このキューから読み取るためのアクセス許可が必要です。

### S3 バケットとバケット通知の作成

Amazon S3 バケットを作成し、バケット通知を有効にするには、以下の手順に従います。

1. AWS コンソールで Amazon S3 に移動します。
2. [**バケットの作成**] を選択します。
3. [**詳細設定**] で、[**イベント**] セクションの [**通知の追加**] を選択します。
4. 以前に構成した SQS キューに送信されるように、すべてのオブジェクト作成イベントを構成します。

{{< img src="/images/hosting/s3-notification.png" alt="エンタープライズファイルストレージ設定" >}}

CORS アクセスを有効にします。CORS 構成は次のようになります。

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

SQS キューを作成するには、以下の手順に従います。

1. AWS コンソールで Amazon SQS に移動します。
2. [**キューの作成**] を選択します。
3. [**詳細**] セクションで、[**標準**] キュータイプを選択します。
4. [アクセス ポリシー] セクションで、次のプリンシパルへのアクセス許可を追加します。
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

オプションで、[**アクセス ポリシー**] セクションに高度なアクセス ポリシーを追加します。たとえば、ステートメントを含む Amazon SQS にアクセスするためのポリシーは次のとおりです。

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

### W&B を実行するノードへのアクセス許可の付与

W&B サーバーが実行されているノードは、Amazon S3 および Amazon SQS へのアクセスを許可するように構成する必要があります。選択したサーバーデプロイメントのタイプに応じて、次のポリシー ステートメントをノードロールに追加する必要がある場合があります。

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

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin` で W&B 設定ページに移動します。
2. [***外部ファイルストレージバックエンドを使用する*] オプションを有効にします。
3. 次の形式で、Amazon S3 バケット、リージョン、および Amazon SQS キューに関する情報を提供します。
* **ファイルストレージバケット**: `s3://<bucket-name>`
* **ファイルストレージリージョン (AWS のみ)**: `<region>`
* **通知サブスクリプション**: `sqs://<queue-name>`

{{< img src="/images/hosting/configure_file_store.png" alt="" >}}

4. [**設定の更新**] を選択して、新しい設定を適用します。

## W&B バージョンをアップグレードする

W&B を更新するには、ここに概説されている手順に従います。

1. `wandb_app` モジュールの構成に `wandb_version` を追加します。アップグレードする W&B のバージョンを指定します。たとえば、次の行は W&B バージョン `0.48.1` を指定します。

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  {{% alert %}}
  または、`wandb_version` を `terraform.tfvars` に追加し、同じ名前の変数を作成して、リテラル値を使用する代わりに `var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 構成を更新した後、[推奨されるデプロイメントセクション]({{< relref path="#recommended-deployment-option" lang="ja" >}}) で説明されている手順を完了します。

## オペレーターベースの AWS Terraform モジュールへの移行

このセクションでは、[terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) モジュールを使用して、_プレオペレーター_ 環境から _ポストオペレーター_ 環境にアップグレードするために必要な手順について詳しく説明します。

{{% alert %}}
W&B アーキテクチャでは、Kubernetes [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) パターンへの移行が必要です。アーキテクチャの移行に関する詳細な説明については、[このセクション]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#reasons-for-the-architecture-shift" lang="ja" >}}) を参照してください。
{{% /alert %}}

### 移行前後のアーキテクチャ

以前の W&B アーキテクチャでは、次のものが使用されていました。

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

インフラストラクチャを制御します。

{{< img src="/images/hosting/pre-operator-infra.svg" alt="プレオペレーターインフラストラクチャ" >}}

また、このモジュールを使用して W&B サーバーをデプロイします。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

{{< img src="/images/hosting/pre-operator-k8s.svg" alt="プレオペレーターk8s" >}}

移行後のアーキテクチャでは、次のものが使用されます。

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

インフラストラクチャのインストールと Kubernetes クラスターへの W&B サーバーのデプロイの両方を管理するため、`post-operator.tf` では `module "wandb_app"` は不要になります。

{{< img src="/images/hosting/post-operator-k8s.svg" alt="ポストオペレーターk8s" >}}

このアーキテクチャの移行により、SRE/インフラストラクチャチームによる手動の Terraform 操作を必要とせずに、追加機能 (OpenTelemetry、Prometheus、HPA、Kafka、およびイメージ更新など) を有効にできます。

W&B Pre-Operator の基本インストールを開始するには、`post-operator.tf` に `.disabled` ファイル拡張子があり、`pre-operator.tf` がアクティブであることを確認します (.disabled` 拡張子がない)。これらのファイルは、[ここ](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration) にあります。

### 前提条件

移行プロセスを開始する前に、次の前提条件が満たされていることを確認してください。

- **エグレス**: デプロイメントはエアギャップにできません。**_Release Channel_** の最新の仕様を取得するには、[deploy.wandb.ai](https://deploy.wandb.ai) へのアクセスが必要です。
- **AWS 認証情報**: AWS リソースと対話するように構成された適切な AWS 認証情報。
- **Terraform のインストール**: 最新バージョンの Terraform がシステムにインストールされている必要があります。
- **Route53 ホストゾーン**: アプリケーションが提供されるドメインに対応する既存の Route53 ホストゾーン。
- **Pre-Operator Terraform ファイル**: `pre-operator.tf` および関連する変数ファイル (例: `pre-operator.tfvars`) が正しく設定されていることを確認します。

### Pre-Operator の設定

次の Terraform コマンドを実行して、Pre-Operator セットアップの構成を初期化して適用します。

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

`pre-operator.tf` 構成は、次の 2 つのモジュールを呼び出します。

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

このモジュールはインフラストラクチャを起動します。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

このモジュールはアプリケーションをデプロイします。

### Post-Operator の設定

`pre-operator.tf` に `.disabled` 拡張子があり、`post-operator.tf` がアクティブであることを確認します。

`post-operator.tfvars` には、追加の変数が含まれています。

```ini
...
# wandb_version = "0.51.2" は、リリースチャネルを介して管理されるか、ユーザースペックで設定されるようになりました。

# アップグレードに必要なオペレーター変数:
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

次のコマンドを実行して、Post-Operator 構成を初期化して適用します。

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

プランと適用手順では、次のリソースが更新されます。

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

次のようなものが表示されます。

{{< img src="/images/hosting/post-operator-apply.png" alt="ポストオペレーター適用" >}}

`post-operator.tf` には、次のように 1 つがあります。

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### ポストオペレーター構成の変更点:

1. **必要なプロバイダーの更新**: プロバイダーの互換性を保つために、`required_providers.aws.version` を `3.6` から `4.0` に変更します。
2. **DNS とロード バランサーの構成**: Ingress を介して DNS レコードと AWS ロード バランサーの設定を管理するために、`enable_dummy_dns` と `enable_operator_alb` を統合します。
3. **ライセンスとサイズの構成**: 新しい運用要件に合わせて、`license` パラメーターと `size` パラメーターを `wandb_infra` モジュールに直接転送します。
4. **カスタムドメインの処理**: 必要に応じて、`kube-system` 名前空間内の外部 DNS ポッドログを確認して DNS の問題をトラブルシューティングするために、`custom_domain_filter` を使用します。
5. **Helm プロバイダーの構成**: Helm プロバイダーを有効にして構成し、Kubernetes リソースを効果的に管理します。

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

この包括的な設定により、オペレーターモデルによって有効になる新しい効率と機能を活用して、Pre-Operator 構成から Post-Operator 構成へのスムーズな移行が保証されます。
