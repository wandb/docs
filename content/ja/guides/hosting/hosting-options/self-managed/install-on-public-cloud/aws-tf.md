---
title: AWS に W&B プラットフォーム をデプロイ
description: W&B サーバーの AWS 上でのホスティング。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-aws-tf
    parent: install-on-public-cloud
weight: 10
---

{{% alert %}}
W&B は [W&B マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [W&B 専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) デプロイメントタイプのようなフルマネージドデプロイメントオプションを推奨しています。 W&B のフルマネージドサービスはシンプルで安全に利用でき、設定が最小限または不要です。
{{% /alert %}}

W&B は AWS 上にプラットフォームをデプロイするために [W&B サーバー AWS Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/aws/latest) の使用を推奨しています。

始める前に、W&B は Terraform が [状態ファイル](https://developer.hashicorp.com/terraform/language/state) を保存するための利用可能な[リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend) を選択することを推奨します。

状態ファイルは、全てのコンポーネントを再作成せずに、アップグレードを展開したりデプロイメントに変更を加えるために必要なリソースです。

Terraform モジュールは以下の `必須` コンポーネントをデプロイします：

- ロードバランサー
- AWS アイデンティティ & アクセスマネジメント (IAM)
- AWS キーマネジメントシステム (KMS)
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager (ACM)
- Amazon Elastic Load Balancing (ALB)
- Amazon Secrets Manager

他のデプロイメントオプションには、以下のオプションコンポーネントを含めることもできます：

- Redis 用のエラスティックキャッシュ
- SQS

## 前提条件の許可

Terraform を実行するアカウントは、イントロダクションで説明されたすべてのコンポーネントを作成できる必要があり、**IAM ポリシー** と **IAM ロール** を作成し、リソースにロールを割り当てる許可が必要です。

## 一般的なステップ

このトピックのステップは、このドキュメントでカバーされる任意のデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
   - W&B はバージョンコントロール用の Git リポジトリを作成することを推奨します。
2. `terraform.tfvars` ファイルを作成します。

   `tfvars` ファイルの内容はインストールタイプに応じてカスタマイズできますが、推奨される最低限の内容は以下の例のようになります。

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

   変数をデプロイ前に `tfvars` ファイルで定義してください。`namespace` 変数は Terraform によって作成される全てのリソースのプレフィックスとして使用される文字列です。

   `subdomain` と `domain` の組み合わせにより W&B が設定される FQDN が形成されます。上記の例では、W&B の FQDN は `wandb-aws.wandb.ml` となり、FQDN 記録が作成される DNS `zone_id` になります。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` も設定が必要です。このモジュールでは、これは必須の入力です。進行例では、W&B インストールへのアクセスを任意のソースから許可します。

3. `versions.tf` ファイルを作成します。

   このファイルは、AWS に W&B をデプロイするために必要な Terraformおよび Terraform プロバイダーのバージョンを含むものとします。

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

   AWS プロバイダーを設定するには [Terraform 公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)を参照してください。

   オプションですが強く推奨されるのは、このドキュメントの最初で触れられている[リモートバックエンド設定](https://developer.hashicorp.com/terraform/language/backend)を追加することです。

4. `variables.tf` ファイルを作成します。

   `terraform.tfvars` で設定されたオプションごとに、Terraform は対応する変数宣言を必要とします。

   ```
   variable "namespace" {
     type        = string
     description = "リソースに使用される名前のプレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "インスタンスにアクセスするために使用されるドメイン名。"
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
     description = "Weights & Biases サブドメインを作成するためのドメイン。"
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-server にアクセスを許可される CIDR。"
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server にアクセスを許可される CIDR。"
    nullable    = false
    type        = list(string)
   }

   variable "eks_cluster_version" {
    description = "EKS クラスター用の Kubernetes バージョン"
    nullable    = false
    type        = string
   }
   ```

## 推奨されるデプロイメントオプション

これは、全ての `必須` コンポーネントを作成し、最新バージョンの `W&B` を `Kubernetes クラスター` にインストールする最も簡単なデプロイメントオプションの設定です。

1. `main.tf` を作成します。

   `一般的なステップ` で作成したファイルと同じディレクトリに、以下の内容を持つ `main.tf` ファイルを作成してください。

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

   W&B をデプロイするには、以下のコマンドを実行してください：

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS を有効にする

別のデプロイメントオプションでは、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスを読み込む際のアプリケーションの応答をスピードアップさせます。

キャッシュを有効にするには、推奨されるデプロイメント [Recommended deployment]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明されている同じ `main.tf` ファイルに `create_elasticache_subnet = true` オプションを追加する必要があります。

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

デプロイメントオプション3は、外部の `メッセージブローカー` を有効にすることを目的としています。これはオプションですが、W&B 内にブローカーが埋め込まれているため、これによってパフォーマンスが向上するわけではありません。

AWS リソースが提供するメッセージブローカーは `SQS` です。これを有効にするには、推奨されるデプロイメント [Recommended deployment]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明されている同じ `main.tf` に `use_internal_queue = false`オプションを追加する必要があります。

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

同じファイルにすべての設定を追加することで、これらの3つのデプロイメントオプションを組み合わせることができます。 [Terraform Module](https://github.com/wandb/terraform-aws-wandb) は、標準オプションと `デプロイメント - 推奨` に見つかる最小限の構成と共に組み合わせることができるいくつかのオプションを提供します。

## 手動設定

Amazon S3 バケットを W&B のファイルストレージバックエンドとして使用する場合は：

* [S3 バケットとバケット通知の作成]({{< relref path="#create-an-s3-bucket-and-bucket-notifications" lang="ja" >}})
* [SQS キューの作成]({{< relref path="#create-an-sqs-queue" lang="ja" >}})
* [W&B を実行するノードへの権限付与]({{< relref path="#grant-permissions-to-node-that-runs-wb" lang="ja" >}})

バケットと、バケットからのオブジェクト作成通知を受け取るように設定された SQS キューを作成する必要があります。インスタンスにはこのキューを読み取る権限が必要です。

### S3 バケットとバケット通知の作成

以下の手順を実行して Amazon S3 バケットを作成し、バケット通知を有効化します。

1. AWS コンソールの Amazon S3 に移動します。
2. **バケットを作成** を選択します。
3. **詳細設定** の中で、**イベント** セクション内の **通知を追加** を選択します。
4. すべてのオブジェクト作成イベントを、先に設定した SQS キューに送信するように構成します。

{{< img src="/images/hosting/s3-notification.png" alt="エンタープライズファイルストレージ設定" >}}

CORS アクセスを有効にします。あなたの CORS 設定は以下のようになるはずです：

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

以下の手順に従って SQS キューを作成します：

1. AWS コンソールの Amazon SQS に移動します。
2. **キューの作成** を選択します。
3. **詳細** セクションから **標準** キュータイプを選択します。
4. アクセスポリシーセクション内で、以下の主体に許可を追加します：
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

また、**アクセスポリシー** セクションで、高度なアクセスポリシーを追加することもできます。例えば、Amazon SQS へのアクセスを声明するポリシーは以下のようになります：

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

W&B サーバーが実行されているノードは、Amazon S3 および Amazon SQS へのアクセスを許可するように設定されている必要があります。選択したサーバーデプロイメントの種類に応じて、以下のポリシーステートメントをノードロールに追加する必要があります：

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

1. W&B 設定ページに移動: `http(s)://YOUR-W&B-SERVER-HOST/system-admin`. 
2. ***外部ファイルストレージバックエンド使用* オプションを有効化
3. 以下の形式であなたの Amazon S3 バケット、リージョン、および Amazon SQS キューに関する情報を提供します：
* **ファイルストレージバケット**: `s3://<bucket-name>`
* **ファイルストレージリージョン (AWS のみ)**: `<region>`
* **通知サブスクリプション**: `sqs://<queue-name>`

{{< img src="/images/hosting/configure_file_store.png" alt="" >}}

4. **設定の更新** を選択して新しい設定を適用します。

## W&B のバージョンをアップグレードする

W&B を更新するための手順をここに従ってください：

1. `wandb_app` モジュール内の設定に `wandb_version` を追加します。アップグレード先の W&B のバージョンを指定します。例えば、次の行は W&B バージョン `0.48.1` を指定します：

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  {{% alert %}}
  または、`wandb_version` を `terraform.tfvars` に追加して、同じ名前の変数を作成し、リテラル値の代わりに `var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 設定を更新したら、[推奨デプロイメントセクション]({{< relref path="#recommended-deployment-option" lang="ja" >}})で説明されている手順を完了します。

## オペレーターに基づくAWS Terraformモジュールへの移行

このセクションは、[terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) モジュールを使用して、_プレオペレーター_ 環境から _ポストオペレーター_ 環境へのアップグレードに必要な手順を詳細に説明します。

{{% alert %}}
Kubernetes [オペレーター](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) パターンへの移行は、W&B アーキテクチャーにとって必要です。アーキテクチャー変更の詳細な説明については[このセクション]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#reasons-for-the-architecture-shift" lang="ja" >}})を参照してください。
{{% /alert %}}


### アーキテクチャーのビフォーアフター

以前は、W&B アーキテクチャは以下のように使用されていました：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

インフラストラクチャーを管理するために

{{< img src="/images/hosting/pre-operator-infra.svg" alt="pre-operator-infra" >}}

そしてこのモジュールで W&B サーバーをデプロイしていました：

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

{{< img src="/images/hosting/pre-operator-k8s.svg" alt="pre-operator-k8s" >}}

移行後、アーキテクチャーは以下のように使用されます：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

これにより、インフラストラクチャーと W&B サーバーの Kubernetes クラスターへのインストールの両方を管理し、`post-operator.tf` で `module "wandb_app"` は不要となります。

{{< img src="/images/hosting/post-operator-k8s.svg" alt="post-operator-k8s" >}}

このアーキテクチャーの変更により、OpenTelemetry、Prometheus、HPAs、Kafka、およびイメージの更新などの追加機能を、SRE/インフラストラクチャーチームによる手動の Terraform 操作なしで使用できるようになります。

W&B プレオペレーターの基本インストールを開始するには、`post-operator.tf` に `.disabled` ファイル拡張子が付いていることを確認し、`pre-operator.tf` が有効であることを確認してください（`.disabled` 拡張子が付いていないもの）。これらのファイルは[こちら](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration)で確認できます。

### 前提条件

移行プロセスを開始する前に、次の前提条件が満たされていることを確認してください：

- **アウトゴーイング接続**: デプロイメントはエアギャップされていない必要があります。**_リリース チャンネル_** の最新の仕様を取得するために [deploy.wandb.ai](https://deploy.wandb.ai) へのアクセスが必要です。
- **AWS 資格情報**: AWS リソースと対話するために適切に構成された AWS 資格情報が必要です。
- **Terraform のインストール**: 最新バージョンの Terraform がシステムにインストールされている必要があります。
- **Route53 ホステッドゾーン**: アプリケーションが提供されるドメインに対応した既存の Route53 ホステッドゾーン。
- **プレオペレーターTerraformファイル**: `pre-operator.tf` と `pre-operator.tfvars` のような関連変数ファイルが正しく設定されていることを確認してください。

### プリアペレーター セットアップ

プレオペレーター設定の構成を初期化および適用するには、次の Terraform コマンドを実行します：

```bash
terraform init -upgrade
terraform apply -var-file=./pre-operator.tfvars
```

`pre-operator.tf` は次のようになっています：

```ini
namespace     = "operator-upgrade"
domain_name   = "sandbox-aws.wandb.ml"
zone_id       = "Z032246913CW32RVRY0WU"
subdomain     = "operator-upgrade"
wandb_license = "ey..."
wandb_version = "0.51.2"
```

`pre-operator.tf` の構成は二つのモジュールを呼び出します：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

このモジュールはインフラストラクチャーを起動します。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

このモジュールはアプリケーションをデプロイします。

### ポストオペレーター設定

`pre-operator.tf` に `.disabled` 拡張子が付いていること、そして `post-operator.tf` がアクティブであることを確認してください。

`post-operator.tfvars` には追加の変数が含まれています：

```ini
...
# wandb_version = "0.51.2" はリリースチャンネル経由で管理されるか、ユーザースペックで設定されます。

# アップグレードのための必須オペレーター変数：
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

以下のコマンドを実行してポストオペレーター設定を初期化および適用します：

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

計画および適用手順は、次のリソースを更新します：

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

以下のような結果が表示されるはずです：

{{< img src="/images/hosting/post-operator-apply.png" alt="post-operator-apply" >}}

`post-operator.tf` では一つの以下があります：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### ポストオペレーター構成の変更：

1. **必要なプロバイダーの更新**: `required_providers.aws.version` を `3.6` から `4.0` に変更し、プロバイダー互換性を確保します。
2. **DNS およびロードバランサーの設定**: `enable_dummy_dns` および `enable_operator_alb` を統合して、DNS レコードおよび AWS ロードバランサー設定を Ingress 経由で管理します。
3. **ライセンスおよびサイズ構成**: 新しいオペレーション要件に合わせて、`license` および `size` パラメーターを直接 `wandb_infra` モジュールに転送します。
4. **カスタムドメインの処理**: 必要に応じて、`custom_domain_filter` を使用して `kube-system` 名前空間内の外部 DNS ポッドログをチェックし、DNS 問題のトラブルシューティングを行います。
5. **Helmプロバイダー構成**: 効果的に Kubernetes リソースを管理するためにHelm プロバイダーを有効にし、構成します：

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

この包括的なセットアップにより、オペレーターモデルによって可能になった新しい効率性と機能を活用しながら、プレオペレーターからポストオペレーター構成への円滑な移行が可能になります。