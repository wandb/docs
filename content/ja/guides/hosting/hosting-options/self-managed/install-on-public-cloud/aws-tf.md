---
title: Deploy W&B Platform on AWS
description: AWS で W&B サーバー をホスティングする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-aws-tf
    parent: install-on-public-cloud
weight: 10
---

{{% alert %}}
W&B は、[W&B マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) もしくは [W&B 専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のような完全管理型のデプロイメントオプションを推奨します。W&B の完全管理サービスは、シンプルで安全に使用でき、ほとんど設定を必要としません。
{{% /alert %}}

W&B は、AWS にプラットフォームをデプロイするために [W&B サーバー AWS Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/aws/latest)を使用することを推奨します。

開始する前に、Terraform 用の [リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選んで、[ステートファイル](https://developer.hashicorp.com/terraform/language/state) を保存することを W&B では推奨しています。

ステートファイルは、すべてのコンポーネントを再作成することなく、アップグレードを展開したりデプロイメントに変更を加えたりするために必要なリソースです。

Terraform モジュールは、次の `必須` コンポーネントをデプロイします。

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

他のデプロイメントオプションには、次のオプションコンポーネントも含めることができます。

- Redis 用 Elastic Cache
- SQS

## 必要な権限

Terraform を実行するアカウントは、イントロダクションに記載されているすべてのコンポーネントを作成できることと、**IAM ポリシー**および**IAM ロール**を作成し、リソースにロールを割り当てる権限を持つ必要があります。

## 一般的な手順

このトピックの手順は、このドキュメントでカバーされている任意のデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストール
   - W&B はバージョン管理のために Git リポジトリを作成することを推奨します。
2. `terraform.tfvars` ファイルを作成します。

   `tfvars` ファイルの内容はインストールタイプに応じてカスタマイズできますが、最低限の推奨設定は次の例のようになります。

   ```bash
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   ```

   `namespace` 変数は Terraform によって作成されるすべてのリソースの接頭辞として使用される文字列であるため、デプロイする前に `tfvars` ファイルで変数を定義することを必ず行ってください。

   `subdomain` と `domain` の組み合わせにより、W&B が設定される FQDN が形成されます。上記の例では、W&B の FQDN は `wandb-aws.wandb.ml` となり、FQDN レコードが作成される DNS `zone_id` が設定されます。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` も設定が必要です。モジュールでは、これは入力必須事項です。以下の例では、W&B インストールへのアクセスを任意のソースから許可しています。

3. `versions.tf` ファイルを作成します。

   このファイルには、AWS で W&B をデプロイするために必要な Terraform と Terraform プロバイダーのバージョンが含まれます。

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

   AWS プロバイダーを設定するための情報は [Terraform 公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

   任意ではありますが、画して最初に言及された [リモートバックエンド設定](https://developer.hashicorp.com/terraform/language/backend) の追加は強く推奨されています。

4. `variables.tf` ファイルを作成します。

   `terraform.tfvars` で設定された各オプションに対して、Terraform は対応する変数の宣言を必要とします。

   ```
   variable "namespace" {
     type        = string
     description = "リソースに使用する名前の接頭辞"
   }

   variable "domain_name" {
     type        = string
     description = "インスタンスにアクセスするために使用されるドメイン名。"
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UI にアクセスするためのサブドメイン"
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biases サブドメインを作成するためのドメイン"
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-server へのアクセスを許可する CIDRs。"
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server へのアクセスを許可する CIDRs。"
    nullable    = false
    type        = list(string)
   }
   ```

## 推奨デプロイオプション

これは、すべての`必須`コンポーネントを作成し、`Kubernetes クラスター`に最新バージョンの`W&B`をインストールする最も簡単なデプロイオプション設定です。

1. `main.tf` を作成

   一般的な手順で作成したファイルと同じディレクトリー内に、次の内容を含む `main.tf` ファイルを作成します。

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

     # TF はワークグループを
     # スピンアップ中にデプロイしようとする
     depends_on = [module.wandb_infra]
   }

   output "bucket_name" {
     value = module.wandb_infra.bucket_name
   }

   output "url" {
     value = module.wandb_infra.url
   }
   ```

2. W&B をデプロイ

   W&B をデプロイするには、以下のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS を有効化

別のデプロイメントオプションでは、`Redis`を使用して SQL クエリをキャッシュし、実験のメトリクスを読み込む際のアプリケーション応答を高速化します。

キャッシュを有効にするには、[推奨デプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明したのと同じ `main.tf` ファイルに `create_elasticache_subnet = true` オプションを追加する必要があります。

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

## メッセージブローカー (キュー) を有効化

デプロイメントオプション 3 は、外部 `メッセージブローカー`を有効にすることから成ります。これは W&B に埋め込まれたブローカーを提供しているためオプションです。このオプションはパフォーマンスの向上をもたらしません。

メッセージブローカーを提供する AWS リソースは `SQS` であり、それを有効にするには、[推奨デプロイメント]({{< relref path="#recommended-deployment-option" lang="ja" >}}) セクションで説明されているのと同じ `main.tf` に `use_internal_queue = false` オプションを追加する必要があります。

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

## その他のデプロイオプション

すべてのデプロイオプションの設定を 1 つのファイルに追加して組み合わせることができます。
[Terraform モジュール](https://github.com/wandb/terraform-aws-wandb) は標準オプションと `デプロイメント - 推奨` に見られる最小設定とともに組み合わせることができるいくつかのオプションを提供します。

## 手動設定

ファイルストレージのバックエンドとして Amazon S3 バケットを使用するには、次のことを行う必要があります。

* [Amazon S3 バケットとバケット通知を作成する]({{< relref path="#create-an-s3-bucket-and-bucket-notifications" lang="ja" >}})
* [SQS キューを作成する]({{< relref path="#create-an-sqs-queue" lang="ja" >}})
* [W&Bを実行するノードに権限を付与する]({{< relref path="#grant-permissions-to-node-that-runs-wb" lang="ja" >}})


バケットを作成し、オブジェクト作成通知をそのバケットから受け取るように設定された SQS キューを作成する必要があります。 インスタンスには、このキューから読み取る権限が必要です。

### S3 バケットとバケット通知を作成する

Amazon S3 バケットを作成し、バケット通知を有効にする手順を以下に示します。

1. AWS コンソールで Amazon S3 に移動します。
2. **Create bucket** を選択します。
3. **詳細設定** の中で、**イベント** セクションの中で **通知を追加** を選択します。
4. すべてのオブジェクト作成イベントを、先に設定した SQS キューに送信するよう設定します。

{{< img src="/images/hosting/s3-notification.png" alt="エンタープライズファイルストレージ設定" >}}

CORS アクセスを有効にします。あなたの CORS 設定は次のようになります。

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

SQS キューを作成する方法に従います：

1. AWS コンソールで Amazon SQS に移動します。
2. **Create queue** を選択します。
3. **詳細** セクションで、**標準** キュータイプを選択します。
4. アクセスポリシーセクションで、次のプリンシパルに権限を追加します：
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

**アクセスポリシー** セクションで、高度なアクセス ポリシーをオプションで追加します。 たとえば、statement と共に Amazon SQS にアクセスするためのポリシーは次のようになります。

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

### W&B を実行するノードに権限を付与する

W&B サーバーが実行されているノードは、Amazon S3 および Amazon SQS へのアクセスを許可するように設定する必要があります。どのサーバーデプロイタイプを選択したかによって、ノードロールに次のポリシーステートメントを追加する必要がある場合があります。

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
最後に、W&B サーバーを設定します。

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin` で W&B 設定ページに移動します。
2. **外部ファイルストレージバックエンドを使用する** オプション を有効にします
3. 次の形式で Amazon S3 バケット、リージョン、および Amazon SQS キューに関する情報を提供します：
* **ファイルストレージ バケット**: `s3://<bucket-name>`
* **ファイル ストレージ リージョン (AWS のみ)**: `<region>`
* **通知サブスクリプション**: `sqs://<queue-name>`

{{< img src="/images/hosting/configure_file_store.png" alt="" >}}

4. **設定の更新** を選択して新しい設定を適用します。

## W&B バージョンをアップグレードする

W&B を更新するための手順は以下の通りです：

1. `wandb_app` モジュールの設定に `wandb_version` を追加します。アップグレードする W&B のバージョンを指定します。たとえば、次の行は W&B バージョン `0.48.1` を指定します：

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

{{% alert %}}
  また、`wandb_version` を `terraform.tfvars` に追加し、同じ名前の変数を作成して、リテラル値の代わりに `var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 設定を更新した後、[推奨デプロイメントセクション]({{< relref path="#recommended-deployment-option" lang="ja" >}}) で説明されている手順を完了します。

## オペレーターベースの AWS Terraform モジュールに移行する

このセクションでは、[terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) モジュールを使用して、_operator 以前の_ 環境から _operator 以降の_ 環境にアップグレードするために必要な手順を説明します。

{{% alert %}}
Kubernetes [オペレーター](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) パターンへの移行は、W&B アーキテクチャにとって必要です。アーキテクチャのシフトの詳細な説明については、[このセクション]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#reasons-for-the-architecture-shift" lang="ja" >}}) を参照してください。
{{% /alert %}}

### アーキテクチャの前後

以前は、W&B アーキテクチャは以下を使用していました：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

インフラを制御するために：

{{< img src="/images/hosting/pre-operator-infra.svg" alt="pre-operator-infra" >}}

そして、このモジュールを使用して W&B サーバーをデプロイしていました：

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

{{< img src="/images/hosting/pre-operator-k8s.svg" alt="pre-operator-k8s" >}}

移行後のアーキテクチャは、次を利用します：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

インフラと Kubernetes クラスターへの W&B サーバーのインストールの両方を管理し、それによって `post-operator.tf` の `module "wandb_app"` の必要性を排除します。

{{< img src="/images/hosting/post-operator-k8s.svg" alt="post-operator-k8s" >}}

このアーキテクチャのシフトにより、OpenTelemetry、Prometheus、HPAs、Kafka、イメージの更新などの追加機能が手動の Terraform 操作を SRE/インフラストラクチャチームによって要求されることなく利用可能になりました。

W&B プレオペレーターの基本インストールを開始するには、`post-operator.tf` に `.disabled` ファイル拡張子があり、`pre-operator.tf` がアクティブであることを確認してください（`.disabled` 拡張子がない場合）。これらのファイルは [ここ](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration) にあります。

### 前提条件

移行プロセスを開始する前に、次の前提条件が満たされていることを確認してください：

- **エグレス**: デプロイメントはエアギャップされていることはできません。最新の **_リリースチャネル_** の仕様を取得するために [deploy.wandb.ai](https://deploy.wandb.ai) へのアクセスが必要です。
- **AWS クレデンシャル**: AWS リソースとやり取りするために、適切に構成された AWS クレデンシャル。
- **Terraform がインストールされていること**: テラフォームの最新バージョンがシステムにインストールされていること。
- **Route53 ホストゾーン**: アプリケーションが提供されるドメインに対応する既存の Route53 ホストゾーン。
- **Pre-Operator Terraform ファイル**: `pre-operator.tf` と `pre-operator.tfvars` などの関連変数ファイルが正しく設定されていることを確認する。

### プレオペレーター セットアップ

次の Terraform コマンドを実行して、プレオペレーター セットアップの構成を初期化し、適用します：

```bash
terraform init -upgrade
terraform apply -var-file=./pre-operator.tfvars
```

`pre-operator.tf` は、次のようになります：

```ini
namespace     = "operator-upgrade"
domain_name   = "sandbox-aws.wandb.ml"
zone_id       = "Z032246913CW32RVRY0WU"
subdomain     = "operator-upgrade"
wandb_license = "ey..."
wandb_version = "0.51.2"
```

`pre-operator.tf` 構成は 2 つのモジュールを呼び出します：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

このモジュールはインフラストラクチャをスピンアップします。

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

このモジュールはアプリケーションをデプロイします。

### ポストオペレーターの設定

`pre-operator.tf` に `.disabled` 拡張子があり、`post-operator.tf` がアクティブであることを確認してください。

`post-operator.tfvars` には追加の変数が含まれています：

```ini
...
# wandb_version = "0.51.2" はリリース チャネル経由またはユーザー仕様で管理されます。

# アップグレードのために必要なオペレータ変数：
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

次のコマンドを実行して、ポストオペレーター設定を初期化し、適用します：

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

計画と適用ステップは、次のリソースを更新します：

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

次のような出力が表示されるはずです：

{{< img src="/images/hosting/post-operator-apply.png" alt="post-operator-apply" >}}

`post-operator.tf` では、次のものしかありません：

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### ポストオペレーター構成の変更点：

1. **必要なプロバイダーの更新**: プロバイダーの互換性のために、`required_providers.aws.version` を `3.6` から `4.0` に変更します。
2. **DNS およびロードバランサーの構成**: `enable_dummy_dns` と `enable_operator_alb` を統合して、DNS レコードと AWS ロードバランサー設定を Ingress を通じて管理します。
3. **ライセンスおよびサイズ構成**: `license` および `size` パラメーターを新しい操作要件に合わせて `wandb_infra` モジュールに直接転送します。
4. **カスタムドメイン処理**: 必要に応じて、DNS 問題をトラブルシューティングするために `custom_domain_filter` を使用し、`kube-system` 名前空間内の External DNS pod ログを確認します。
5. **Helm プロバイダー構成**: Kubernetes リソースを効果的に管理するために Helm プロバイダーを有効にし、構成：

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

この包括的なセットアップにより、Pre-Operator から Post-Operator 構成へのスムーズな移行が確実に行われ、オペレーターモデルによって可能になった新しい効率性と機能が活用されます。