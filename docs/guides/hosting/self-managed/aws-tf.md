---
title: AWS
description: AWS での W&B サーバー のホスティング
displayed_sidebar: default
---

# AWS

:::info
Weights & Biases は、[W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) や [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) などのフルマネージドデプロイメントオプションを推奨しています。Weights & Biases のフルマネージドサービスは、設定がほとんど不要でシンプルかつ安全に使用できます。
:::

Weights & Biases は、AWS 上でプラットフォームをデプロイするために [W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) の使用を推奨しています。

モジュールのドキュメントは非常に充実しており、利用可能なすべてのオプションが含まれています。このドキュメントでは、いくつかのデプロイメントオプションについて説明します。

始める前に、Terraform の [State File](https://developer.hashicorp.com/terraform/language/state) を保存するために、利用可能な [remote backends](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) のいずれかを選択することをお勧めします。

State File は、コンポーネント全体を再作成することなく、アップグレードをロールアウトしたり、デプロイメントに変更を加えたりするために必要なリソースです。

Terraform Module は、次の `mandatory` コンポーネントをデプロイします:

- Load Balancer
- AWS Identity & Access Management (IAM)
- AWS Key Management System (KMS)
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager (ACM)
- Amazon Elastic Loadbalancing (ALB)
- Amazon Secrets Manager

その他のデプロイメントオプションには、次のオプションコンポーネントを含めることもできます:

- Elastic Cache for Redis
- SQS

## **必要な権限**

Terraform を実行するアカウントは、イントロダクションで説明したすべてのコンポーネントを作成できる必要があり、**IAM Policies** および **IAM Roles** を作成し、ロールをリソースに割り当てる権限を持っている必要があります。

## 一連の手順

このトピックの手順は、このドキュメントで取り上げるすべてのデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします
   - コードを保持する Git リポジトリーを作成することをお勧めしますが、ローカルにファイルを保持することもできます。
2. `terraform.tfvars` ファイルを作成します。

   `tvfars` ファイルの内容は、インストールタイプに応じてカスタマイズできますが、最低限の推奨は以下の例のようになります。

   ```bash
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   ```

   デプロイする前に `tvfars` ファイルに変数を定義してください。`namespace` 変数は、Terraform によって作成されるすべてのリソースの接頭辞として使われる文字列です。

   `subdomain` と `domain` の組み合わせは、Weights & Biases が構成される FQDN を形成します。上記の例では、Weights & Biases の FQDN は `wandb-aws.wandb.ml` となり、FQDN レコードが作成される DNS `zone_id` となります。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` も設定が必要です。このモジュールでは、これは必須の入力項目です。以下の例では、任意のソースから Weights & Biases インストールへのアクセスを許可しています。

3. `versions.tf` ファイルを作成します

   このファイルには、AWS に Weights & Biases をデプロイするために必要な Terraform および Terraform プロバイダーのバージョンが含まれます。

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

   AWS プロバイダーを設定するには、[Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

   オプションですが非常に推奨されるのが、ドキュメントの最初で言及された [remote backend configuration](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) を追加することです。

4. `variables.tf` ファイルを作成します

   `terraform.tfvars` で設定されているすべてのオプションには、対応する変数の宣言が必要です。

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
   ```

## デプロイメント - 推奨 (~20 分)

これは、すべての必須コンポーネントを作成し、`Kubernetes クラスター` に最新バージョンの Weights & Biases をインストールする、最も簡単なデプロイメントオプションの設定です。

1. `main.tf` を作成します

   `General Steps` でファイルを作成したのと同じディレクトリーに `main.tf` ファイルを以下の内容で作成します:

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

     # If we dont wait, tf will start trying to deploy while the work group is
     # still spinning up
     depends_on = [module.wandb_infra]
   }

   output "bucket_name" {
     value = module.wandb_infra.bucket_name
   }

   output "url" {
     value = module.wandb_infra.url
   }
   ```

2. Weights & Biases のデプロイ

   Weights & Biases をデプロイするには、以下のコマンドを実行します:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS の有効化

他のデプロイメントオプションでは、SQL クエリをキャッシュして、実験のメトリクスをロードするときのアプリケーション応答速度を向上させるために `Redis` を使用します。

キャッシュを有効にするために、`main.tf` ファイルにオプション `create_elasticache_subnet = true` を追加する必要があります。

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

## メッセージブローカー (キュー) の有効化

デプロイメントオプション 3 では、外部 `message broker` の有効化が含まれます。これはオプションであり、Weights & Biases は埋め込みブローカーを備えています。このオプションはパフォーマンスの向上をもたらしません。

メッセージブローカーを提供する AWS リソースは `SQS` であり、有効にするには、`Recommended Deployment` で作業した `main.tf` に `use_internal_queue = false` オプションを追加する必要があります。

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

## 他のデプロイメントオプション

すべての構成を同じファイルに追加することで、3 つのデプロイメントオプションを組み合わせることができます。
[Terraform Module](https://github.com/wandb/terraform-aws-wandb) は、標準オプションおよび `デプロイメント - 推奨` で見つかる最小構成と一緒に組み合わせることができる複数のオプションを提供します。

## 手動設定

Amazon S3 Bucket を Weights & Biases 用のファイルストレージバックエンドとして使用するには、以下を実行する必要があります:

* [Amazon S3 Bucket とバケット通知の作成](#create-an-s3-bucket-and-bucket-notifications)
* [SQS キューの作成](#create-an-sqs-queue)
* [Weights & Biases 実行ノードへの権限付与](#grant-permissions-to-node-running-wb)

バケットを作成し、そのバケットからオブジェクト作成通知を受信するように構成された SQS キューを作成する必要があります。インスタンスは、このキューから読み取る権限を持っている必要があります。

### S3 Bucket とバケット通知の作成

以下の手順に従って、Amazon S3 バケットを作成し、バケット通知を有効化します。

1. AWS コンソールで Amazon S3 に移動します。
2. **Create bucket** を選択します。
3. **Advanced settings** 内の **Events** セクションで、**Add notification** を選択します。
4. すべてのオブジェクト作成イベントを、以前に設定した SQS キューに送信するように設定します。

![Enterprise file storage settings](/images/hosting/s3-notification.png)

CORS アクセスを有効にします。CORS 設定は以下のようになります:

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

以下の手順に従って、SQS キューを作成します:

1. AWS コンソールで Amazon SQS に移動します。
2. **Create queue** を選択します。
3. **Details** セクションから、**Standard** キュータイプを選択します。
4. アクセスポリシーセクションで、次のプリンシパルに権限を追加します:
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

オプションとして、**Access Policy** セクションで高度なアクセスポリシーを追加します。例えば、次のようなステートメントを含むポリシーです。

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

### Weights & Biases 実行ノードへの権限付与

Weights & Biases サーバーが実行されているノードは、Amazon S3 と Amazon SQS へのアクセスを許可するように構成する必要があります。選択したサーバーデプロイメントのタイプによっては、以下のポリシーステートメントをノード役割に追加する必要があるかもしれません:

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

### Weights & Biases サーバーの設定
Finally, configure your W&B Server.

1. Weights & Biases の設定ページに移動します。URL は `http(s)://YOUR-W&B-SERVER-HOST/system-admin` です。 
2. ***Use an external file storage backend** オプションを有効にします。
3. 次の形式で、Amazon S3 バケット、リージョン、および Amazon SQS キューに関する情報を提供します:
* **File Storage Bucket**: `s3://<bucket-name>`
* **File Storage Region (AWS only)**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

![](/images/hosting/configure_file_store.png)

4. **Update settings** を選択して、新しい設定を適用します。

## Weights & Biases バージョンのアップグレード

Weights & Biases を更新する手順は以下の通りです:

1. `wandb_app` モジュールの設定に `wandb_version` を追加します。アップグレードしたい Weights & Biases のバージョンを指定します。例えば、次の行は Weights & Biases バージョン `0.48.1` を指定しています。

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  :::info
  または、`terraform.tfvars` に `wandb_version` を追加し、同じ名前の変数を作成して、リテラル値の代わりに `var.wandb_version` を使用することもできます。
  :::

2. 設定を更新した後、[デプロイメント - 推奨 (~20 分)](#deployment---recommended-20-mins) で説明されている手順を完了します。