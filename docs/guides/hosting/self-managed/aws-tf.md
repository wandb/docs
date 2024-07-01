---
title: AWS
description: AWS で W&B サーバー をホスティングする
displayed_sidebar: default
---

# AWS

:::info
W&B は、[W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) や [W&B Dedicated Cloud](../hosting-options/dedicated_cloud.md) のデプロイメントオプションを推奨しています。W&B の完全管理サービスはシンプルで安全に使用でき、設定は最小限または不要です。
:::

W&B は AWS 上にプラットフォームをデプロイするために [W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) を使用することを推奨しています。

このモジュールのドキュメントは非常に充実しており、利用可能なオプションが全て記載されています。この記事では、いくつかのデプロイメントオプションについて説明します。

開始する前に、Terraformの[State File](https://developer.hashicorp.com/terraform/language/state)を保存するための[リモートバックエンド](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)のいずれかを選択することをお勧めします。

State File は、すべてのコンポーネントを再作成することなく、アップグレードをロールアウトしたりデプロイメントに変更を加えるために必要なリソースです。

Terraform Module は次の `必須` コンポーネントをデプロイします：

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

その他のデプロイメントオプションには、以下のようなオプションのコンポーネントも含めることができます：

- Elastic Cache for Redis
- SQS

## **前提条件としての権限**

Terraform を実行するアカウントは、イントロダクションで説明したすべてのコンポーネントを作成できる必要があり、**IAM ポリシー** および **IAM ロール** を作成し、リソースにロールを割り当てる権限が必要です。

## 一般的な手順

このトピックの手順は、このドキュメントでカバーされている任意のデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform をインストール](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
   - 使用するコードの Git リポジトリを作成することをお勧めしますが、ファイルをローカルに保持することもできます。
2. `terraform.tfvars` ファイルを作成します。

   `tvfars` ファイルの内容はインストールタイプに応じてカスタマイズできますが、最小限の推奨内容は以下の例のようになります。

   ```bash
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   ```

   デプロイする前に、`namespace` 変数は Terraform が作成するすべてのリソースのプレフィックスとして使用される文字列であるため、 `tvfars` ファイルで変数を定義してください。

   `subdomain` と `domain` の組み合わせが W&B が設定される FQDN を形成します。上記の例では、W&B の FQDN は `wandb-aws.wandb.ml` となり、FQDN レコードが作成される DNS `zone_id` となります。

   `allowed_inbound_cidr` と `allowed_inbound_ipv6_cidr` も設定する必要があります。このモジュールでは、これが必須の入力となります。進行中の例では、W&B インストールへのアクセスを任意のソースから許可しています。

3. `versions.tf` ファイルを作成します。

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

   AWS プロバイダーを設定するには、[Terraform 公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)を参照してください。

   オプションですが、**強く推奨**されるのは、このドキュメントの最初に記載された [リモートバックエンドの設定](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)を追加することです。

4. `variables.tf` ファイルを作成します。

   `terraform.tfvars` で設定されたすべてのオプションに対応する変数宣言が必要です。

   ```
   variable "namespace" {
     type        = string
     description = "リソースに使用される名前プレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "インスタンスにアクセスするために使用されるドメイン名。"
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UI へのアクセス用のサブドメイン。"
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biases のサブドメインを作成するためのドメイン。"
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-server へのアクセスを許可する CIDR。"
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server へのアクセスを許可する CIDR。"
    nullable    = false
    type        = list(string)
   }
   ```

## デプロイメント - 推奨設定 (~20 分)

これは、すべての `必須` コンポーネントを作成し、最新バージョンの `W&B` を `Kubernetes クラスター` にインストールする最も簡単なデプロイメントオプションの設定です。

1. `main.tf` を作成します

   `一般的な手順` でファイルを作成したのと同じディレクトリに、以下の内容で `main.tf` ファイルを作成します。

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

     # 作業グループがスピンアップしている間、tf がデプロイを開始しないようにします。
     depends_on = [module.wandb_infra]
   }

   output "bucket_name" {
     value = module.wandb_infra.bucket_name
   }

   output "url" {
     value = module.wandb_infra.url
   }
   ```

2. W&B をデプロイします

   W&B をデプロイするためには、以下のコマンドを実行します：

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS の有効化

別のデプロイメントオプションとして、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスを読み込む際のアプリケーションの応答速度を向上させる方法があります。

キャッシュを有効にするには、`main.tf` ファイルに `create_elasticache_subnet = true` のオプションを追加する必要があります。

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

デプロイメントオプション 3 では、外部の `メッセージブローカー` の有効化が含まれています。これはオプションで、W&B には組み込まれたブローカーが同梱されています。このオプションはパフォーマンスの改善をもたらしません。

メッセージブローカーを提供する AWS リソースは `SQS` であり、それを有効にするためには、`Recommended Deployment` で作成した `main.tf` に `use_internal_queue = false` のオプションを追加する必要があります。

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

## その他のデプロイメントオプション

同じファイルにすべての設定を追加することで、3 つのデプロイメントオプションすべてを組み合わせることができます。
[Terraform Module](https://github.com/wandb/terraform-aws-wandb) は、標準オプションと `Deployment - Recommended` にある最小限の設定と組み合わせて使用できる複数のオプションを提供しています。

## 手動設定

W&B のファイルストレージバックエンドとして Amazon S3 バケットを使用するには、以下の手順が必要です：

* [Amazon S3 バケットとバケット通知の作成](#create-an-s3-bucket-and-bucket-notifications)
* [SQS キューの作成](#create-an-sqs-queue)
* [W&B を実行するノードに権限を付与](#grant-permissions-to-node-running-wb)

バケットを作成し、そのバケットからオブジェクト作成通知を受け取るように設定された SQS キューを作成する必要があります。インスタンスにはこのキューから読み取る権限が必要です。

### Amazon S3 バケットとバケット通知の作成

以下の手順に従って、Amazon S3 バケットを作成し、バケット通知を有効にします。

1. AWS コンソールで Amazon S3 に移動します。
2. **Create bucket**（バケットの作成）を選択します。
3. **Advanced settings**（詳細設定）の **Events**（イベント）セクションで、**Add notification**（通知を追加）を選択します。
4. すべてのオブジェクト作成イベントを、前に設定した SQS キューに送信するように設定します。

![Enterprise file storage settings](/images/hosting/s3-notification.png)

CORS アクセスを有効にします。あなたの CORS 設定は以下のようになります：

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

1. AWS コンソールで Amazon SQS に移動します。
2. **Create queue**（キューの作成）を選択します。
3. **Details**（詳細）セクションで、**Standard**（標準）キュータイプを選択します。
4. アクセスポリシーセクションで、以下のプリンシパルに権限を付与します：
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

オプションとして、**Access Policy**（アクセスポリシー）セクションで高度なアクセスポリシーを追加します。例えば、以下はアマゾン SQS へのアクセスポリシーの例です：

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

### W&B 実行ノードに権限を付与

W&B サーバーが実行されているノードは、Amazon S3 および Amazon SQS へのアクセスを許可するように設定する必要があります。選択したサーバーデプロイメントタイプによっては、以下のポリシーステートメントをノードロールに追加する必要があります：

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

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin` にある W&B 設定ページに移動します。
2. **外部ファイルストレージバックエンドを使用** オプションを有効にします。
3. Amazon S3 バケット、リージョン、および Amazon SQS キューに関する情報を以下の形式で提供します：
* **File Storage Bucket**: `s3://<bucket-name>`
* **File Storage Region (AWS only)**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

![](/images/hosting/configure_file_store.png)

4. **Update settings**（設定の更新）を選択して、新しい設定を適用します。

## W&B バージョンのアップグレード

ここに記載されている手順に従って W&B を更新します：

1. `wandb_app` モジュールの設定に `wandb_version` を追加します。アップグレードしたい W&B のバージョンを指定します。例えば、以下の行では W&B バージョン `0.48.1` を指定しています：

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  :::info
  また、`wandb_version` を `terraform.tfvars` に追加し、同名の変数を作成して、リテラル値の代わりに `var.wandb_version` を使用することもできます。
  :::

