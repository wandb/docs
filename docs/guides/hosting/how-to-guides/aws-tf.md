---
description: Hosting W&B Server on AWS.
displayed_sidebar: default
---

# AWS

Weights and Biasesが開発した [Terraformモジュール](https://registry.terraform.io/modules/wandb/wandb/aws/latest) を使って、W&BサーバーをAWSに展開することをお勧めします。

モジュールのドキュメントは非常に充実しており、使用可能なすべてのオプションが網羅されています。この文章では、いくつかの展開オプションについて説明します。

始める前に、Terraformの[State File](https://developer.hashicorp.com/terraform/language/state) を保存するための [リモートバックエンド](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) の中から1つ選ぶことをお勧めします。

State Fileは、すべてのコンポーネントを再作成せずに展開をアップグレードしたり、変更を加えるために必要なリソースです。

Terraformモジュールは以下の `必須` コンポーネントを展開します:

- ロードバランサー
- AWS Identity & Access Management (IAM)
- AWS Key Management System (KMS)
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager (ACM)
- Amazon Elastic Loadbalancing (ALB)
- Amazon Secrets Manager

その他の展開オプションには、以下のオプションコンポーネントも含めることができます:
- Elastic Cache for Redis
- SQS

## **事前準備として必要な権限**

Terraformを実行するアカウントは、はじめに述べたすべてのコンポーネントを作成する権限と、**IAMポリシー**、**IAMロール**を作成し、リソースにロールを割り当てる権限が必要です。

## 共通手順

このドキュメントでカバーされているどの展開オプションも、このトピックの手順が共通です。

1. 開発環境を準備する。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールする
   - 使用するコードを含むGitリポジトリを作成することをお勧めしますが、ローカルにファイルを保管してもかまいません。
2. `terraform.tfvars`ファイルを作成します。

   `tfvars`ファイルの内容は、インストールタイプに応じてカスタマイズできますが、最低限推奨される内容は以下の例のようになります。

   ```bash
   namespace     = "wandb"
   wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain     = "wandb-aws"
   domain_name   = "wandb.ml"
   zone_id       = "xxxxxxxxxxxxxxxx"
   ```

   ここで定義された変数は、展開前に決定しておく必要があります。`namespace`変数は、Terraformによって作成されるすべてのリソースにプレフィックスとなる文字列です。

   `subdomain`と`domain`の組み合わせで、W&Bが構成されるFQDNが形成されます。上記の例では、W&BのFQDNは`wandb-aws.wandb.ml`となり、FQDNレコードが作成されるDNSの`zone_id`です。
3. `versions.tf`ファイルを作成する

   このファイルには、AWSでW&Bをデプロイするために必要なTerraformおよびTerraformプロバイダのバージョンが含まれます。

   ```bash
   terraform {
     required_providers {
       aws = {
         source  = "hashicorp/aws"
         version = "~> 3.60"
       }
     }
   }

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

   AWSプロバイダの設定については、[Terraform公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)を参照してください。
オプションですが、**強く推奨される**方法として、このドキュメントの最初で言及されている[リモートバックエンド設定](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)を追加できます。

4. `variables.tf`ファイルを作成する

   `terraform.tfvars`に設定された各項目に対して、Terraformは対応する変数宣言が必要です。

   ```
   variable "namespace" {
     type        = string
     description = "リソースに使用される名前のプレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "インスタンスにアクセスするために使用されるドメイン名"
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UIにアクセスするためのサブドメイン"
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biasesのサブドメインを作成するためのドメイン"
   }
   ```
## 展開 - お勧め（〜20分）

この設定は、最も簡単な展開オプションで、すべての`Mandatory`コンポーネントを作成し、`Kubernetesクラスター`に`W&B`の最新バージョンをインストールします。

1. `main.tf`を作成する

   `General Steps`でファイルを作成したのと同じディレクトリで、次の内容の`main.tf`ファイルを作成します。

   ```
   module "wandb_infra" {
     source  = "wandb/wandb/aws"
     version = "1.6.0"

     namespace   = var.namespace
     domain_name = var.domain_name
     subdomain   = var.subdomain
     zone_id     = var.zone_id
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
     version = "1.5.0"

     license                    = var.license
     host                       = module.wandb_infra.url
     bucket                     = "s3://${module.wandb_infra.bucket_name}"
     bucket_aws_region          = module.wandb_infra.bucket_region
     bucket_queue               = "internal://"
     database_connection_string = "mysql://${module.wandb_infra.database_connection_string}"

     # もし待たないと、ワークグループが
     # まだ立ち上がっている間にtfがデプロイし始める
     depends_on = [module.wandb_infra]
   }

   output "bucket_name" {
     value = module.wandb_infra.bucket_name
   }

   output "url" {
     value = module.wandb_infra.url
   }
   ```

2. W&Bのデプロイ

   W&Bをデプロイするには、以下のコマンドを実行してください:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```
## REDISの有効化

別の展開オプションでは、`Redis`を使用してSQLクエリをキャッシュし、実験のメトリクスを読み込む際のアプリケーションのレスポンスを高速化します。

キャッシュを有効にするには、`Recommended Deployment`で取り上げた`main.tf`ファイルに`create_elasticache_subnet = true`オプションを追加する必要があります。

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.6.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
  **create_elasticache_subnet = true**
}
[...]
```

## メッセージブローカー（キュー）の有効化

展開オプション3は、外部の`メッセージブローカー`を有効にするものです。これはオプションであり、W＆Bにはブローカーが組み込まれているためです。このオプションはパフォーマンスの向上をもたらさないことに注意してください。

メッセージブローカーを提供するAWSリソースは`SQS`で、これを有効にするには、`Recommended Deployment`で取り上げた`main.tf`に`use_internal_queue = false`オプションを追加する必要があります。

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.6.0"
namespace   = var.namespace

  domain_name = var.domain_name

  subdomain   = var.subdomain

  zone_id     = var.zone_id

  **use_internal_queue = false**



[...]



```



## 他の展開オプション



すべての設定を同じファイルに追加して、3つの展開オプションを組み合わせることができます。

[Terraform Module](https://github.com/wandb/terraform-aws-wandb)では、`Deployment - Recommended`で見つかった標準オプションと最小構成に加えて、いくつかのオプションが組み合わせられます。



<!-- ## アップグレード（近日公開予定） -->