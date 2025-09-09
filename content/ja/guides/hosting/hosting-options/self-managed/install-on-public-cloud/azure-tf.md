---
title: W&B プラットフォームを Azure 上にデプロイする
description: Azure で W&B サーバーをホストする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B 専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドなデプロイメントタイプを推奨します。W&B のフルマネージドサービスは、設定が最小限か不要で、シンプルかつ安全に利用できます。
{{% /alert %}}

W&B Server をセルフマネージドで運用することを決めた場合は、Azure 上にプラットフォームをデプロイするために [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) を使用することを推奨します。

このモジュールのドキュメントは充実しており、利用可能なすべてのオプションを網羅しています。本ドキュメントでは、そのうちいくつかのデプロイメントオプションを扱います。

開始前に、Terraform の [remote backends](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選び、[State File](https://developer.hashicorp.com/terraform/language/state) を保管することを推奨します。

State File は、すべてのコンポーネントを再作成することなく、アップグレードの展開やデプロイメントの変更を行うために必要なリソースです。

Terraform Module は、次の「mandatory」コンポーネントをデプロイします:

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Fliexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

他のデプロイメントオプションとして、以下のオプションコンポーネントも含められます:

- Azure Cache for Redis
- Azure Event Grid

## 事前に必要な権限

AzureRM プロバイダーを設定する最も簡単な方法は [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) を使うことです。自動化には [Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) の利用も有用です。
いずれの認証方法でも、Terraform を実行するアカウントはイントロダクションで説明したすべてのコンポーネントを作成できる必要があります。

## General steps
このトピックの手順は、このドキュメントで扱うすべてのデプロイメントオプションに共通です。

1. 開発環境を準備します。
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
  * 利用するコードを格納する Git リポジトリーを作成することを推奨しますが、ローカルにファイルを置くこともできます。

2. 「terraform.tfvars」ファイルを作成します。`tvfars` ファイルの内容はインストールタイプに合わせてカスタマイズできますが、最低限の推奨例は以下のとおりです。

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   ここで定義する変数はデプロイ前に決めておく必要があります。`namespace` 変数は、Terraform が作成するすべてのリソースに付与される接頭辞の文字列です。

   `subdomain` と `domain` の組み合わせが、W&B を設定する FQDN を構成します。上記の例では、W&B の FQDN は `wandb-aws.wandb.ml` となり、その FQDN のレコードは該当する DNS の `zone_id` に作成されます。

3. `versions.tf` ファイルを作成します。このファイルには、AWS で W&B をデプロイするために必要な Terraform および Terraform プロバイダーのバージョンを記載します。
  ```bash
  terraform {
    required_version = "~> 1.3"

    required_providers {
      azurerm = {
        source  = "hashicorp/azurerm"
        version = "~> 3.17"
      }
    }
  }
  ```

  AWS プロバイダーの設定方法は、[Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

  オプションですが、強く推奨します。ドキュメント冒頭で触れた [remote backend の設定](https://developer.hashicorp.com/terraform/language/backend) を追加できます。

4. `variables.tf` ファイルを作成します。`terraform.tfvars` で設定した各オプションに対して、Terraform では対応する変数宣言が必要です。

  ```bash
    variable "namespace" {
      type        = string
      description = "String used for prefix resources."
    }

    variable "location" {
      type        = string
      description = "Azure Resource Group location"
    }

    variable "domain_name" {
      type        = string
      description = "Domain for accessing the Weights & Biases UI."
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Subdomain for accessing the Weights & Biases UI. Default creates record at Route53 Route."
    }

    variable "license" {
      type        = string
      description = "Your wandb/local license"
    }
  ```

## Recommended deployment

これは最もシンプルなデプロイメントの設定で、すべての「Mandatory」コンポーネントを作成し、`Kubernetes Cluster` に最新の `W&B` をインストールします。

1. `main.tf` を作成します。`General Steps` で作成したのと同じディレクトリーに、以下の内容で `main.tf` ファイルを作成します。

  ```bash
  provider "azurerm" {
    features {}
  }

  provider "kubernetes" {
    host                   = module.wandb.cluster_host
    cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
    client_key             = base64decode(module.wandb.cluster_client_key)
    client_certificate     = base64decode(module.wandb.cluster_client_certificate)
  }

  provider "helm" {
    kubernetes {
      host                   = module.wandb.cluster_host
      cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
      client_key             = base64decode(module.wandb.cluster_client_key)
      client_certificate     = base64decode(module.wandb.cluster_client_certificate)
    }
  }

  # 必要なサービスをすべて起動
  module "wandb" {
    source  = "wandb/wandb/azurerm"
    version = "~> 1.2"

    namespace   = var.namespace
    location    = var.location
    license     = var.license
    domain_name = var.domain_name
    subdomain   = var.subdomain

    deletion_protection = false

    tags = {
      "Example" : "PublicDns"
    }
  }

  output "address" {
    value = module.wandb.address
  }

  output "url" {
    value = module.wandb.url
  }
  ```

2. W&B へデプロイします。W&B をデプロイするには、次のコマンドを実行します:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## Deployment with REDIS Cache

別のデプロイメントオプションとして、`Redis` を用いて SQL クエリーをキャッシュし、Experiments のメトリクスを読み込む際のアプリケーション応答を高速化する方法があります。

キャッシュを有効化するには、[recommended deployment]({{< relref path="#recommended-deployment" lang="ja" >}}) で使用した同じ `main.tf` に `create_redis = true` オプションを追加します。

```bash
# 必要なサービスをすべて起動
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"


  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  create_redis       = true # Redis を作成
  [...]
```

## Deployment with External Queue

デプロイメントオプション 3 は、外部の `message broker` を有効化する構成です。W&B にはブローカーが組み込まれているため、このオプションは任意です。このオプションで性能が向上することはありません。

メッセージブローカーを提供する Azure リソースは `Azure Event Grid` です。これを有効化するには、[recommended deployment]({{< relref path="#recommended-deployment" lang="ja" >}}) で使用した同じ `main.tf` に `use_internal_queue = false` オプションを追加します。
```bash
# 必要なサービスをすべて起動
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"


  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  use_internal_queue       = false # Azure Event Grid を有効化
  [...]
}
```

## Other deployment options

3 つのデプロイメントオプションは、同じファイルにすべての設定を追加して組み合わせることができます。
[Terraform Module](https://github.com/wandb/terraform-azure-wandb) には、標準オプションや [recommended deployment]({{< relref path="#recommended-deployment" lang="ja" >}}) の最小構成と組み合わせて利用できる複数のオプションが用意されています。