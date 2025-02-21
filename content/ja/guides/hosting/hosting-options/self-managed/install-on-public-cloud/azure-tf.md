---
title: Deploy W&B Platform on Azure
description: Azure での W&B サーバー のホスティング。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) もしくは [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) デプロイメントタイプのような完全に管理されたデプロイメントオプションを推奨します。W&B 完全管理サービスは、シンプルかつ安全に使用でき、設定は最小限または不要です。
{{% /alert %}}

W&B Server を自己管理することを決定した場合、W&B は Azure 上でプラットフォームをデプロイするために [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) を使用することを推奨します。

モジュールのドキュメントは非常に詳細で、使用できるすべてのオプションが含まれています。このドキュメントではいくつかのデプロイメントオプションを紹介します。

開始する前に、Terraform の [State File](https://developer.hashicorp.com/terraform/language/state) を保存するための、利用可能な [remote backends](https://developer.hashicorp.com/terraform/language/backend) の一つを選択することをお勧めします。

State File は、すべてのコンポーネントを再作成することなく、アップグレードを展開したりデプロイメントを変更するために必要なリソースです。

Terraform Module は以下の `必須` コンポーネントをデプロイします:

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Flexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

他のデプロイメントオプションには、次のオプショナルコンポーネントが含まれる場合もあります:

- Azure Cache for Redis
- Azure Event Grid

## **前提条件の権限**

AzureRM プロバイダーを設定する最も簡単な方法は [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) を通じてですが、[Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) を使用した自動化の場合も役立ちます。
認証メソッドに関係なく、Terraform を実行するアカウントはイントロダクションに記載されたすべてのコンポーネントを作成できる必要があります。

## 一般的な手順
このトピックの手順は、このドキュメントで取り上げられた任意のデプロイメントオプションで共通です。

1. 開発環境を準備します。
   * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
   * 使用するコードで Git リポジトリを作成することをお勧めしますが、ローカルにファイルを保持しておくことも可能です。

2. **`terraform.tfvars` ファイルを作成します。** `tvfars` ファイルのコンテンツはインストールタイプによってカスタマイズできますが、最低限の推奨設定は以下の例のようになります。

   ```bash
   namespace     = "wandb"
   wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain     = "wandb-aws"
   domain_name   = "wandb.ml"
   location      = "westeurope"
   ```

   ここで定義される変数はデプロイメントの前に決定される必要があります。`namespace` 変数は Terraform が作成したすべてのリソースにプレフィックスとして付けられる文字列です。

   `subdomain` と `domain` の組み合わせがW&Bが設定される FQDN を構成します。上記の例では、W&B FQDN は `wandb-aws.wandb.ml` となり、DNS の `zone_id` が FQDN レコードを作成する場所となります。

3. **ファイル `versions.tf` を作成します。** このファイルには、W&B を AWS にデプロイするために必要な Terraform と Terraform プロバイダのバージョンが含まれます。
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

   AWS プロバイダーを設定するには [Terraform 公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

   オプションとして、**しかし非常に推奨されています**、このドキュメントの冒頭で述べた [remote backend configuration](https://developer.hashicorp.com/terraform/language/backend) を追加することができます。

4. **ファイル** `variables.tf` を作成します。`terraform.tfvars` で設定されたオプションごとに、Terraform は対応する変数宣言が必要です。

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

## 推奨デプロイメント

これは、すべての`必須`コンポーネントを作成し、`Kubernetes クラスター`に最新の`W&B`のバージョンをインストールする最も簡単なデプロイメントオプション設定です。

1. **`main.tf` を作成します。** `General Steps` で作成したファイルと同じディレクトリに、以下の内容で `main.tf` ファイルを作成します。

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

   # Spin up all required services
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

2. **W&B にデプロイ** W&B をデプロイするには、次のコマンドを実行します:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS Cache を使ったデプロイメント

別のデプロイメントオプションでは、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスをロードする際のアプリケーション応答を高速化します。

キャッシュを有効にするために、`recommended deployment`({{< relref path="#recommended-deployment" lang="ja" >}}) で使用したのと同じ `main.tf` ファイルに `create_redis = true` オプションを追加する必要があります。

```bash
# Spin up all required services
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"

  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  create_redis       = true # Create Redis
  [...]
```

## External Queue を使ったデプロイメント

デプロイメントオプション 3 は、外部 `message broker` を有効にすることです。これはオプションですが、W&B は組み込みのブローカーを提供しています。このオプションはパフォーマンスの向上をもたらしません。

メッセージブローカーを提供する Azure リソースは `Azure Event Grid` であり、有効にするには、`recommended deployment`({{< relref path="#recommended-deployment" lang="ja" >}}) で使用したのと同じ `main.tf` に `use_internal_queue = false` オプションを追加する必要があります。
```bash
# Spin up all required services
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"

  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  use_internal_queue       = false # Enable Azure Event Grid
  [...]
}
```

## その他のデプロイメントオプション

3 つのデプロイメントオプションをすべて組み合わせ、すべての設定を同じファイルに追加できます。
[Terraform Module](https://github.com/wandb/terraform-azure-wandb) は、標準オプションおよび [recommended deployment]({{< relref path="#recommended-deployment" lang="ja" >}}) に見られる最小構成と組み合わせられるいくつかのオプションを提供しています。