---
title: Azure で W&B プラットフォームをデプロイする
description: Azure で W&B サーバーをホスティングする
menu:
  default:
    identifier: azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) や [W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) のような、フルマネージド型のデプロイメントオプションを推奨しています。W&B のフルマネージドサービスは、非常にシンプルかつセキュアに利用でき、設定も最小限または不要です。
{{% /alert %}}

もしご自身で W&B Server を運用する場合は、Azure 上にプラットフォームをデプロイするために [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) の利用をおすすめします。

このモジュールのドキュメントは充実しており、利用可能な全てのオプションが解説されています。本ドキュメントでは、一部のデプロイメントオプションについて解説します。

開始前に、[State File](https://developer.hashicorp.com/terraform/language/state) を保存するために利用できる [remote backend](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択することをおすすめします。

State File は、アップグレードやデプロイメントの変更時に、すべての構成要素を再作成することなく変更を適用するために必要なリソースです。

Terraform Module は、以下の `必須` コンポーネントをデプロイします：

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Flexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

また、他にも以下のようなオプションコンポーネントを追加することも可能です：

- Azure Cache for Redis
- Azure Event Grid

## **前提となる権限**

AzureRM プロバイダーの簡単な設定方法は、[Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) を利用する方法です。自動化の場合は [Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) の利用も役立ちます。
認証方法に関わらず、Terraform を実行するアカウントにはイントロダクションで記載したすべての構成要素を作成できる権限が必要です。

## 一般的な手順
このトピックの手順は、ここで解説するどのデプロイメントオプションにも共通です。

1. 開発環境の準備
   * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストール
   * 利用するコードを Git リポジトリーで管理することを推奨しますが、ローカルファイルでも問題ありません。

2. **`terraform.tfvars` ファイルの作成**  
   `tfvars` ファイルの内容は、インストールタイプに合わせてカスタマイズ可能ですが、推奨される最小限の例は下記の通りです。

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   ここで定義する各変数はデプロイメント前に決めておく必要があります。`namespace` 変数は、Terraform が作成するすべてのリソース名の先頭に付与される文字列です。

   `subdomain` と `domain` の組み合わせが、W&B が設定される FQDN になります。上記の例では、W&B の FQDN は `wandb-aws.wandb.ml` となり、この FQDN レコードが作成される DNS の `zone_id` も指定します。

3. **`versions.tf` ファイルの作成**  
   このファイルには、AWS 上に W&B をデプロイするために必要な Terraform と Terraform プロバイダーのバージョンを記述します。
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

   AWS プロバイダーの設定については、[Terraform 公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

   任意ですが、**強く推奨** されるのは、ドキュメントの冒頭で述べた [remote backend の設定](https://developer.hashicorp.com/terraform/language/backend) を追加することです。

4. **`variables.tf` ファイルの作成**  
   `terraform.tfvars` で設定したオプションごとに、Terraform では対応する変数宣言が必要です。

   ```bash
     variable "namespace" {
       type        = string
       description = "リソース名のプレフィックス用の文字列。"
     }

     variable "location" {
       type        = string
       description = "Azure Resource Group のロケーション"
     }

     variable "domain_name" {
       type        = string
       description = "Weights & Biases UI にアクセスするためのドメイン。"
     }

     variable "subdomain" {
       type        = string
       default     = null
       description = "Weights & Biases UI にアクセスするためのサブドメイン。デフォルトでは Route53 Route にレコードが作成されます。"
     }

     variable "license" {
       type        = string
       description = "Your wandb/local license"
     }
   ```

## 推奨デプロイメント

この方法が最もシンプルで、すべての `必須` コンポーネントを作成し、`Kubernetes Cluster` 上に最新バージョンの `W&B` をインストールします。

1. **`main.tf` の作成**  
   `General Steps` で作成したファイルと同じディレクトリに、以下の内容で `main.tf` を作成します。

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

   # 必要なすべてのサービスを起動
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

2. **W&B へのデプロイ**  
   W&B をデプロイするには、以下のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュありのデプロイメント

別のデプロイメントオプションとして、`Redis` を使って SQL クエリをキャッシュし、実験結果のメトリクス表示時のレスポンス高速化が可能です。

キャッシュを有効化するには、[推奨デプロイメント]({{< relref "#recommended-deployment" >}}) で使った `main.tf` ファイルに `create_redis = true` オプションを追加してください。

```bash
# 必要なすべてのサービスを起動
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

## 外部キュー利用のデプロイメント

デプロイメントオプション 3 では、外部の `message broker` を有効にします。これはオプションで、W&B にはブローカーが内蔵されているためパフォーマンスの向上はありません。

Azure の `Azure Event Grid` がメッセージブローカーの提供リソースで、[推奨デプロイメント]({{< relref "#recommended-deployment" >}}) で使った `main.tf` ファイルに `use_internal_queue = false` を追加して有効にします。

```bash
# 必要なすべてのサービスを起動
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

## その他のデプロイメントオプション

3つのデプロイメントオプションはすべて組み合わせて、同じファイル内で構成することが可能です。
[Tdrrraform Module](https://github.com/wandb/terraform-azure-wandb) では、多数のオプションが提供されており、標準オプションや [推奨デプロイメント]({{< relref "#recommended-deployment" >}}) の最小構成と組み合わせて利用できます。