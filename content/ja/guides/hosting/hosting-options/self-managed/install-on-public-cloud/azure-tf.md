---
title: Azure で W&B プラットフォーム をデプロイする
description: Azure で W&B サーバーをホスティングする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) などのフルマネージドなデプロイメントオプションを推奨しています。W&B のフルマネージドサービスは、シンプルかつセキュアに利用でき、設定も最小限もしくは不要です。
{{% /alert %}}

ご自身で W&B Server を管理する場合は、Azure 上でプラットフォームをデプロイするために [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) の利用を W&B は推奨します。

このモジュールのドキュメントは非常に充実しており、利用可能な全てのオプションが記載されています。本ドキュメントでは主要なデプロイメントオプションについて解説します。

開始前に、Terraform の [State File](https://developer.hashicorp.com/terraform/language/state) を保存するため、[リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend)を選択することをおすすめします。

State File は、デプロイメントのコンポーネントをすべて再作成せずにアップグレードや変更を行う際に必要なリソースです。

Terraform モジュールでは、以下の `必須` コンポーネントがデプロイされます：

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Flexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

その他のデプロイメントオプションとして、以下のオプションコンポーネントも追加できます：

- Azure Cache for Redis
- Azure Event Grid

## **前提となる権限**

最も簡単な AzureRM プロバイダーの設定方法は [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) の利用ですが、自動化用途には [Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) の利用も便利です。  
どちらの認証方法を使用する場合でも、Terraform を実行するアカウントにはイントロダクションで記載した全コンポーネントを作成できる権限が必要です。

## 全体的なステップ
本トピックのステップは、本ドキュメントで取り上げている全てのデプロイメントオプションに共通です。

1. 開発環境を準備します。
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします
  * 利用するコードは Git リポジトリーで管理することを推奨しますが、ローカルファイルでも構いません。

2. **`terraform.tfvars` ファイルを作成**  
   `tfvars` ファイルの内容はインストールタイプに応じてカスタマイズできますが、最低限おすすめの例は下記の通りです。

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   ここで定義する変数は、デプロイメント前に決めておく必要があります。  
   `namespace` 変数は、Terraform で作成される全リソースのプレフィックスに利用される文字列です。

   `subdomain` と `domain` を組み合わせることで、W&B が設定される FQDN となります。上記例だと、W&B の FQDN は `wandb-aws.wandb.ml`、該当 FQDN レコードが作成される DNS `zone_id` となります。

3. **`versions.tf` ファイルを作成**  
   このファイルには、AWS 上で W&B をデプロイするために必要な Terraform とプロバイダーのバージョンを記載します。
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

  AWS プロバイダーの設定方法は [Terraform 公式ドキュメント](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) をご確認ください。

  **オプションですが強く推奨** されるのは、本ドキュメント冒頭で紹介した [リモートバックエンドの設定](https://developer.hashicorp.com/terraform/language/backend) を追加することです。

4. **`variables.tf` ファイルを作成**  
   `terraform.tfvars` で設定する各オプションに対して、Terraform では対応する変数宣言が必要です。

  ```bash
    variable "namespace" {
      type        = string
      description = "各リソースのプレフィックスに使われる文字列"
    }

    variable "location" {
      type        = string
      description = "Azure Resource Group のロケーション"
    }

    variable "domain_name" {
      type        = string
      description = "Weights & Biases UI にアクセスするためのドメイン"
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Weights & Biases UI にアクセスするためのサブドメイン。デフォルトでは Route53 上にレコードを作成します。"
    }

    variable "license" {
      type        = string
      description = "ご自身の wandb/local ライセンス"
    }
  ```

## 推奨デプロイメント

この手順は全ての `必須` コンポーネントを作成し、`Kubernetes クラスター`上に最新バージョンの W&B をインストールする最もシンプルな構成です。

1. **`main.tf` を作成**  
   `General Steps` で作成したファイルと同じディレクトリーで、下記内容で `main.tf` ファイルを作成します。

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

2. **W&B へデプロイ**  
   W&B をデプロイするには、下記のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを活用したデプロイメント

別のデプロイメントオプションとして、`Redis` を使って SQL クエリのキャッシュやアプリケーションレスポンスの高速化が可能です（特に Experiments のメトリクス表示時）。

キャッシュを有効にするには、[推奨デプロイメント]({{< relref path="#recommended-deployment" lang="ja" >}}) で使った `main.tf` ファイルに `create_redis = true` オプションを追加します。

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

## 外部キューを利用したデプロイメント

3つ目のオプションとして、外部の `メッセージブローカー` を有効にする方法があります。これは必須ではなく、W&B にはブローカーが組み込まれているため、パフォーマンス向上のメリットはありません。

Azure でメッセージブローカーを提供するリソースは `Azure Event Grid` です。有効化するには、[推奨デプロイメント]({{< relref path="#recommended-deployment" lang="ja" >}}) で使った `main.tf` に `use_internal_queue = false` オプションを追加してください。

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

## その他のデプロイメントオプション

すべてのデプロイメントオプションは、同一ファイル内で組み合わせて設定可能です。  
[Terraform モジュール](https://github.com/wandb/terraform-azure-wandb) には、標準及び最小限の設定以外にも、様々な組み合わせが可能なオプションが用意されています。  
詳細は [推奨デプロイメント]({{< relref path="#recommended-deployment" lang="ja" >}}) をご確認ください。