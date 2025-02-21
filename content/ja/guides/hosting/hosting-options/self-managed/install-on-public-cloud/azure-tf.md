---
title: Deploy W&B Platform on Azure
description: Azure で W&B サーバー をホストする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
Weights & Biases では、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドのデプロイメントオプションを推奨します。W&B のフルマネージドサービスは、シンプルで安全に使用でき、設定は最小限で済みます。
{{% /alert %}}

W&B Server の自己管理を行う場合、Azure 上にプラットフォームをデプロイするために、[W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) を使用することを推奨します。

モジュールのドキュメントは広範囲にわたり、利用可能なすべてのオプションが記載されています。このドキュメントでは、いくつかのデプロイメントオプションについて説明します。

開始する前に、[State File](https://developer.hashicorp.com/terraform/language/state) を保存するために、Terraform で利用可能な [remote backends](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択することを推奨します。

State File は、すべてのコンポーネントを再作成することなく、アップグレードを実施したり、デプロイメントに変更を加えたりするために必要なリソースです。

Terraform Module は、以下の `必須` コンポーネントをデプロイします。

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Flexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

その他のデプロイメントオプションには、以下のオプションコンポーネントを含めることもできます。

- Azure Cache for Redis
- Azure Event Grid

## **前提条件となる権限**

AzureRM プロバイダーを設定する最も簡単な方法は、[Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) を使用することですが、[Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) を使用した自動化も役立ちます。
どの認証方法を使用するにしても、Terraform を実行するアカウントは、イントロダクションで説明されているすべてのコンポーネントを作成できる必要があります。

## 一般的な手順

このトピックの手順は、このドキュメントで説明するすべてのデプロイメントオプションに共通です。

1. 開発環境を準備します。
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
  * 使用するコードで Git リポジトリを作成することを推奨しますが、ファイルをローカルに保存することもできます。

2. **`terraform.tfvars` ファイルを作成します** `tvfars` ファイルの内容は、インストールタイプに応じてカスタマイズできますが、最小限の推奨設定は以下の例のようになります。

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   ここで定義されている変数は、デプロイメントの前に決定する必要があります。`namespace` 変数は、Terraform によって作成されたすべてのリソースのプレフィックスとなる文字列です。

   `subdomain` と `domain` の組み合わせで、W&B が設定される FQDN が形成されます。上記の例では、W&B の FQDN は `wandb-aws.wandb.ml` になり、FQDN レコードが作成される DNS の `zone_id` になります。

3. **ファイル `versions.tf` を作成します** このファイルには、AWS に W&B をデプロイするために必要な Terraform と Terraform プロバイダーの バージョンが含まれます。
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

  AWS プロバイダーの設定については、[Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

  オプションで、**強く推奨されます** が、このドキュメントの冒頭で述べた [remote backend configuration](https://developer.hashicorp.com/terraform/language/backend) を追加できます。

4. **ファイル** `variables.tf` を作成します。`terraform.tfvars` で設定されたすべてのオプションについて、Terraform は対応する変数宣言を必要とします。

  ```bash
    variable "namespace" {
      type        = string
      description = "リソースのプレフィックスに使用される文字列。"
    }

    variable "location" {
      type        = string
      description = "Azure Resource Group の場所"
    }

    variable "domain_name" {
      type        = string
      description = "Weights & Biases UI にアクセスするためのドメイン。"
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Weights & Biases UI にアクセスするためのサブドメイン。デフォルトでは、Route53 Route にレコードを作成します。"
    }

    variable "license" {
      type        = string
      description = "Your wandb/local license"
    }
  ```

## 推奨されるデプロイメント

これは最も簡単なデプロイメントオプションの設定で、すべての `必須` コンポーネントを作成し、最新バージョンの `W&B` を `Kubernetes Cluster` にインストールします。

1. **`main.tf` を作成します** `General Steps` でファイルを作成したのと同じディレクトリーに、以下の内容で `main.tf` ファイルを作成します。

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

2. **W&B にデプロイします** W&B をデプロイするには、次のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS Cache を使用したデプロイメント

別のデプロイメントオプションでは、SQL クエリをキャッシュし、Experiments のメトリクスをロードする際のアプリケーションの応答を高速化するために `Redis` を使用します。

キャッシュを有効にするには、[recommended deployment]({{< relref path="#recommended-deployment" lang="ja" >}}) で使用したのと同じ `main.tf` ファイルにオプション `create_redis = true` を追加する必要があります。

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

  create_redis       = true # Redis を作成
  [...]
```

## 外部キューを使用したデプロイメント

デプロイメントオプション 3 は、外部の `message broker` を有効にすることです。W&B にはブローカーが組み込まれているため、これはオプションです。このオプションは、パフォーマンスの向上をもたらしません。

メッセージブローカーを提供する Azure リソースは `Azure Event Grid` であり、それを有効にするには、[recommended deployment]({{< relref path="#recommended-deployment" lang="ja" >}}) で使用したのと同じ `main.tf` にオプション `use_internal_queue = false` を追加する必要があります。
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

  use_internal_queue       = false # Azure Event Grid を有効にする
  [...]
}
```

## その他のデプロイメントオプション

3 つのデプロイメントオプションをすべて組み合わせて、すべての設定を同じファイルに追加できます。
[Terraform Module](https://github.com/wandb/terraform-azure-wandb) は、標準オプションと [recommended deployment]({{< relref path="#recommended-deployment" lang="ja" >}}) にある最小限の設定とともに組み合わせることができる、いくつかのオプションを提供します。
