---
title: Deploy W&B Platform on Azure
description: Azure 上での W&B サーバー のホスティング
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
Weights & Biases では、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドなデプロイメントオプションをお勧めします。Weights & Biases のフルマネージドサービスは、シンプルで安全に使用でき、設定は最小限で済みます。
{{% /alert %}}

W&B Server の自己管理を選択した場合、Azure にプラットフォームをデプロイするには、[W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) を使用することをお勧めします。

このモジュールのドキュメントは広範囲にわたり、利用可能なすべてのオプションが記載されています。このドキュメントでは、いくつかのデプロイメントオプションについて説明します。

開始する前に、Terraform で利用可能な [remote backends](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択して、[State File](https://developer.hashicorp.com/terraform/language/state) を保存することをお勧めします。

State File は、すべてのコンポーネントを再作成せずに、アップグレードを展開したり、デプロイメントに変更を加えたりするために必要なリソースです。

Terraform Module は、次の `mandatory` コンポーネントをデプロイします。

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Fliexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

その他のデプロイメントオプションには、次のオプションコンポーネントも含まれます。

- Azure Cache for Redis
- Azure Event Grid

## **前提条件となる権限**

AzureRM プロバイダーを設定する最も簡単な方法は、[Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) を使用することですが、[Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) を使用した自動化も役立ちます。
使用する認証方法に関係なく、Terraform を実行するアカウントは、イントロダクションで説明されているすべてのコンポーネントを作成できる必要があります。

## 一般的な手順

このトピックの手順は、このドキュメントで説明されているすべてのデプロイメントオプションに共通です。

1. 開発 環境 を準備します。
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
  * 使用する コード で Git リポジトリを作成することをお勧めしますが、ファイルをローカルに保持することもできます。

2. **`terraform.tfvars` ファイルを作成します。** `tvfars` ファイルの内容は、インストールタイプに応じてカスタマイズできますが、推奨される最小限の内容は以下の例のようになります。

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   ここで定義されている変数は、デプロイメントの前に決定する必要があります。`namespace` 変数は、Terraform によって作成されたすべてのリソースにプレフィックスを付ける 文字列 になります。

   `subdomain` と `domain` の組み合わせで、Weights & Biases が設定される FQDN が形成されます。上記の例では、Weights & Biases の FQDN は `wandb-aws.wandb.ml` になり、FQDN レコードが作成される DNS `zone_id` になります。

3. **ファイル `versions.tf` を作成します。** このファイルには、Weights & Biases を AWS にデプロイするために必要な Terraform および Terraform プロバイダーの バージョン が含まれます。
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

  AWS プロバイダーを設定するには、[Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

  オプションで、**強く推奨されますが**、このドキュメントの冒頭で説明した [remote backend configuration](https://developer.hashicorp.com/terraform/language/backend) を追加できます。

4. **ファイル** `variables.tf` を作成します。`terraform.tfvars` で設定されたすべてのオプションについて、Terraform は対応する変数宣言を必要とします。

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

## 推奨されるデプロイメント

これは最も簡単なデプロイメントオプションの設定で、すべての `Mandatory` コンポーネントを作成し、最新 バージョン の `W&B` を `Kubernetes Cluster` にインストールします。

1. **`main.tf` を作成します。** `General Steps` でファイルを作成したのと同じ ディレクトリー に、次の内容で `main.tf` ファイルを作成します。

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

2. **W&B にデプロイします。** W&B をデプロイするには、次の コマンド を実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS Cache を使用したデプロイメント

別のデプロイメントオプションでは、SQL クエリをキャッシュし、Experiments の Metrics をロードする際のアプリケーション応答を高速化するために `Redis` を使用します。

キャッシュを有効にするには、[推奨されるデプロイメント]({{< relref path="#recommended-deployment" lang="ja" >}}) で使用したのと同じ `main.tf` ファイルにオプション `create_redis = true` を追加する必要があります。

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

## 外部キューを使用したデプロイメント

デプロイメントオプション 3 は、外部 `message broker` を有効にすることで構成されます。W&B にはブローカーが組み込まれているため、これはオプションです。このオプションは、パフォーマンスの向上をもたらしません。

message broker を提供する Azure リソースは `Azure Event Grid` であり、これを有効にするには、[推奨されるデプロイメント]({{< relref path="#recommended-deployment" lang="ja" >}}) で使用したのと同じ `main.tf` にオプション `use_internal_queue = false` を追加する必要があります。
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

3 つのデプロイメントオプションすべてを組み合わせて、すべての構成を同じファイルに追加できます。
[Terraform Module](https://github.com/wandb/terraform-azure-wandb) には、標準オプションと [推奨されるデプロイメント]({{< relref path="#recommended-deployment" lang="ja" >}}) にある最小限の構成とともに組み合わせることができる、いくつかのオプションが用意されています。
