---
title: Azure
description: AzureでのW&Bサーバーのホスティング
displayed_sidebar: default
---

# Azure

:::info
W&Bは、[W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) や [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) のデプロイメントタイプなど、完全管理のデプロイメントオプションを推奨しています。W&Bの完全管理サービスはシンプルで安全に使用でき、設定が最小または不要です。
:::

自己管理のW&B Serverを選んだ場合は、Azure上でプラットフォームをデプロイするために [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) を使用することをお勧めします。

モジュールのドキュメントは豊富で、利用可能なすべてのオプションが含まれています。このドキュメントでは、一部のデプロイメントオプションをカバーします。

始める前に、Terraformの[State File](https://developer.hashicorp.com/terraform/language/state)を格納するために使用可能な[リモートバックエンド](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)のいずれかを選択することをお勧めします。

State Fileはアップグレードを実施したり、すべてのコンポーネントを再作成することなくデプロイメントに変更を加えるために必要なリソースです。

Terraform Moduleは以下の `必須` コンポーネントをデプロイします：

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Fliexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

その他のデプロイメントオプションには、以下のオプションコンポーネントも含まれる場合があります：

- Azure Cache for Redis
- Azure Event Grid

## **事前条件の権限**

AzureRMプロバイダーを設定する最も簡単な方法は [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) を使用することですが、自動化の場合は [Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) を使用することもできます。
認証メソッドにかかわらず、Terraformを実行するアカウントは、イントロダクションで説明されているすべてのコンポーネントを作成できる必要があります。

## 一般的な手順
このトピックのステップは、このドキュメントでカバーされている任意のデプロイメントオプションに共通です。

1. 開発環境を準備する。
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールする
  * 使用するコードを含むGitリポジトリを作成することをお勧めしますが、ファイルをローカルに保持することもできます。

2. **`terraform.tfvars`ファイルを作成する** `tfvars`ファイルの内容はインストールタイプに応じてカスタマイズできますが、最低限の推奨事項は以下の例のようになります。

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   ここで定義された変数はデプロイメント前に決定する必要があります。`namespace`変数は、Terraformによって作成されるすべてのリソースにプレフィックスとして使用される文字列です。

   `subdomain`と`domain`の組み合わせは、W&Bが設定されるFQDNを形成します。上記の例では、W&B FQDNは`wandb-aws.wandb.ml`となり、FQDNレコードが作成されるDNS `zone_id`です。

3. **`versions.tf`ファイルを作成する** このファイルには、W&BをAWSにデプロイするために必要なTerraformおよびTerraformプロバイダーのバージョンが含まれます。
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

  AWSプロバイダーを設定するための詳細は、[Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) を参照してください。

  任意ですが強くお勧めするのは、このドキュメントの冒頭で述べた[リモートバックエンド設定](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)を追加することです。

4. **`variables.tf`ファイルを作成する** `terraform.tfvars`に構成された各オプションには対応する変数宣言が必要です。

   ```bash
    variable "namespace" {
      type        = string
      description = "プレフィックスリソースに使用される文字列。"
    }

    variable "location" {
      type        = string
      description = "Azure Resource Groupの場所"
    }

    variable "domain_name" {
      type        = string
      description = "Weights & Biases UIへアクセスするためのドメイン。"
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Weights & Biases UIへアクセスするためのサブドメイン。デフォルトはRoute53 Routeにレコードを作成。"
    }

    variable "license" {
      type        = string
      description = "あなたのwandb/localライセンス"
    }
  ```

## デプロイメント - 推奨 (~20分)

これは、すべての`必須`コンポーネントを作成し、最新バージョンの`W&B`を`Kubernetes クラスター`にインストールする最も簡単なデプロイメントオプション設定です。

1. **`main.tf`を作成する** 一般的な手順でファイルを作成したディレクトリに、以下の内容で`main.tf`ファイルを作成します：

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

  # 必要なすべてのサービスを稼働させます
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

2. **W&Bにデプロイする** W&Bにデプロイするには、以下のコマンドを実行します：

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDISキャッシュを使用したデプロイメント

別のデプロイメントオプションとして、`Redis`を使用してSQLクエリをキャッシュし、実験のメトリクスを読み込む際のアプリケーション応答速度を向上させる方法があります。

キャッシュを有効にするには、[`Deployment Recommended`](azure-tf.md#deployment---recommended-20-mins)で説明した同じ`main.tf`ファイルに`create_redis = true`オプションを追加する必要があります。

```bash
# 必要なすべてのサービスを稼働させます
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"


  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  create_redis       = true # Redisを作成
  [...]
```

## 外部キューを使用したデプロイメント

デプロイメントオプション3は、外部の`メッセージブローカー`を有効にするもので、これは任意です。なぜなら、W&Bには埋め込みブローカーが含まれているためです。このオプションはパフォーマンスの改善をもたらしません。

メッセージブローカーを提供するAzureリソースは`Azure Event Grid`であり、これを有効にするには、[`Deployment Recommended`](azure-tf.md#deployment---recommended-20-mins)で説明した同じ`main.tf`ファイルに`use_internal_queue = false`オプションを追加する必要があります。
```bash
# 必要なすべてのサービスを稼働させます
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"


  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  use_internal_queue       = false # Azure Event Gridを有効にする
  [...]
}
```

## その他のデプロイメントオプション

すべての構成を同じファイルに追加することで、3つのデプロイメントオプションを組み合わせることができます。
[Terraform Module](https://github.com/wandb/terraform-azure-wandb) には、標準オプションや[`Deployment Recommended`](azure-tf.md#deployment---recommended-20-mins)で見つかる最小構成とともに組み合わせることができる複数のオプションが用意されています。