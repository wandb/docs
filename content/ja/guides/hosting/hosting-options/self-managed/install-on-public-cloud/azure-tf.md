---
title: Azureで W&B プラットフォーム を展開する
description: Azure で W&B サーバー をホスティングする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) デプロイメント タイプのような完全に管理されたデプロイメント オプションをお勧めします。W&B の完全管理サービスは簡単で安全に使用でき、設定がほとんどまたは全く必要ありません。
{{% /alert %}}

自己管理の W&B サーバーを選択した場合、W&B は [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) を使用して Azure 上でプラットフォームをデプロイすることをお勧めします。

このモジュールのドキュメントは詳細で、使用可能なオプションがすべて含まれています。本書では、一部のデプロイメント オプションについて説明します。

開始する前に、Terraform の [State File](https://developer.hashicorp.com/terraform/language/state) を保存するために利用可能な [リモート バックエンド](https://developer.hashicorp.com/terraform/language/backend) のいずれかを選択することをお勧めします。

State File は、アップグレードを展開したり、すべてのコンポーネントを再作成することなくデプロイメントの変更を行ったりするために必要なリソースです。

Terraform モジュールは、次の「必須」コンポーネントをデプロイします。

- Azure リソース グループ
- Azure 仮想ネットワーク (VPC)
- Azure MySQL Flexible サーバー
- Azure ストレージ アカウント & Blob ストレージ
- Azure Kubernetes サービス
- Azure アプリケーション ゲートウェイ

その他のデプロイメント オプションには、次のオプション コンポーネントが含まれる場合があります。

- Azure Cache for Redis
- Azure Event Grid

## **前提条件の権限**

AzureRM プロバイダーを設定する最も簡単な方法は [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli) 経由ですが、[Azure サービス プリンシパル](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret) を使用した自動化の場合も便利です。 使用される認証メソッドに関わらず、Terraform を実行するアカウントはイントロダクションで説明されているすべてのコンポーネントを作成できる必要があります。

## 一般的な手順
このトピックの手順は、このドキュメントでカバーされているいずれのデプロイメント オプションにも共通しています。

1. 開発環境を準備します。
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
  * 使用するコードで Git リポジトリを作成することをお勧めしますが、ファイルをローカルに保持することもできます。

2. **`terraform.tfvars` ファイルを作成します** `tvfars` ファイルの内容はインストール タイプに応じてカスタマイズできますが、最低限の推奨事項は以下の例のようになります。

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   ここで定義されている変数は、デプロイメントの前に決定する必要があります。`namespace` 変数は、Terraform によって作成されるすべてのリソースの接頭辞となる文字列です。

   `subdomain` と `domain` の組み合わせは、W&B が設定される FQDN を形成します。上記の例では、W&B の FQDN は `wandb-aws.wandb.ml` となり、FQDN レコードが作成される DNS `zone_id` が指定されます。

3. **`versions.tf` ファイルを作成します** このファイルには、AWS に W&B をデプロイするのに必要な Terraform および Terraform プロバイダーのバージョンが含まれています。
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

  また、**強く推奨される** のは、ドキュメントの冒頭で言及された [リモート バックエンド設定](https://developer.hashicorp.com/terraform/language/backend) を追加することです。

4. **ファイル** `variables.tf` を作成します。`terraform.tfvars` で構成されたすべてのオプションについて、Terraform は対応する変数宣言を必要とします。

  ```bash
    variable "namespace" {
      type        = string
      description = "リソースの接頭辞に使用される文字列。"
    }

    variable "location" {
      type        = string
      description = "Azure リソース グループの場所"
    }

    variable "domain_name" {
      type        = string
      description = "Weights & Biases UI へのアクセス用ドメイン。"
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Weights & Biases UI へのアクセス用サブドメイン。デフォルトは Route53 Route でレコードを作成します。"
    }

    variable "license" {
      type        = string
      description = "あなたの wandb/local ライセンス"
    }
  ```

## 推奨デプロイメント

これは、すべての「必須」コンポーネントを作成し、最新バージョンの `W&B` を `Kubernetes クラスター` にインストールする最も簡単なデプロイメント オプション設定です。

1. **`main.tf` を作成します** `General Steps` で作成したファイルと同じディレクトリに、次の内容で `main.tf` ファイルを作成します：

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

  # 必要なすべてのサービスをスピンアップ
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

2. **W&B にデプロイ** W&B にデプロイするには、次のコマンドを実行します：

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを使用したデプロイメント

別のデプロイメント オプションとして、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスを読み込む際のアプリケーション応答を高速化します。

キャッシュを有効にするには、`recommended deployment`({{< relref path="#recommended-deployment" lang="ja" >}}) で使用したのと同じ `main.tf` ファイルに `create_redis = true` オプションを追加する必要があります。

```bash
# 必要なすべてのサービスをスピンアップ
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

デプロイメント オプション 3 は、外部の `message broker` を有効にすることです。 これはオプションであり、W&B にはブローカーが組み込まれているため、パフォーマンスの向上はもたらされません。

message broker を提供する Azure リソースは `Azure Event Grid` であり、有効にするには、`recommended deployment`({{< relref path="#recommended-deployment" lang="ja" >}}) で使用したのと同じ `main.tf` に `use_internal_queue = false` オプションを追加する必要があります。
```bash
# 必要なすべてのサービスをスピンアップ
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

## その他のデプロイメント オプション

3 つのデプロイメント オプションすべてを組み合わせて、すべての構成を同じファイルに追加できます。
[Teraform モジュール](https://github.com/wandb/terraform-azure-wandb) は、標準オプションや [recommended deployment]({{< relref path="#recommended-deployment" lang="ja" >}}) に見られる最小構成と組み合わせることができるいくつかのオプションを提供します。