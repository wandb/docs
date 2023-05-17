---
description: W&BサーバーをGCPにホスティングする方法。
---

# GCP

W&BサーバーをGoogle Cloudに展開するには、Weights and Biasesが開発した[Terraformモジュール](https://registry.terraform.io/modules/wandb/wandb/google/latest)の使用を推奨します。

モジュールのドキュメントは充実しており、使用可能なすべてのオプションが含まれています。このドキュメントでは、いくつかの展開オプションについて説明します。

始める前に、Terraformが[State File](https://developer.hashicorp.com/terraform/language/state)を保存するための[リモートバックエンド](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)のいずれかを選択することをお勧めします。

State Fileは、すべてのコンポーネントを再作成せずに展開でアップグレードや変更を適用するために必要なリソースです。

Terraformモジュールは、以下の `必須` コンポーネントを展開します:

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- ロードバランサー

他の展開オプションには、以下のオプションコンポーネントも含めることができます:

- Redis用のメモリストア
- Pub/Subメッセージシステム
## **前提条件としての権限**

Terraformを実行するアカウントは、使用するGCPプロジェクトで`roles/owner`の役割を持っている必要があります。

## 一般的な手順

このドキュメントでカバーされているデプロイメントオプションに共通の手順です。

1. 開発環境を準備する。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)をインストールする
   - 使用するコードを含むGitリポジトリを作成することをお勧めしますが、ローカルにファイルを保存してもかまいません。
   - [Google Cloud Console](https://console.cloud.google.com/)でプロジェクトを作成する
   - GCPで認証する（[gcloudをインストール](https://cloud.google.com/sdk/docs/install)してから実行してください）
     `gcloud auth application-default login`
2. `terraform.tfvars`ファイルを作成する。

   `tvfars`ファイルの内容は、インストールタイプに応じてカスタマイズできますが、最低限の推奨設定は以下の例のようになります。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義された変数は、展開前に決定しておく必要があります。`namespace`変数は、Terraformによって作成されるすべてのリソースに接頭辞として追加される文字列になります。

   `subdomain`と`domain`の組み合わせにより、W&Bが設定されるFQDNが形成されます。上記の例では、W&BのFQDNは`wandb-gcp.wandb.ml`になります。
3. `versions.tf`ファイルを作成する

   このファイルには、GCPでW&Bをデプロイするために必要なTerraformおよびTerraformプロバイダのバージョンが含まれます。

   ```
   terraform {
     required_version = "~> 1.0"
     required_providers {
       google = {
         source  = "hashicorp/google"
         version = "~> 4.15"
       }
       kubernetes = {
         source  = "hashicorp/kubernetes"
         version = "~> 2.9"
       }
     }
   }
   ```

   任意で、**ただし強くお勧めします**が、このドキュメントの始めに述べた[リモートバックエンド設定](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)を追加できます。

4. `variables.tf`ファイルを作成する

   `terraform.tfvars`で設定されたすべてのオプションに対して、Terraformは対応する変数宣言が必要です。

   ```
   variable "project_id" {
     type        = string
     description = "プロジェクトID"
   }
変数 "region" {
     type        = string
     description = "Google リージョン"
   }

   変数 "zone" {
     type        = string
     description = "Google ゾーン"
   }

   変数 "namespace" {
     type        = string
     description = "リソースに使用される名前空間の接頭語"
   }

   変数 "domain_name" {
     type        = string
     description = "Weights & Biases UIにアクセスするためのドメイン名。"
   }

   変数 "subdomain" {
     type        = string
     description = "Weights & Biases UIにアクセスするためのサブドメイン。"
   }

   変数 "license" {
     type        = string
     description = "W&B ライセンス"
   }
   ```
## 展開 - 推奨（〜20分）

この展開オプション設定は、最も簡単なもので、すべての`Mandatory`コンポーネントを作成し、`Kubernetes Cluster`に`W&B`の最新バージョンをインストールします。

1. `main.tf`を作成

   `General Steps`でファイルを作成した同じディレクトリーに、次の内容の`main.tf`ファイルを作成します：

   ```
   data "google_client_config" "current" {}

   provider "kubernetes" {
     host                   = "https://${module.wandb.cluster_endpoint}"
     cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
     token                  = data.google_client_config.current.access_token
   }

   # 必要なすべてのサービスを立ち上げる
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "1.12.2"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain

   }

   # 用意されたIPアドレスを使用してDNSを更新する必要があります
   output "url" {
     value = module.wandb.url
   }

output "address" {
     value = module.wandb.address
   }

   output "bucket_name" {
     value = module.wandb.bucket_name
   }
   ```

2. W&Bの展開

   W&Bを展開するには、以下のコマンドを実行してください:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDISの有効化

別の展開オプションでは、`Redis`を使用してSQLクエリをキャッシュし、実験のメトリクスをロードする際のアプリケーションの応答速度を向上させます。

キャッシュを有効にするために、`Deployment option 1`で作業したのと同じ`main.tf`ファイルに`create_redis = true`オプションを追加する必要があります。

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "1.12.2"

namespace    = var.namespace
  license      = var.license
  domain_name  = var.domain_name
  subdomain    = var.subdomain
  #Redisを有効化
  create_redis = true

}
[...]
```

## メッセージブローカ（キュー）を有効にする

展開オプション3は、外部の`メッセージブローカ`を有効にするものです。これはオプションであり、W&Bにはすでにブローカが組み込まれているためです。このオプションはパフォーマンスの向上をもたらさないことに注意してください。

GCPでメッセージブローカを提供するリソースは`Pub/Sub`であり、これを有効にするには、`Deployment option 1`で取り扱った同じ`main.tf`に`use_internal_queue = false`オプションを追加する必要があります。

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "1.12.2"

  namespace          = var.namespace
  license            = var.license
  domain_name        = var.domain_name
  subdomain          = var.subdomain
  # Pub/Subの作成と使用
  use_internal_queue = false

}



[...]



```



## その他の展開オプション



すべての設定を同じファイルに追加して、3つの展開オプションを組み合わせることができます。

[テラフォームモジュール](https://github.com/wandb/terraform-google-wandb)では、`Deployment - Recommended`にある標準オプションと最小構成と組み合わせて使用できるいくつかのオプションが提供されています。



<!-- ## アップグレード（近日公開予定） -->