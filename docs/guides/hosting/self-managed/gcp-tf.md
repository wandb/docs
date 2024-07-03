---
title: GCP
description: GCPでのW&Bサーバーのホスティング
displayed_sidebar: default
---

# GCP

:::info
W&B 推奨のデプロイメントオプションとして、[W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) または [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) があります。W&B のフルマネージドサービスは、設定がほとんど不要で簡単かつ安全に使用できます。
:::

自己管理の W&B サーバーを選択した場合、W&B は [W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) を使用して GCP にプラットフォームをデプロイすることを推奨します。

モジュールのドキュメントは非常に充実しており、使用可能なすべてのオプションが含まれています。このドキュメントではいくつかのデプロイメントオプションをカバーします。

開始する前に、Terraform の [リモートバックエンド](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) のいずれかを選択して、[ステートファイル](https://developer.hashicorp.com/terraform/language/state) を保存することをお勧めします。

ステートファイルは、すべてのコンポーネントを再作成することなく、アップグレードやデプロイメントの変更を行うために必要なリソースです。

Terraform モジュールは以下の必須コンポーネントをデプロイします:

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

他のデプロイメントオプションでは、以下のオプションコンポーネントも含めることができます:

- Redis 用のメモリーストア
- Pub/Sub メッセージシステム

## **前提条件の権限**

Terraform を実行するアカウントには、使用する GCP プロジェクトで `roles/owner` ロールを持っている必要があります。

## 一般的な手順

このトピックの手順は、ドキュメントでカバーされているすべてのデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストール
   - 使用するコードで Git リポジトリを作成することをお勧めしますが、ローカルにファイルを保持することもできます。
   - [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成
   - GCP で認証（事前に [gcloud をインストール](https://cloud.google.com/sdk/docs/install) しておくこと）
     `gcloud auth application-default login`
2. `terraform.tfvars` ファイルを作成します。

   `tfvars` ファイルの内容はインストールタイプに応じてカスタマイズできますが、最低限の推奨内容は以下の例のようになります。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義する変数はデプロイメント前に決定する必要があります。`namespace` 変数は Terraform が作成するすべてのリソースのプレフィックスとなる文字列です。

   `subdomain` と `domain` の組み合わせが W&B が設定される FQDN を形成します。上記の例では、W&B の FQDN は `wandb-gcp.wandb.ml` になります。

3. `variables.tf` ファイルを作成します。

   `terraform.tfvars` で設定されたすべてのオプションに対して、Terraform では対応する変数宣言が必要です。

   ```
   variable "project_id" {
     type        = string
     description = "Project ID"
   }

   variable "region" {
     type        = string
     description = "Google region"
   }

   variable "zone" {
     type        = string
     description = "Google zone"
   }

   variable "namespace" {
     type        = string
     description = "Namespace prefix used for resources"
   }

   variable "domain_name" {
     type        = string
     description = "Domain name for accessing the Weights & Biases UI."
   }

   variable "subdomain" {
     type        = string
     description = "Subdomain for access the Weights & Biases UI."
   }

   variable "license" {
     type        = string
     description = "W&B License"
   }
   ```

## デプロイメント - 推奨 (~20分)

これは最も簡単なデプロイメントオプション設定で、すべての `必須` コンポーネントを作成し、`Kubernetes クラスター` に最新の `W&B` バージョンをインストールします。

1. `main.tf` を作成します。

   `一般的な手順` でファイルを作成した同じディレクトリーに `main.tf` ファイルを以下の内容で作成します。

   ```
   provider "google" {
    project = var.project_id
    region  = var.region
    zone    = var.zone
   }

   provider "google-beta" {
    project = var.project_id
    region  = var.region
    zone    = var.zone
   }

   data "google_client_config" "current" {}

   provider "kubernetes" {
     host                   = "https://${module.wandb.cluster_endpoint}"
     cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
     token                  = data.google_client_config.current.access_token
   }

   # 必要なすべてのサービスを立ち上げる
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "~> 1.0"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain
     allowed_inbound_cidrs = ["*"]
   }

   # プロビジョニングされた IP アドレスで DNS を更新します
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

2. W&B をデプロイ

   W&B をデプロイするには、以下のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを使用したデプロイメント

別のデプロイメントオプションとして `Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスのロード時にアプリケーションの応答速度を向上させます。

`main.tf` ファイルに `create_redis = true` オプションを追加することでキャッシュを有効にします。このファイルは `デプロイメントオプション 1` で使用したものです。

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "~> 1.0"

  namespace    = var.namespace
  license      = var.license
  domain_name  = var.domain_name
  subdomain    = var.subdomain
  allowed_inbound_cidrs = ["*"]
  # Redisを有効にする
  create_redis = true

}
[...]
```

## 外部キューを使用したデプロイメント

デプロイメントオプション 3 では、外部の `メッセージブローカー` を有効にします。これはオプションで、W&B には組み込みのブローカーが付属しています。このオプションはパフォーマンスの向上をもたらしません。

GCP リソースでメッセージブローカーを提供するのは `Pub/Sub` です。これを有効にするには `use_internal_queue = false` オプションを、`デプロイメントオプション 1` で使用した `main.tf` に追加する必要があります。

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "~> 1.0"

  namespace          = var.namespace
  license            = var.license
  domain_name        = var.domain_name
  subdomain          = var.subdomain
  allowed_inbound_cidrs = ["*"]
  # Pub/Subを作成して使用する
  use_internal_queue = false

}

[...]

```

## その他のデプロイメントオプション

すべてのデプロイメントオプションを組み合わせて、同じファイルにすべての設定を追加できます。
[Terraform Module](https://github.com/wandb/terraform-google-wandb) は、標準オプションや `推奨デプロイメント` の最小構成と組み合わせることができる複数のオプションを提供します。

## 手動設定

W&B のファイルストレージバックエンドとして GCP Storage バケットを使用するには、以下を作成する必要があります:

* [PubSub トピックとサブスクリプション](#create-pubsub-topic-and-subscription)
* [Storage Bucket](#create-storage-bucket)
* [PubSub Notification](#create-pubsub-notification)

### PubSub トピックとサブスクリプションの作成

以下の手順を実行して PubSub トピックとサブスクリプションを作成します。

1. GCP コンソールで Pub/Sub サービスに移動します。
2. **Create Topic** を選択し、トピックに名前を付けます。
3. ページの下部で **Create subscription** を選択します。**Delivery Type** が **Pull** に設定されていることを確認します。
4. **Create** をクリックします。

サービスアカウントまたはインスタンスを実行しているアカウントが、このサブスクリプションに対する `pubsub.admin` ロールを持っていることを確認してください。詳細は https://cloud.google.com/pubsub/docs/access-control#console を参照してください。

### ストレージバケットの作成

1. **Cloud Storage Buckets** ページに移動します。
2. **Create bucket** を選択し、バケットに名前を付けます。**Standard** [ストレージクラス](https://cloud.google.com/storage/docs/storage-classes) を選択することを確認します。

インスタンスを実行しているサービスアカウントまたはアカウントが以下の権限を持っていることを確認してください:
* 前のステップで作成したバケットへのアクセス
* このバケットに対する `storage.objectAdmin` ロール。詳細は https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照してください。

:::info
インスタンスは署名付きファイル URL を作成するために GCP での `iam.serviceAccounts.signBlob` 権限も必要です。サービスアカウントまたはインスタンスを実行している IAM メンバーに `Service Account Token Creator` ロールを追加して権限を有効にします。
:::

3. CORS アクセスを有効にします。これはコマンドラインでのみ実行できます。まず、以下の CORS 設定を持つ JSON ファイルを作成します。

```
cors:
- maxAgeSeconds: 3600
  method:
   - GET
   - PUT
     origin:
   - '<YOUR_W&B_SERVER_HOST>'
     responseHeader:
   - Content-Type
```

オリジンの値のスキーム、ホスト、ポートが正確に一致することを確認してください。

4. `gcloud` がインストールされ、正しい GCP プロジェクトにログインしていることを確認します。
5. 次に、以下を実行します。

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub Notification の作成
コマンドラインで以下の手順を実行して、Storage Bucket から Pub/Sub トピックへの通知ストリームを作成します。

:::info
通知ストリームを作成するには CLI を使用する必要があります。`gcloud` がインストールされていることを確認してください。
:::

1. GCP プロジェクトにログインします。
2. ターミナルで以下を実行します。

```bash
gcloud pubsub topics list  # トピック名をリストするための参照
gcloud storage ls          # バケット名をリストするための参照

# バケット通知を作成
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[Cloud Storage サイトで追加の参考資料をご確認ください。](https://cloud.google.com/storage/docs/reporting-changes)

### W&B サーバーを構成

1. 最後に、`http(s)://YOUR-W&B-SERVER-HOST/system-admin` の W&B 設定ページに移動します。
2. 「外部ファイルストレージバックエンドを使用する」オプションを有効にします。
3. 次の形式で AWS S3 バケットの名前、バケットが保存されているリージョン、および SQS キューを指定します：
* **File Storage Bucket**: `gs://<bucket-name>`
* **File Storage Region**: 空白
* **Notification Subscription**: `pubsub:/<project-name>/<topic-name>/<subscription-name>`

![](/images/hosting/configure_file_store.png)

4. **Update settings** を押して新しい設定を適用します。

## W&B サーバーのアップグレード

以下の手順に従って W&B をアップデートします：

1. あなたの `wandb_app` モジュールの設定に `wandb_version` を追加します。アップグレードしたいバージョンの W&B を指定します。例えば、次の行は W&B バージョン `0.48.1` を指定します：

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  :::info
  代わりに、`wandb_version` を `terraform.tfvars` に追加し、同じ名前の変数を作成して、リテラル値の代わりに `var.wandb_version` を使用できます。
  :::

2. 設定を更新したら、[デプロイメントセクション](#deployment---recommended-20-mins) に記載されている手順を完了します。