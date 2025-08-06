---
title: GCP に W&B プラットフォームをデプロイする
description: GCP で W&B サーバーをホスティングする
menu:
  default:
    identifier: gcp-tf
    parent: install-on-public-cloud
weight: 20
---

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) や [W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) といった、フルマネージドなデプロイメントオプションを推奨しています。W&B のフルマネージドサービスは、シンプルかつセキュアにご利用いただけ、ほとんどの場合設定不要ですぐに使い始められます。
{{% /alert %}}


セルフマネージド型の W&B Server をご利用される場合は、[W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) を使い、GCP 上にプラットフォームをデプロイすることを推奨します。

このモジュールのドキュメントには、利用可能な全てのオプション詳細が網羅されています。

開始する前に、Terraform が管理する [State File](https://developer.hashicorp.com/terraform/language/state) を保存するための [remote backend](https://developer.hashicorp.com/terraform/language/backend/remote) を選択してください。

State File は、デプロイメントの際、構成変更やアップグレードをリソースを再作成せずに安全に反映させるために必要となります。

Terraform Module では、以下の `必須` コンポーネントがデプロイされます:

- VPC
- MySQL 用 Cloud SQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- ロードバランサー

また、その他のオプションコンポーネントも追加でデプロイすることが可能です:

- Redis 用メモリストア
- Pub/Sub メッセージシステム

## 前提となる権限

Terraform を実行するアカウントには、利用する GCP プロジェクトで `roles/owner` のロールが必要です。

## 全体の手順

ここで説明する手順は、このドキュメントでカバーされているどのデプロイメントオプションにも共通しています。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールしてください
   - 利用予定のコードを管理する Git リポジトリを作成することを推奨しますが、ローカルにファイルを保存しても構いません。
   - [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成します
   - GCP に認証します（事前に [gcloud のインストール](https://cloud.google.com/sdk/docs/install) を済ませてください）
     `gcloud auth application-default login`
2. `terraform.tfvars` ファイルを作成します。

   `tfvars` ファイルの内容は、インストールタイプに応じてカスタマイズ可能ですが、最低限必要となる推奨例は下記の通りです。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義する変数は、デプロイ前に決めておく必要があります。`namespace` 変数は、Terraform が作成する全リソースのプレフィックスとなる文字列です。

   `subdomain` と `domain` の組み合わせが、W&B の FQDN になります。上記例では FQDN は `wandb-gcp.wandb.ml` となります。

3. `variables.tf` ファイルを作成します。

   `terraform.tfvars` で設定したオプションごとに、対応する変数宣言が必要です。

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
     description = "リソースで使用する namespace プレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "Weights & Biases UI にアクセスするためのドメイン名"
   }

   variable "subdomain" {
     type        = string
     description = "Weights & Biases UI にアクセスするためのサブドメイン"
   }

   variable "license" {
     type        = string
     description = "W&B ライセンス"
   }
   ```

## デプロイメント - 推奨構成 (約20分)

この方法が最もシンプルで、`必須` コンポーネントをすべて作成し、`Kubernetes クラスター` 上に最新バージョンの `W&B` をインストールします。

1. `main.tf` ファイルを作成します。

   [全体の手順]({{< relref "#general-steps" >}})で作成したファイルと同じディレクトリーに、下記内容で `main.tf` ファイルを作成してください。

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

   # 必要なサービスをすべて起動
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "~> 5.0"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain
   }

   # DNS をプロビジョニングされた IP アドレスで更新する
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

2. W&B をデプロイします

   W&B をデプロイするには、次のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを利用したデプロイメント

もう一つのデプロイメントオプションとして、`Redis` を用いて SQL クエリをキャッシュし、実験のメトリクス読込時のアプリケーション応答速度を向上させる方法があります。

キャッシュを利用するには、推奨される [デプロイメントオプションセクション]({{< relref "#deployment---recommended-20-mins" >}}) の `main.tf` ファイルに `create_redis = true` オプションを追加してください。

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
  # Redis を有効化
  create_redis = true

}
[...]
```

## 外部キューを利用したデプロイメント

3つめのオプションは、外部 `message broker` を有効化する構成です。W&B には標準でブローカーが組み込まれているため、このオプションは必須ではなく、パフォーマンス向上もありません。

GCP でメッセージブローカーとして使えるリソースは `Pub/Sub` で、有効化するには、推奨される [デプロイメントオプションセクション]({{< relref "#deployment---recommended-20-mins" >}}) の `main.tf` ファイルに `use_internal_queue = false` を追加してください。

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
  # Pub/Sub の作成・利用
  use_internal_queue = false

}

[...]

```

## その他のデプロイメントオプション

3つのデプロイメントオプションは、全て同時に設定ファイルへ追加し組み合わせることも可能です。
[Terraform Module](https://github.com/wandb/terraform-google-wandb) には、多様な設定項目が揃っており、`Deployment - Recommended` の最小構成と併せて柔軟に組み合わせ利用できます。



## マニュアル設定

W&B のファイルストレージのバックエンドとして GCP Storage bucket を利用する場合、次のリソースを作成する必要があります:

* [PubSub トピックとサブスクリプションの作成]({{< relref "#create-pubsub-topic-and-subscription" >}})
* [Storage Bucket の作成]({{< relref "#create-storage-bucket" >}})
* [PubSub Notification の作成]({{< relref "#create-pubsub-notification" >}})


### PubSub トピックとサブスクリプションの作成

下記手順に従い PubSub トピックとサブスクリプションを作成します。

1. GCP コンソールの Pub/Sub サービスに移動します
2. **Create Topic** を選択し、トピック名を入力します
3. ページ下部で **Create subscription** を選び、**Delivery Type** は **Pull** のままにします。
4. **Create** をクリック。

サービスアカウントもしくはインスタンスを実行しているアカウントが、このサブスクリプションで `pubsub.admin` 権限を持っていることを確認してください。詳細は https://cloud.google.com/pubsub/docs/access-control#console を参照してください。

### Storage Bucket の作成

1. **Cloud Storage Buckets** ページへアクセスします。
2. **Create bucket** を選択しバケット名を入力します。**Standard** [storage class](https://cloud.google.com/storage/docs/storage-classes) を必ず選択してください。

インスタンスが実行されているサービスアカウント、またはアカウントには下記権限が必要です:
* 前ステップで作成したバケットへのアクセス権
* そのバケットで `storage.objectAdmin` ロールを持っていること。詳細は https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照してください。

{{% alert %}}
署名付きファイルURLを作成するには、GCP で `iam.serviceAccounts.signBlob` 権限がインスタンスにも必要です。Service Account Token Creator ロールを当該サービスアカウントまたは IAM メンバーに付与してください。
{{% /alert %}}

3. CORS アクセスを有効にします。これはコマンドラインのみで設定可能です。まず、次のような CORS 設定の JSON ファイルを作成してください。

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

origin の値は、スキーマ・ホスト・ポートが完全一致する必要があります。

4. `gcloud` がインストール済みで、正しい GCP プロジェクトにログインしていることを確認しましょう。
5. 次のコマンドを実行します。

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub Notification の作成
コマンドラインで Storage Bucket から Pub/Sub トピックへの通知ストリームを作成します。

{{% alert %}}
通知ストリームは CLI でのみ作成可能です。`gcloud` をインストールしてください。
{{% /alert %}}

1. ご自身の GCP プロジェクトへログインします。
2. 下記をターミナルで実行します。

```bash
gcloud pubsub topics list  # トピック名を参照用に一覧表示
gcloud storage ls          # バケット名を参照用に一覧表示

# バケット通知作成
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[更なる参考資料は Cloud Storage サイトをご参照ください。](https://cloud.google.com/storage/docs/reporting-changes)

### W&B サーバーの設定

1. 最後に、ご自身の W&B `System Connections` ページ（`http(s)://YOUR-W&B-SERVER-HOST/console/settings/system`）へアクセスします。
2. プロバイダーとして `Google Cloud Storage (gcs)` を選択します。
3. 対象の GCS バケット名を入力します

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="GCP file storage configuration" >}}

4. **Update settings** を押して設定を反映させてください。

## W&B Server のアップグレード方法

W&B をアップデートする場合は、下記手順に従ってください。

1. 設定ファイルの `wandb_app` モジュールに `wandb_version` を追加し、アップグレードしたいバージョンを指定します。例えば、`0.48.1` バージョンへのアップグレードの場合、以下のように記述します。

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  もしくは、`wandb_version` を `terraform.tfvars` に追加し、同名の変数を作成すればリテラル値の代わりに `var.wandb_version` を使用することができます。
  {{% /alert %}}

2. 設定を更新した後は、[デプロイメントオプションセクション]({{< relref "#deployment---recommended-20-mins" >}})で記載された手順を再度実行してください。