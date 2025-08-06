---
title: GCP で W&B プラットフォームをデプロイする
description: GCP で W&B サーバーをホスティングする
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-gcp-tf
    parent: install-on-public-cloud
weight: 20
---

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) などの完全マネージド型デプロイメントオプションを推奨しています。W&B の完全マネージドサービスはシンプルかつセキュアで、設定もほとんど、あるいは全く不要でご利用いただけます。
{{% /alert %}}

W&B Server をセルフマネージドで運用する場合は、[W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) を利用して GCP 上にプラットフォームをデプロイすることを推奨します。

このモジュールのドキュメントは充実しており、利用可能なすべてのオプションが記載されています。

開始する前に、Terraform の [リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend/remote) のいずれかを選択して、[ステートファイル](https://developer.hashicorp.com/terraform/language/state) を保存することをおすすめします。

ステートファイルは、デプロイメント時にすべてのコンポーネントを再作成することなく、アップグレードや変更を行うために必要なリソースです。

Terraform モジュールは、以下の `必須` コンポーネントをデプロイします:

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

その他のデプロイメントオプションでは、次のオプショナルコンポーネントも含めることができます:

- Redis 用 Memory store
- Pub/Sub メッセージシステム

## 必要な権限

Terraform を実行するアカウントには、利用する GCP プロジェクトで `roles/owner` ロールが必要です。

## 全体の流れ

ここで紹介する手順は、本ドキュメントで取り扱う全てのデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストール
   - 利用するコードを Git リポジトリで管理することを推奨しますが、ローカルファイルでも問題ありません。
   - [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成
   - GCP への認証（事前に [gcloud のインストール](https://cloud.google.com/sdk/docs/install) を行ってください）
     `gcloud auth application-default login`
2. `terraform.tfvars` ファイルを作成

   `tfvars` ファイルの内容はインストールタイプによってカスタマイズできますが、最低限推奨される内容は以下の通りです。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義する変数はデプロイメント前に決めておく必要があります。`namespace` 変数は Terraform が作成する全てのリソースに接頭辞として付与されます。

   `subdomain` と `domain` の組み合わせが W&B の FQDN となります。上記例の場合、W&B の FQDN は `wandb-gcp.wandb.ml` です。

3. `variables.tf` ファイルを作成

   `terraform.tfvars` で設定した各オプションには、Terraform 上で対応する変数宣言が必要です。

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

## デプロイメント ― 推奨構成（約20分）

この構成は最もシンプルなデプロイメントオプションで、`必須` の全コンポーネントを作成し、`Kubernetes クラスター` に最新バージョンの `W&B` をインストールします。

1. `main.tf` の作成

   [全体の流れ]({{< relref path="#general-steps" lang="ja" >}}) の手順で作成したファイルと同じディレクトリーに、下記内容の `main.tf` ファイルを作成します。

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

   # プロビジョニングされた IP アドレスで DNS を更新してください
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

2. W&B のデプロイ

   W&B をデプロイするには、以下のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを有効にしたデプロイメント

もう一つのデプロイメントオプションとして、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスを読み込む際のアプリケーションの応答速度を向上させる方法があります。

キャッシュを有効にするには、推奨 [デプロイメントオプションのセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で指定したのと同じ `main.tf` ファイルに `create_redis = true` オプションを追加してください。

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

デプロイメントオプション3では、外部の `メッセージブローカー` を有効化します。W&B には組み込みブローカーがあるため、こちらはオプショナルで、パフォーマンス向上はありません。

GCP でメッセージブローカーを提供するリソースは `Pub/Sub` で、これを有効にするには、推奨 [デプロイメントオプションのセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) の `main.tf` に `use_internal_queue = false` オプションを追加してください。

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
  # Pub/Sub を作成・使用
  use_internal_queue = false

}

[...]

```

## その他のデプロイメントオプション

3つのデプロイメントオプションは、すべて同じファイルに設定を追加して組み合わせることが可能です。[Terraform モジュール](https://github.com/wandb/terraform-google-wandb) には、標準オプションや `Deployment - Recommended` の最小構成と組み合わせて利用できる、さまざまなオプションが用意されています。


## 手動による設定

GCP Storage バケットを W&B のファイルストレージバックエンドとして利用するには、以下を作成する必要があります:

* [PubSub トピックとサブスクリプション]({{< relref path="#create-pubsub-topic-and-subscription" lang="ja" >}})
* [Storage バケット]({{< relref path="#create-storage-bucket" lang="ja" >}})
* [PubSub 通知]({{< relref path="#create-pubsub-notification" lang="ja" >}})

### PubSub トピックとサブスクリプションの作成

以下の手順で PubSub のトピックとサブスクリプションを作成してください。

1. GCP Console の Pub/Sub サービスに移動
2. **Create Topic** を選択し、トピック名を入力
3. ページ下部で **Create subscription** を選択し、**Delivery Type** を **Pull** に設定
4. **Create** をクリック

サービスアカウントまたはインスタンスで利用しているアカウントには、当該サブスクリプションに `pubsub.admin` ロールが付与されている必要があります。詳細は https://cloud.google.com/pubsub/docs/access-control#console をご覧ください。

### Storage バケットの作成

1. **Cloud Storage Buckets** ページに移動
2. **Create bucket** を選択し、バケット名を入力。必ず **Standard** [ストレージクラス](https://cloud.google.com/storage/docs/storage-classes) を選択してください。

インスタンスで利用しているサービスアカウントまたはアカウントには、下記の権限が必要です:
* 先ほど作成したバケットへのアクセス
* このバケットでの `storage.objectAdmin` ロール（詳細は https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照）

{{% alert %}}
サイン付きファイル URL を作成するには、インスタンスに `iam.serviceAccounts.signBlob` 権限も必要です。サービスアカウントまたは IAM メンバーに `Service Account Token Creator` ロールを付与してください。
{{% /alert %}}

3. CORS アクセスの有効化。これはコマンドラインのみで可能です。まず、下記内容の CORS 設定ファイル（JSON）を作成します。

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

origin の値には、スキーマ・ホスト・ポートが完全一致する必要があります。

4. `gcloud` がインストールされ、正しい GCP プロジェクトにログインしていることを確認
5. 次に、以下を実行します

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub 通知の作成
Storage Bucket から Pub/Sub トピックへの通知ストリームを作成するには、下記手順をコマンドラインで実行します。

{{% alert %}}
通知ストリーム作成には CLI が必要です。`gcloud` がインストールされていることを確認してください。
{{% /alert %}}

1. GCP プロジェクトにログイン
2. ターミナルで下記を実行:

```bash
gcloud pubsub topics list  # 参照用にトピック名を一覧表示
gcloud storage ls          # 参照用にバケット名を一覧表示

# バケット通知の作成
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[詳細は Cloud Storage 公式サイトをご参照ください。](https://cloud.google.com/storage/docs/reporting-changes)

### W&B サーバーの設定

1. 最後に、W&B の `System Connections` ページ （`http(s)://YOUR-W&B-SERVER-HOST/console/settings/system`）にアクセス
2. プロバイダで `Google Cloud Storage (gcs)` を選択
3. GCS バケット名を入力

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="GCP ファイルストレージ設定" >}}

4. **Update settings** を押して設定を反映します。

## W&B サーバーのアップグレード

W&B をアップデートするには、以下の手順に従ってください。

1. `wandb_app` モジュールの設定に `wandb_version` を追加し、アップグレードしたい W&B のバージョンを指定。例として、W&B バージョン `0.58.1` を指定する場合は以下のようになります。

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  また、`terraform.tfvars` に `wandb_version` を追加し、同名の変数を設定することで、リテラル値の代わりに `var.wandb_version` を利用することも可能です。
  {{% /alert %}}

2. 設定を更新した後、[デプロイメントオプションのセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で説明されている手順を完了してください。