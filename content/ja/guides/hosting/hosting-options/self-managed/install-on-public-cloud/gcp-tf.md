---
title: W&B プラットフォームを GCP にデプロイする
description: GCP 上で W&B サーバーをホストする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-gcp-tf
    parent: install-on-public-cloud
weight: 20
---

{{% alert %}}
W&B は、[W&B マルチテナント クラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B 専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドなデプロイメントを推奨しています。W&B のフルマネージド サービスは、設定が最小限または不要で、簡単かつ安全にご利用いただけます。
{{% /alert %}}

W&B Server をセルフマネージドで運用する場合は、GCP 上にプラットフォームをデプロイするために [W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) の使用を推奨します。

このモジュールのドキュメントには、利用可能なすべてのオプションが詳しく記載されています。

開始前に、Terraform の [remote backends](https://developer.hashicorp.com/terraform/language/backend/remote) のいずれかを選択し、[State File](https://developer.hashicorp.com/terraform/language/state) を保存することを推奨します。

State File は、すべてのコンポーネントを作り直すことなく、アップグレードの展開やデプロイメントの変更を行うために必要なリソースです。

Terraform Module は次の「必須」コンポーネントをデプロイします:

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

その他のデプロイメントでは、次の任意コンポーネントも含められます:

- MemoryStore for Redis
- Pub/Sub メッセージ システム

## 前提となる権限

Terraform を実行するアカウントには、対象の GCP プロジェクトで `roles/owner` ロールが必要です。

## 全体の手順 {#general-steps}

このトピックの手順は、本ドキュメントで扱うすべてのデプロイメント オプションで共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストール
   - 使用するコードを保存する Git リポジトリの作成を推奨しますが、ローカルにファイルを置いても構いません。
   - [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成
   - GCP に認証します（事前に [gcloud をインストール](https://cloud.google.com/sdk/docs/install) しておいてください）
     `gcloud auth application-default login`
2. `terraform.tfvars` ファイルを作成します。

   `tvfars` ファイルの内容はインストール種別に応じてカスタマイズできますが、最小構成の例は以下のとおりです。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義する変数はデプロイ前に決めておく必要があります。`namespace` 変数は、Terraform が作成するすべてのリソースに付与される接頭辞となる文字列です。

   `subdomain` と `domain` の組み合わせが、W&B を設定する FQDN になります。上記の例では、W&B の FQDN は `wandb-gcp.wandb.ml` です。

3. `variables.tf` ファイルを作成します。

   `terraform.tfvars` で設定した各オプションには、対応する変数宣言が Terraform に必要です。

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

## デプロイメント - 推奨（約 20 分） {#deployment---recommended-20-mins}

これは最もシンプルなデプロイメント構成で、すべての「必須」コンポーネントを作成し、`Kubernetes Cluster` に最新の `W&B` をインストールします。

1. `main.tf` を作成

   [全体の手順]({{< relref path="#general-steps" lang="ja" >}}) で作成したのと同じディレクトリーに、以下の内容で `main.tf` ファイルを作成します。

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

   # 付与された IP アドレスで DNS を更新してください
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

   W&B をデプロイするには、次のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを使ったデプロイメント

別のデプロイメント オプションとして、`Redis` を用いて SQL クエリをキャッシュし、Experiments のメトリクスを読み込む際のアプリケーションの応答を高速化できます。

キャッシュを有効にするには、推奨の [デプロイメント オプション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で示したのと同じ `main.tf` に `create_redis = true` オプションを追加します。

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

## 外部キューを使ったデプロイメント

デプロイメント オプション 3 は、外部の `message broker` を有効にする構成です。W&B には組み込みのブローカーがあるため任意であり、このオプションによるパフォーマンス向上はありません。

メッセージ ブローカーを提供する GCP のリソースは `Pub/Sub` です。これを有効にするには、推奨の [デプロイメント オプション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) と同じ `main.tf` に `use_internal_queue = false` を追加します。

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
  # Pub/Sub を作成して使用する
  use_internal_queue = false

}

[...]

```

## その他のデプロイメント オプション

3 つのデプロイメント オプションは、同じファイルに設定を追加することで組み合わせて利用できます。
[Terraform Module](https://github.com/wandb/terraform-google-wandb) には、標準オプションや「デプロイメント - 推奨」の最小構成と組み合わせ可能な、いくつかの追加オプションが用意されています。

## 手動設定

W&B のファイル ストレージのバックエンドとして GCP Storage のバケットを使用するには、以下を作成する必要があります:

* [Pub/Sub トピックとサブスクリプション]({{< relref path="#create-pubsub-topic-and-subscription" lang="ja" >}})
* [Storage バケット]({{< relref path="#create-storage-bucket" lang="ja" >}})
* [Pub/Sub 通知]({{< relref path="#create-pubsub-notification" lang="ja" >}})

### Pub/Sub トピックとサブスクリプションを作成 {#create-pubsub-topic-and-subscription}

以下の手順に従って、Pub/Sub のトピックとサブスクリプションを作成します。

1. GCP Console 内の Pub/Sub サービスに移動します。
2. **Create Topic** を選択し、トピック名を指定します。
3. ページ下部で **Create subscription** を選択します。**Delivery Type** は **Pull** に設定してください。
4. **Create** をクリックします。

インスタンスが実行されているサービス アカウントまたはアカウントに、このサブスクリプションに対する `pubsub.admin` ロールが付与されていることを確認してください。詳細は https://cloud.google.com/pubsub/docs/access-control#console を参照してください。

### Storage バケットを作成 {#create-storage-bucket}

1. **Cloud Storage Buckets** ページに移動します。
2. **Create bucket** を選択し、バケット名を指定します。**Standard** の [storage class](https://cloud.google.com/storage/docs/storage-classes) を選択してください。

インスタンスが実行されているサービス アカウントまたはアカウントに、以下の権限があることを確認してください:
* 直前の手順で作成したバケットへのアクセス
* このバケットに対する `storage.objectAdmin` ロール。詳細は https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照してください。

{{% alert %}}
署名付きファイル URL を作成するには、インスタンスに GCP の `iam.serviceAccounts.signBlob` 権限が必要です。権限を有効にするには、インスタンスが実行されているサービス アカウントまたは IAM メンバーに `Service Account Token Creator` ロールを付与してください。
{{% /alert %}}

3. CORS アクセスを有効にします。これはコマンドラインからのみ実行できます。まず、以下の CORS 設定で JSON ファイルを作成します。

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

origin の値のスキーム、ホスト、ポートは完全に一致している必要があります。

4. `gcloud` がインストールされ、正しい GCP プロジェクトにログインしていることを確認します。
5. 次に、以下を実行します。

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### Pub/Sub 通知を作成 {#create-pubsub-notification}
コマンドラインで以下の手順に従い、Storage バケットから Pub/Sub トピックへの通知ストリームを作成します。

{{% alert %}}
通知ストリームの作成は CLI を使用する必要があります。`gcloud` がインストールされていることを確認してください。
{{% /alert %}}

1. GCP プロジェクトにログインします。
2. ターミナルで以下を実行します。

```bash
gcloud pubsub topics list  # 参照用にトピック名を一覧表示
gcloud storage ls          # 参照用にバケット名を一覧表示

# バケット通知を作成
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[Cloud Storage サイトに詳細なリファレンスがあります。](https://cloud.google.com/storage/docs/reporting-changes)

### W&B サーバーを設定

1. 最後に、W&B の `System Connections` ページ `http(s)://YOUR-W&B-SERVER-HOST/console/settings/system` に移動します。
2. プロバイダーに `Google Cloud Storage (gcs)` を選択します。
3. GCS バケット名を入力します。

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="GCP ファイル ストレージの設定" >}}

4. **Update settings** を押して、新しい設定を適用します。

## W&B Server をアップグレード

W&B を更新するには、以下の手順に従ってください。

1. `wandb_app` モジュールの設定に `wandb_version` を追加し、アップグレード先の W&B のバージョンを指定します。例えば、次の行は W&B バージョン `0.48.1` を指定しています:

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  代わりに、`terraform.tfvars` に `wandb_version` を追加し、同名の変数を作成して、リテラル値の代わりに `var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 設定を更新したら、[デプロイメント オプション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) に記載の手順を実行します。