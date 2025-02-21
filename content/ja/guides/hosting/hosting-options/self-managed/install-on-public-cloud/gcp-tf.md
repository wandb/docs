---
title: Deploy W&B Platform on GCP
description: W&B サーバーを GCP でホスティングする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-gcp-tf
    parent: install-on-public-cloud
weight: 20
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) デプロイメントタイプのような完全管理されたデプロイメントオプションを推奨しています。W&B の完全管理サービスは、設定がほとんど必要なく、簡単かつ安全に使用できます。
{{% /alert %}}

W&B Server を自己管理することを決定した場合、W&B は、GCP 上にプラットフォームをデプロイするために [W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) を使用することを推奨しています。

モジュールのドキュメントは非常に詳細で、使用可能なオプションがすべて含まれています。

作業を開始する前に、W&B は、Terraform で使用可能な [remote backends](https://developer.hashicorp.com/terraform/language/backend/remote) のいずれかを選択して [State File](https://developer.hashicorp.com/terraform/language/state) を保存することを推奨しています。

State File は、すべてのコンポーネントを再作成することなく、アップグレードを実施したり、デプロイメントに変更を加えたりするために必要なリソースです。

Terraform Module は、以下の `mandatory` コンポーネントをデプロイします：

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

その他のデプロイメントオプションには以下のオプションコンポーネントも含まれます：

- Memory store for Redis
- Pub/Sub メッセージシステム

## 前提条件の権限

Terraform を実行するアカウントは、使用する GCP プロジェクトでの `roles/owner` ロールを持っている必要があります。

## 一般的な手順

このトピックの手順は、このドキュメントでカバーされているすべてのデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)をインストールします。
   - 使用するコードを Git リポジトリに作成することを推奨しますが、ファイルをローカルに保存することもできます。
   - [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成します。
   - GCP に認証します (事前に [gcloud をインストール](https://cloud.google.com/sdk/docs/install) しておくことが必要です)。
     `gcloud auth application-default login`
2. `terraform.tfvars` ファイルを作成します。

   `tfvars` ファイルの内容はインストールタイプに応じてカスタマイズできますが、最低限の推奨事項は次の例のようになります。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義されている変数は、デプロイメントの前に決定する必要があります。`namespace` 変数は、Terraform により作成されるすべてのリソースのプレフィックスとなる文字列になります。

   `subdomain` と `domain` の組み合わせで W&B が設定される FQDN が形成されます。上記の例では、W&B の FQDN は `wandb-gcp.wandb.ml` になります。

3. `variables.tf` ファイルを作成します。

   `terraform.tfvars` 内で設定された各オプションには、対応する変数宣言が必要です。

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

## デプロイメント - 推奨 (~20 分)

これは、すべての `Mandatory` コンポーネントを作成し、`Kubernetes Cluster` に最新の `W&B` をインストールする最も簡単なデプロイメントオプション設定です。

1. `main.tf` を作成します。

   [一般的な手順]({{< relref path="#general-steps" lang="ja" >}}) でファイルを作成したのと同じディレクトリーに、以下の内容の `main.tf` ファイルを作成します。

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

   # 必要なサービスをすべて開始する
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "~> 5.0"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain
   }

   # プロビジョニングされた IP アドレスで DNS を更新する必要があります
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

2. W&B をデプロイします。

   W&B をデプロイするために、次のコマンドを実行します:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを使用したデプロイメント

別のデプロイメントオプションでは、SQL クエリをキャッシュして実験のメトリクスを読み込む際のアプリケーションレスポンスを速めるために `Redis` を使用します。

キャッシュを有効にするには、おすすめの [デプロイメントオプションセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) に指定されているのと同じ `main.tf` ファイルにオプション `create_redis = true` を追加する必要があります。

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
  #Redis を有効化する
  create_redis = true

}
[...]
```

## 外部キューを使用したデプロイメント

デプロイメントオプション 3 は、外部の `メッセージブローカー` を有効にすることです。これはオプションであり、W&B は内部にブローカーを組み込んでいます。このオプションはパフォーマンスの改善をもたらしません。

メッセージブローカーを提供する GCP リソースは `Pub/Sub` であり、有効にするには、推奨される [デプロイメントオプションセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) に指定されたのと同じ `main.tf` にオプション `use_internal_queue = false` を追加する必要があります。

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
  #Pub/Sub を作成して使用する
  use_internal_queue = false

}

[...]

```

## その他のデプロイメントオプション

すべての構成を同じファイルに追加して、3 つのデプロイメントオプションを組み合わせることができます。
[Terraform Module](https://github.com/wandb/terraform-google-wandb)は、標準オプションと `Deployment - Recommended` で見つかる最小構成とともに組み合わせることができるいくつかのオプションを提供しています。

## 手動設定

W&B のファイルストレージバックエンドとして GCP Storage バケットを使用するには、次を作成する必要があります：

* [PubSub Topic と Subscription]({{< relref path="#create-pubsub-topic-and-subscription" lang="ja" >}})
* [Storage Bucket]({{< relref path="#create-storage-bucket" lang="ja" >}})
* [PubSub Notification]({{< relref path="#create-pubsub-notification" lang="ja" >}})

### PubSub Topic と Subscription を作成する

以下の手順に従って、PubSub トピックとサブスクリプションを作成してください：

1. GCP コンソール内で Pub/Sub サービスに移動します。
2. **Create Topic** を選択し、トピックの名前を指定します。
3. ページの下部で、**Create subscription** を選択します。**Delivery Type** が **Pull** に設定されていることを確認します。
4. **Create** をクリックします。

サービスアカウントまたはインスタンスが実行されているアカウントがこのサブスクリプションで `pubsub.admin` ロールを持っていることを確認してください。詳細は、https://cloud.google.com/pubsub/docs/access-control#console を参照してください。

### Storage Bucket を作成する

1. **Cloud Storage Buckets** ページに移動します。
2. **Create bucket** を選択し、バケットの名前を指定します。**Standard** ストレージクラス (https://cloud.google.com/storage/docs/storage-classes) を選択してください。

インスタンスが実行されているサービスアカウントまたはアカウントが次を持っていることを確認してください：
* 前のステップで作成したバケットへのアクセス
* このバケットでの `storage.objectAdmin` ロール。詳細は、https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照してください。

{{% alert %}}
また、インスタンスが署名付きファイル URL を作成するために GCP の `iam.serviceAccounts.signBlob` 権限を持つ必要があります。この権限を有効にするために、インスタンスが実行されているサービスアカウントまたは IAM メンバーに `Service Account Token Creator` ロールを追加してください。
{{% /alert %}}

3. CORS アクセスを有効にします。これはコマンドラインを使用してのみ行えます。まず、次の CORS 設定を含む JSON ファイルを作成します。

```
cors:
- maxAgeSeconds: 3600
  method:
   - GET
   - PUT
     origin:
   - '<YOUR-W&B-SERVER-HOST>'
     responseHeader:
   - Content-Type
```

オリジンの値の scheme、host、port は正確に一致する必要があることに注意してください。

4. `gcloud` がインストールされ、正しい GCP プロジェクトにログインしていることを確認します。
5. 次に、以下を実行します：

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub Notification を作成する

コマンドラインで次の手順に従って、Storage Bucket から Pub/Sub トピックへの通知ストリームを作成します。

{{% alert %}}
通知ストリームを作成するには CLI を使用する必要があります。`gcloud` がインストールされていることを確認してください。
{{% /alert %}}

1. GCP プロジェクトにログインします。
2. 次のコマンドをターミナルで実行します：

```bash
gcloud pubsub topics list  # トピックの名前を参照用に一覧表示
gcloud storage ls          # バケットの名前を参照用に一覧表示

# バケット通知を作成
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[Cloud Storage のウェブサイトにさらなる参考資料があります。](https://cloud.google.com/storage/docs/reporting-changes)

### W&B サーバーを設定する

1. 最後に、W&B の `System Connections` ページに移動し、`http(s)://YOUR-W&B-SERVER-HOST/console/settings/system` にアクセスします。
2. プロバイダーとして `Google Cloud Storage (gcs)` を選択します。
3. GCS バケットの名前を指定します。

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="" >}}

4. **Update settings** を押して、新しい設定を適用します。

## W&B Server のアップグレード

ここに示されている手順に従って W&B を更新します：

1. `wandb_app` モジュールの設定に `wandb_version` を追加します。アップグレードする W&B のバージョンを指定します。たとえば、次の行は W&B バージョン `0.48.1` を指定します：

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  または、`terraform.tfvars` に `wandb_version` を追加し、同じ名前の変数を作成し、文字列の代わりに `var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 設定を更新した後、[デプロイメントオプションセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) に記載されている手順を完了します。