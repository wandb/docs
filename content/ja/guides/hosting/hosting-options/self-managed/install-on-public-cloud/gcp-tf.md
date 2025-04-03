---
title: Deploy W&B Platform on GCP
description: GCP 上での W&B サーバー のホスティング。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-gcp-tf
    parent: install-on-public-cloud
weight: 20
---

{{% alert %}}
Weights & Biases では、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドなデプロイメントオプションをお勧めします。Weights & Biases のフルマネージドサービスは、シンプルで安全に使用でき、設定は最小限で済みます。
{{% /alert %}}

W&B Server の自己管理を行う場合、GCP 上にプラットフォームをデプロイするために、[W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) を使用することを推奨します。

モジュールのドキュメントは広範囲にわたり、利用可能なすべてのオプションが記載されています。

開始する前に、Terraform の [リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend/remote) のいずれかを選択して、[State File](https://developer.hashicorp.com/terraform/language/state) を保存することを推奨します。

State File は、すべてのコンポーネントを再作成せずに、アップグレードを展開したり、デプロイメントに変更を加えたりするために必要なリソースです。

Terraform Module は、以下の必須コンポーネントをデプロイします。

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

その他のデプロイメントオプションには、以下のオプションコンポーネントを含めることもできます。

- Redis 用のメモリーストア
- Pub/Sub メッセージシステム

## 事前requisite 権限

Terraform を実行するアカウントには、使用する GCP プロジェクトで `roles/owner` のロールが必要です。

## 一般的な手順

このトピックの手順は、このドキュメントで説明するすべてのデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
   - 使用するコードで Git リポジトリを作成することをお勧めしますが、ファイルをローカルに保存することもできます。
   - [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成します。
   - GCP で認証します ([gcloud](https://cloud.google.com/sdk/docs/install) をインストールしてください)。
     `gcloud auth application-default login`
2. `terraform.tfvars` ファイルを作成します。

   `tvfars` ファイルの内容は、インストールタイプに応じてカスタマイズできますが、最小限の推奨設定は以下の例のようになります。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義する変数は、デプロイメントの前に決定する必要があります。`namespace` 変数は、Terraform によって作成されるすべてのリソースに接頭辞を付ける文字列になります。

   `subdomain` と `domain` の組み合わせで、Weights & Biases が設定される FQDN が形成されます。上記の例では、Weights & Biases の FQDN は `wandb-gcp.wandb.ml` になります。

3. ファイル `variables.tf` を作成します。

   `terraform.tfvars` で設定されたすべてのオプションについて、Terraform は対応する変数宣言を必要とします。

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
     description = "Weights & Biases UI にアクセスするためのドメイン名"
   }

   variable "subdomain" {
     type        = string
     description = "Weights & Biases UI にアクセスするためのサブドメイン"
   }

   variable "license" {
     type        = string
     description = "W&B License"
   }
   ```

## デプロイメント - 推奨 (~20 分)

これは最も簡単なデプロイメントオプションの設定で、すべての `Mandatory` コンポーネントを作成し、`Kubernetes Cluster` に最新バージョンの `W&B` をインストールします。

1. `main.tf` を作成します。

   [一般的な手順]({{< relref path="#general-steps" lang="ja" >}}) でファイルを作成したのと同じディレクトリーに、次の内容で `main.tf` ファイルを作成します。

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

   # Spin up all required services
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "~> 5.0"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain
   }

   # You'll want to update your DNS with the provisioned IP address
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

   W&B をデプロイするには、次のコマンドを実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを使用したデプロイメント

別のデプロイメントオプションでは、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスをロードする際のアプリケーションの応答を高速化します。

キャッシュを有効にするには、推奨される [デプロイメントオプションのセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で指定されている同じ `main.tf` ファイルにオプション `create_redis = true` を追加する必要があります。

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
  #Enable Redis
  create_redis = true

}
[...]
```

## 外部キューを使用したデプロイメント

デプロイメントオプション 3 は、外部 `message broker` を有効にすることです。W&B にはブローカーが組み込まれているため、これはオプションです。このオプションは、パフォーマンスの向上をもたらしません。

メッセージブローカーを提供する GCP リソースは `Pub/Sub` であり、これを有効にするには、推奨される [デプロイメントオプションのセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で指定されている同じ `main.tf` にオプション `use_internal_queue = false` を追加する必要があります。

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
  #Create and use Pub/Sub
  use_internal_queue = false

}

[...]

```

## その他のデプロイメントオプション

3 つのデプロイメントオプションすべてを組み合わせて、すべての構成を同じファイルに追加できます。
[Terraform Module](https://github.com/wandb/terraform-google-wandb) は、標準オプションと `Deployment - Recommended` にある最小限の構成とともに組み合わせることができるいくつかのオプションを提供します。

## 手動構成

GCP Storage バケットを W&B のファイルストレージバックエンドとして使用するには、以下を作成する必要があります。

* [PubSub トピックとサブスクリプション]({{< relref path="#create-pubsub-topic-and-subscription" lang="ja" >}})
* [ストレージバケット]({{< relref path="#create-storage-bucket" lang="ja" >}})
* [PubSub 通知]({{< relref path="#create-pubsub-notification" lang="ja" >}})

### PubSub トピックとサブスクリプションを作成する

PubSub トピックとサブスクリプションを作成するには、以下の手順に従ってください。

1. GCP Console 内の Pub/Sub サービスに移動します。
2. [**トピックを作成**] を選択し、トピックの名前を指定します。
3. ページの下部で、[**サブスクリプションを作成**] を選択します。[**配信タイプ**] が [**プル**] に設定されていることを確認します。
4. [**作成**] をクリックします。

インスタンスを実行しているサービスアカウントまたはアカウントに、このサブスクリプションに対する `pubsub.admin` ロールがあることを確認してください。詳細については、https://cloud.google.com/pubsub/docs/access-control#console を参照してください。

### ストレージバケットを作成する

1. [**Cloud Storage バケット**] ページに移動します。
2. [**バケットを作成**] を選択し、バケットの名前を指定します。[**標準**] [ストレージクラス](https://cloud.google.com/storage/docs/storage-classes) を選択してください。

インスタンスを実行しているサービスアカウントまたはアカウントに、以下の両方があることを確認してください。
* 前の手順で作成したバケットへのアクセス
* このバケットに対する `storage.objectAdmin` ロール。詳細については、https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照してください。

{{% alert %}}
インスタンスが署名付きファイル URL を作成するには、GCP で `iam.serviceAccounts.signBlob` 権限も必要です。インスタンスを実行しているサービスアカウントまたは IAM メンバーに `Service Account Token Creator` ロールを追加して、権限を有効にします。
{{% /alert %}}

3. CORS アクセスを有効にします。これはコマンドラインでのみ実行できます。まず、次の CORS 構成で JSON ファイルを作成します。

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

オリジンのスキーム、ホスト、ポートの値が正確に一致する必要があることに注意してください。

4. `gcloud` がインストールされ、正しい GCP プロジェクトにログインしていることを確認します。
5. 次に、以下を実行します。

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub 通知を作成する
ストレージバケットから Pub/Sub トピックへの通知ストリームを作成するには、コマンドラインで以下の手順に従ってください。

{{% alert %}}
通知ストリームを作成するには、CLI を使用する必要があります。`gcloud` がインストールされていることを確認してください。
{{% /alert %}}

1. GCP プロジェクトにログインします。
2. ターミナルで以下を実行します。

```bash
gcloud pubsub topics list  # list names of topics for reference
gcloud storage ls          # list names of buckets for reference

# create bucket notification
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[詳細については、Cloud Storage の Web サイトをご覧ください。](https://cloud.google.com/storage/docs/reporting-changes)

### W&B サーバーを設定する

1. 最後に、`http(s)://YOUR-W&B-SERVER-HOST/console/settings/system` で W&B の [システム接続] ページに移動します。
2. プロバイダー `Google Cloud Storage (gcs)` を選択します。
3. GCS バケットの名前を指定します。

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="" >}}

4. [**設定を更新**] を押して、新しい設定を適用します。

## W&B サーバーをアップグレードする

W&B を更新するには、ここで概説する手順に従ってください。

1. `wandb_app` モジュールの構成に `wandb_version` を追加します。アップグレードする W&B のバージョンを指定します。たとえば、次の行は W&B バージョン `0.48.1` を指定します。

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  または、`wandb_version` を `terraform.tfvars` に追加し、同じ名前の変数を作成して、リテラル値を使用する代わりに `var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 構成を更新したら、[デプロイメントオプションのセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で説明されている手順を完了します。
