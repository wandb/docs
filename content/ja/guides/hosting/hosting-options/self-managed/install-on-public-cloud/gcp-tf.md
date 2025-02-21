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
Weights & Biases では、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドの デプロイメント オプションをお勧めします。Weights & Biases のフルマネージドサービスは、シンプルで安全に使用でき、最小限の設定で済みます。
{{% /alert %}}

W&B サーバーの自己管理を行う場合、GCP に プラットフォーム を デプロイ するには、[W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) を使用することをお勧めします。

このモジュールのドキュメントは広範囲にわたり、利用可能なすべてのオプションが記載されています。

開始する前に、[State File](https://developer.hashicorp.com/terraform/language/state) を保存するために、Terraform で利用可能な [remote backends](https://developer.hashicorp.com/terraform/language/backend/remote) のいずれかを選択することを推奨します。

State File は、すべてのコンポーネントを再作成せずに、アップグレードを展開したり、 デプロイメント に変更を加えたりするために必要なリソースです。

Terraform Module は、次の「必須」コンポーネントを デプロイ します。

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

その他の デプロイメント オプションには、次のオプションコンポーネントを含めることもできます。

- Redis 用のメモリーストア
- Pub/Sub メッセージシステム

## 前提条件の権限

Terraform を実行するアカウントには、使用する GCP プロジェクト で `roles/owner` のロールが必要です。

## 一般的な手順

このトピックの手順は、このドキュメントで説明するすべての デプロイメント オプションに共通です。

1. 開発 環境 を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
   - 使用する コード で Git リポジトリを作成することをお勧めしますが、ファイルをローカルに保存することもできます。
   - [Google Cloud Console](https://console.cloud.google.com/) で プロジェクト を作成します。
   - GCP で認証します ([gcloud のインストール](https://cloud.google.com/sdk/docs/install) を確認してください)。
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

   ここで定義された変数は、 デプロイメント の前に決定する必要があります。`namespace` 変数は、Terraform によって作成されたすべてのリソースのプレフィックスとなる文字列です。

   `subdomain` と `domain` の組み合わせで、Weights & Biases が構成される FQDN が形成されます。上記の例では、Weights & Biases の FQDN は `wandb-gcp.wandb.ml` になります。

3. ファイル `variables.tf` を作成します。

   `terraform.tfvars` で構成されたすべてのオプションについて、Terraform は対応する変数宣言を必要とします。

   ```
   variable "project_id" {
     type        = string
     description = "プロジェクト ID"
   }

   variable "region" {
     type        = string
     description = "Google リージョン"
   }

   variable "zone" {
     type        = string
     description = "Google ゾーン"
   }

   variable "namespace" {
     type        = string
     description = "リソースに使用される名前空間プレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "Weights & Biases UI にアクセスするための ドメイン 名。"
   }

   variable "subdomain" {
     type        = string
     description = "Weights & Biases UI にアクセスするための サブドメイン。"
   }

   variable "license" {
     type        = string
     description = "W&B ライセンス"
   }
   ```

## デプロイメント - 推奨（約 20 分）

これは最も簡単な デプロイメント オプションの設定で、すべての「必須」コンポーネントを作成し、`Kubernetes Cluster` に最新バージョンの `W&B` をインストールします。

1. `main.tf` を作成します。

   [一般的な手順]({{< relref path="#general-steps" lang="ja" >}}) でファイルを作成したのと同じディレクトリーに、次の内容でファイル `main.tf` を作成します。

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

2. W&B を デプロイ します。

   W&B を デプロイ するには、次の コマンド を実行します。

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを使用した デプロイメント

別の デプロイメント オプションでは、`Redis` を使用して SQL クエリをキャッシュし、 実験 の メトリクス をロードする際の アプリケーション の応答を高速化します。

キャッシュを有効にするには、推奨される [ デプロイメント オプション のセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で指定されている同じ `main.tf` ファイルにオプション `create_redis = true` を追加する必要があります。

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

## 外部キューを使用した デプロイメント

デプロイメント オプション 3 は、外部 `message broker` を有効にすることです。W&B に broker が組み込まれているため、これはオプションです。このオプションは、パフォーマンスの向上をもたらしません。

message broker を提供する GCP リソースは `Pub/Sub` であり、これを有効にするには、推奨される [ デプロイメント オプション のセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で指定されている同じ `main.tf` にオプション `use_internal_queue = false` を追加する必要があります。

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

## その他の デプロイメント オプション

3 つの デプロイメント オプションをすべて組み合わせて、すべての構成を同じファイルに追加できます。
[Terraform Module](https://github.com/wandb/terraform-google-wandb) は、標準オプションおよび「 デプロイメント - 推奨」にある最小構成とともに組み合わせることができるいくつかのオプションを提供します。

## 手動構成

W&B の ファイル ストレージ バックエンドとして GCP Storage バケットを使用するには、以下を作成する必要があります。

* [PubSub トピックとサブスクリプション]({{< relref path="#create-pubsub-topic-and-subscription" lang="ja" >}})
* [ストレージバケット]({{< relref path="#create-storage-bucket" lang="ja" >}})
* [PubSub 通知]({{< relref path="#create-pubsub-notification" lang="ja" >}})

### PubSub トピックとサブスクリプションの作成

PubSub トピックとサブスクリプションを作成するには、次の手順に従います。

1. GCP Console 内の Pub/Sub サービスに移動します。
2. [**トピックを作成**] を選択し、トピックの名前を入力します。
3. ページの下部で、[**サブスクリプションを作成**] を選択します。[**配信タイプ**] が [**プル**] に設定されていることを確認します。
4. [**作成**] をクリックします。

インスタンスを実行している サービス アカウント またはアカウントに、このサブスクリプションの `pubsub.admin` ロールがあることを確認してください。詳細については、https://cloud.google.com/pubsub/docs/access-control#console を参照してください。

### ストレージ バケットの作成

1. [**Cloud Storage バケット**] ページに移動します。
2. [**バケットを作成**] を選択し、バケットの名前を入力します。[**標準**] [ストレージクラス](https://cloud.google.com/storage/docs/storage-classes) を選択していることを確認します。

インスタンスを実行している サービス アカウント またはアカウントに、次の両方があることを確認してください。
* 前のステップで作成したバケットへの アクセス
* このバケットの `storage.objectAdmin` ロール。詳細については、https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照してください。

{{% alert %}}
インスタンスが署名付き ファイル URL を作成するには、GCP で `iam.serviceAccounts.signBlob` 権限も必要です。`Service Account Token Creator` ロールを、インスタンスが実行されている サービス アカウント または IAM メンバーに追加して、権限を有効にします。
{{% /alert %}}

3. CORS アクセス を有効にします。これは コマンドライン でのみ実行できます。まず、次の CORS 構成で JSON ファイルを作成します。

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

オリジンのスキーム、ホスト、および ポート の 値 は正確に一致する必要があることに注意してください。

4. `gcloud` がインストールされており、正しい GCP プロジェクト にログインしていることを確認します。
5. 次に、以下を実行します。

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub 通知の作成
ストレージバケット から Pub/Sub トピック への通知ストリームを作成するには、 コマンドライン で次の手順に従います。

{{% alert %}}
通知ストリームを作成するには、CLI を使用する必要があります。`gcloud` がインストールされていることを確認してください。
{{% /alert %}}

1. GCP プロジェクト にログインします。
2. ターミナル で以下を実行します。

```bash
gcloud pubsub topics list  # list names of topics for reference
gcloud storage ls          # list names of buckets for reference

# create bucket notification
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[詳細については、Cloud Storage の Web サイトを参照してください。](https://cloud.google.com/storage/docs/reporting-changes)

### W&B サーバーの構成

1. 最後に、`http(s)://YOUR-W&B-SERVER-HOST/console/settings/system` の W&B の [システム接続] ページに移動します。
2. プロバイダー `Google Cloud Storage (gcs)` を選択します。
3. GCS バケットの名前を入力します。

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="" >}}

4. [**設定を更新**] をクリックして、新しい 設定 を適用します。

## W&B サーバーのアップグレード

W&B を更新するには、ここで概説する手順に従います。

1. `wandb_app` モジュールの構成に `wandb_version` を追加します。アップグレードする W&B の バージョン を指定します。たとえば、次の行は W&B バージョン `0.48.1` を指定します。

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  または、`wandb_version` を `terraform.tfvars` に追加し、同じ名前の変数を作成し、リテラル 値 を使用する代わりに、`var.wandb_version` を使用することもできます。
  {{% /alert %}}

2. 構成を更新したら、[ デプロイメント オプション のセクション]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) で説明されている手順を完了します。
