---
title: W&B プラットフォームを GCP にデプロイする
description: GCP で W&B サーバー をホスティングする。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-install-on-public-cloud-gcp-tf
    parent: install-on-public-cloud
weight: 20
---

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) または [W&B 専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のデプロイメントタイプなどの完全管理されたデプロイメントオプションを推奨しています。W&B の完全管理サービスはシンプルで安全に使用でき、設定はほとんど必要ありません。
{{% /alert %}}

W&B Server のセルフマネージドを選択した場合、W&B は [W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) を使用して GCP 上にプラットフォームをデプロイすることを推奨しています。

このモジュールのドキュメントは詳細で、使用可能なすべてのオプションが含まれています。

始める前に、Terraform 用の [リモートバックエンド](https://developer.hashicorp.com/terraform/language/backend/remote) のいずれかを選択し、[ステートファイル](https://developer.hashicorp.com/terraform/language/state) を保存することをお勧めします。

ステートファイルは、コンポーネントを再作成することなく、アップグレードを展開したり、デプロイメントに変更を加えたりするために必要なリソースです。

Terraform モジュールは以下の `必須` コンポーネントをデプロイします：

- VPC
- Cloud SQL for MySQL
- Cloud Storage バケット
- Google Kubernetes Engine
- KMS 暗号キー
- ロードバランサ

他のデプロイメントオプションには次のオプションコンポーネントが含まれることがあります：

- Redis のためのメモリストア
- Pub/Sub メッセージシステム

## 事前要件の許可

Terraform を実行するアカウントには、使用される GCP プロジェクトにおいて `roles/owner` の役割が必要です。

## 一般的な手順

このトピックの手順は、このドキュメントでカバーされている任意のデプロイメントオプションに共通です。

1. 開発環境を準備します。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) をインストールします。
   - 使用するコードを含む Git リポジトリを作成することをお勧めしますが、ファイルをローカルに保持することもできます。
   - [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成します。
   - GCP に認証します（`gcloud` を[インストール](https://cloud.google.com/sdk/docs/install) しておくことを確認してください）
     `gcloud auth application-default login`
2. `terraform.tfvars` ファイルを作成します。

   `tvfars` ファイルの内容はインストールタイプに応じてカスタマイズできますが、最低限の推奨事項は以下の例のようになります。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義する変数はデプロイメントの前に決定する必要があります。`namespace` 変数は、Terraform によって作成されたすべてのリソースにプレフィックスとして付ける文字列になります。

   `subdomain` と `domain` の組み合わせが W&B が設定される FQDN を形成します。上記の例では、W&B FQDN は `wandb-gcp.wandb.ml` となります。

3. `variables.tf` ファイルを作成します。

   `terraform.tfvars` で設定されたすべてのオプションに対して、Terraform は対応する変数宣言を求めます。

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

これは `Mandatory` コンポーネントをすべて作成し、`Kubernetes Cluster` に `W&B` の最新バージョンをインストールする最も単純なデプロイメントオプション設定です。

1. `main.tf` を作成します。

   [一般的な手順]({{< relref path="#general-steps" lang="ja" >}}) でファイルを作成したのと同じディレクトリに、次の内容の `main.tf` ファイルを作成します：

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

   # 必須サービスをすべて起動
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

   W&B をデプロイするには、次のコマンドを実行します：

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS キャッシュを使用したデプロイメント

別のデプロイメントオプションでは、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスをロードする際のアプリケーションの応答速度を向上させます。

推奨される [Deployment option section]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) に示されている同じ `main.tf` ファイルに `create_redis = true` のオプションを追加する必要があります。

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

## 外部キューを使用したデプロイメント

デプロイメントオプション 3 は外部の `メッセージブローカー` を有効化することから成ります。これは W&B が組み込みのブローカーを提供しているため、オプションです。性能改善はもたらしません。

メッセージブローカーを提供する GCP リソースは `Pub/Sub` であり、これを有効にするには、推奨される [Deployment option section]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) に示されている同じ `main.tf` に `use_internal_queue = false` のオプションを追加する必要があります。

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
  # Pub/Sub を作成して使用
  use_internal_queue = false

}

[...]

```

## その他のデプロイメントオプション

すべてのデプロイメントオプションを組み合わせて、すべての設定を同じファイルに追加することができます。 [Terraform Module](https://github.com/wandb/terraform-google-wandb) は、標準のオプションや `Deployment - Recommended` で見つかる最小限の設定と共に組み合わせることができる複数のオプションを提供しています。

## 手動設定

GCP ストレージバケットを W&B のファイルストレージバックエンドとして使用するには、以下を作成する必要があります：

* [PubSub Topic と Subscription]({{< relref path="#create-pubsub-topic-and-subscription" lang="ja" >}})
* [ストレージバケット]({{< relref path="#create-storage-bucket" lang="ja" >}})
* [PubSub 通知]({{< relref path="#create-pubsub-notification" lang="ja" >}})

### PubSub Topic と Subscription の作成

以下の手順に従って、PubSub トピックとサブスクリプションを作成します：

1. GCP Console 内の Pub/Sub サービスに移動します。
2. **Create Topic** を選択してトピックに名前を付けます。
3. ページの下部で、**Create subscription** を選択します。 **Delivery Type** が **Pull** に設定されていることを確認します。
4. **Create** をクリックします。

サービスアカウントまたはインスタンスが実行中のアカウントが、このサブスクリプションの `pubsub.admin` ロールを持っていることを確認します。 詳細については、https://cloud.google.com/pubsub/docs/access-control#console を参照してください。

### ストレージバケットの作成

1. **Cloud Storage バケット** ページに移動します。
2. **Create bucket** を選択してバケットに名前を付けます。 **Standard** の [ストレージクラス](https://cloud.google.com/storage/docs/storage-classes) を選択していることを確認します。

インスタンスが実行中のサービスアカウントまたはアカウントが、以下の条件をすべて満たしていることを確認してください：
* 前のステップで作成したバケットへのアクセス
* このバケットに対する `storage.objectAdmin` ロール。 詳細については、https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照してください。

{{% alert %}}
インスタンスは署名付きファイル URL を作成するために GCP で `iam.serviceAccounts.signBlob` の権限も必要です。 サービスアカウントまたはインスタンスが実行する IAM メンバーに `サービスアカウントトークンクリエーター` のロールを追加して、権限を有効にします。
{{% /alert %}}

3. CORS アクセスを有効化します。これはコマンドラインのみで実行できます。まず、以下の CORS 設定を含む JSON ファイルを作成します。

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

ここでの origin の値のスキーム、ホスト、およびポートが正確に一致していることを確認してください。

4. `gcloud` が正しくインストールされ、適切な GCP プロジェクトにログインしていることを確認してください。
5. 次に、以下を実行します：

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub 通知の作成
コマンドラインで以下の手順に従って、ストレージバケットから Pub/Sub トピックへの通知ストリームを作成します。

{{% alert %}}
通知ストリームを作成するには CLI を使用する必要があります。`gcloud` がインストールされていることを確認してください。
{{% /alert %}}

1. GCP プロジェクトにログインします。
2. ターミナルで次の操作を実行します：

```bash
gcloud pubsub topics list  # トピックの名前を参照用にリスト表示
gcloud storage ls          # バケットの名前を参照用にリスト表示

# バケット通知を作成
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[Cloud Storage のウェブサイトにさらに参考資料があります。](https://cloud.google.com/storage/docs/reporting-changes)

### W&B サーバーの設定

1. 最後に、W&B の `System Connections` ページに http(s)://YOUR-W&B-SERVER-HOST/console/settings/system を開きます。
2. プロバイダーとして `Google Cloud Storage (gcs)` を選択します。
3. GCS バケットの名前を提供します。

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="" >}}

4. **設定を更新** を押して、新しい設定を適用します。

## W&B サーバーのアップグレード

ここに示された手順に従って W&B を更新します：

1. あなたの `wandb_app` モジュールに `wandb_version` を追加します。アップグレードしたい W&B のバージョンを指定します。例えば、以下の行は W&B バージョン `0.48.1` を指定します：

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  代わりに、`wandb_version` を `terraform.tfvars` に追加し、同じ名前の変数を作成して、リテラル値の代わりに `var.wandb_version` を使用することができます。
  {{% /alert %}}

2. 設定を更新した後、[Deployment option section]({{< relref path="#deployment---recommended-20-mins" lang="ja" >}}) に記載されている手順を完了します。