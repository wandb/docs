---
description: Hosting W&B Server on GCP.
displayed_sidebar: default
---

# GCP

Weights and Biasesが開発した[Terraformモジュール](https://registry.terraform.io/modules/wandb/wandb/google/latest)を使用して、W&BサーバーをGoogle Cloudにデプロイすることをお勧めします。

モジュールのドキュメントは非常に充実しており、使用可能なすべてのオプションが含まれています。このドキュメントでは、いくつかの展開オプションについて説明します。

始める前に、Terraformの[リモートバックエンド](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)のいずれかを選択し、[ステートファイル](https://developer.hashicorp.com/terraform/language/state)を保存することをお勧めします。

ステートファイルは、すべてのコンポーネントを再作成することなく、展開にアップグレードを適用したり変更を加えるために必要なリソースです。

Terraformモジュールは、以下の`必須`コンポーネントをデプロイします：

- VPC
- Cloud SQL for MySQL
- Cloud Storageバケット
- Google Kubernetes Engine
- KMS暗号キー
- ロードバランサ

他の展開オプションには、以下のオプションコンポーネントが含まれる場合があります：

- Redis用のメモリストア
- Pub/Subメッセージシステム

## **前提条件とする権限**

Terraformを実行するアカウントは、使用するGCPプロジェクトに`roles/owner`の役割を持っている必要があります。

## 一般的な手順

このドキュメントでカバーされているどの展開オプションにも共通の手順があります。

1. 開発環境を準備する。
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)をインストールする
   - 使用されるコードを含むGitリポジトリを作成することをお勧めしますが、ファイルをローカルに保持することもできます。
   - [Google Cloud Console](https://console.cloud.google.com/)でプロジェクトを作成する
   - GCPで認証する（[gcloudをインストール](https://cloud.google.com/sdk/docs/install)してください）
     `gcloud auth application-default login`
2. `terraform.tfvars`ファイルを作成する。

`tvfars`ファイルの内容は、インストールタイプに応じてカスタマイズできますが、最小限のお勧めは以下の例のようになります。

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   ここで定義されている変数は、展開前に決定する必要があります。`namespace`変数は、Terraformによって作成されるすべてのリソースの接頭辞となる文字列になります。

   `subdomain`と`domain`の組み合わせによって、W&Bが設定されるFQDNが形成されます。上記の例では、W&B FQDNは`wandb-gcp.wandb.ml`になります。

3. `versions.tf`ファイルを作成します。

   このファイルには、GCPでW&Bを展開するために必要なTerraformおよびTerraformプロバイダーのバージョンが含まれます。

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

   任意ですが、**強くお勧めします**、このドキュメントの冒頭で触れている[リモートバックエンド設定](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)を追加することができます。

4. `variables.tf`ファイルを作成します。

   `terraform.tfvars`で設定されたすべてのオプションに対応する変数宣言がTerraformに必要です。

   ```
   variable "project_id" {
     type        = string
     description = "プロジェクトID"
   }

   variable "region" {
     type        = string
     description = "Googleリージョン"
   }

   variable "zone" {
     type        = string
     description = "Googleゾーン"
   }

variable "namespace" {
     type        = string
     description = "リソースに使用される名前空間プレフィックス"
   }

   variable "domain_name" {
     type        = string
     description = "Weights & Biases UIにアクセスするためのドメイン名。"
   }

   variable "subdomain" {
     type        = string
     description = "Weights & Biases UIにアクセスするためのサブドメイン。"
   }

   variable "license" {
     type        = string
     description = "W&B ライセンス"
   }
   ```

## 展開 - 推奨（約20分）

これは最も簡単な展開オプション構成で、すべての `Mandatory` コンポーネントを作成し、`Kubernetes Cluster` に `W&B` の最新バージョンをインストールします。

1. `main.tf` を作成する

   `General Steps` で作成したファイルと同じディレクトリに、次の内容の `main.tf` ファイルを作成します。

   ```
   data "google_client_config" "current" {}

provider "kubernetes" {
     host                   = "https://${module.wandb.cluster_endpoint}"
     cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
     token                  = data.google_client_config.current.access_token
   }

   # すべての必要なサービスを起動する
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "1.12.2"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain

   }

   # 用意されたIPアドレスにDNSを更新する必要があります
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

2. W&Bのデプロイ

   W&Bをデプロイするには、以下のコマンドを実行してください:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDISの有効化

別の展開オプションでは、`Redis` を使用して SQL クエリをキャッシュし、実験のメトリクスを読み込む際のアプリケーションの応答速度を向上させます。

キャッシュを有効にするには、`Deployment option 1` で作業したのと同じ `main.tf` ファイルに `create_redis = true` オプションを追加する必要があります。

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "1.12.2"

  namespace    = var.namespace
  license      = var.license
  domain_name  = var.domain_name
  subdomain    = var.subdomain
  #Enable Redis
  create_redis = true

}
[...]
```

## メッセージブローカー(キュー)の有効化

展開オプション3は、外部の `メッセージブローカー` を有効にすることです。W&B にはブローカーが埋め込まれているため、これはオプションです。このオプションは、パフォーマンスの向上をもたらしません。

メッセージブローカーを提供する GCP リソースは `Pub/Sub` であり、それを有効にするには、`Deployment option 1` で作業したのと同じ `main.tf` に `use_internal_queue = false` オプションを追加する必要があります。

```
[...]

以下は翻訳するMarkdownテキストのチャンクです。それ以外のものは追加せず、翻訳されたテキストのみを返してください。テキスト：
 module "wandb" {
  source  = "wandb/wandb/google"
  version = "1.12.2"

  namespace          = var.namespace
  license            = var.license
  domain_name        = var.domain_name
  subdomain          = var.subdomain
  #Create and use Pub/Sub
  use_internal_queue = false

}

[...]

```

## 他の展開オプション

すべての構成を同じファイルに追加して、3つの展開オプションを組み合わせることができます。
[Terraform Module](https://github.com/wandb/terraform-google-wandb)は、`Deployment - Recommended`で見つかる標準的なオプションと最小限の構成とともに組み合わせることができるいくつかのオプションを提供しています。

<!-- ## Upgrades (coming soon) -->

## 手動設定

GCPストレージバケットをW&Bのファイルストレージバックエンドとして使用するには、次のものを作成する必要があります。

* [PubSubトピックとサブスクリプション](#create-pubsub-topic-and-subscription)
* [ストレージバケット](#create-storage-bucket)
* [PubSub通知](#create-pubsub-notification)

### PubSubトピックとサブスクリプションの作成

以下の手順に従って、PubSubトピックとサブスクリプションを作成してください。

1. GCP Console 内の Pub/Sub サービスに移動
2. **トピックの作成**を選択し、トピックの名前を入力してください。
3. ページの下部で、**サブスクリプションの作成**を選択します。**配信タイプ**が**Pull**に設定されていることを確認してください。
4. **作成**をクリックします。

インスタンスが実行しているサービスアカウントまたはアカウントに、このサブスクリプションで `pubsub.admin` ロールが付与されていることを確認してください。詳細については、https://cloud.google.com/pubsub/docs/access-control#console を参照してください。

### ストレージバケットの作成

1. **Cloud Storage Buckets**ページに移動します。
2. **バケットの作成**を選択し、バケットの名前を入力してください。**Standard**の[ストレージクラス](https://cloud.google.com/storage/docs/storage-classes)を選択してください。

インスタンスが実行されているサービスアカウントまたはアカウントが、以下の両方を持っていることを確認してください:
* 前のステップで作成したバケットへのアクセス
* このバケットの`storage.objectAdmin`ロール。詳細については、https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add を参照してください。

:::info
インスタンスは、GCPで署名されたファイルのURLを作成するために、`iam.serviceAccounts.signBlob`権限も必要です。インスタンスが実行されているサービスアカウントまたはIAMメンバーに`Service Account Token Creator`ロールを追加して、権限を有効にしてください。
:::

3. CORSアクセスを有効にします。これは、コマンドラインを使用してのみ行うことができます。まず、以下のCORS構成を持つJSONファイルを作成します。

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

スキーム、ホスト、およびオリジンの値のポートは、正確に一致している必要があります。

4. `gcloud`がインストールされていることを確認し、適切なGCPプロジェクトにログインしてください。
5. 次に、以下を実行します。

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub通知の作成
以下の手順をコマンドラインで実行し、ストレージバケットからPub/Subトピックへの通知ストリームを作成します。

:::info
通知ストリームを作成するには、CLIを使用する必要があります。`gcloud`がインストールされていることを確認してください。
:::

1. GCPプロジェクトにログインします。
2. ターミナルで以下を実行します。

```bash
gcloud pubsub topics list  # トピック名のリストを参照する
gcloud storage ls          # バケット名のリストを参照する

# バケット通知を作成する
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[Cloud Storageのウェブサイトでさらに参考資料が利用可能です。](https://cloud.google.com/storage/docs/reporting-changes)

### W&Bサーバーの設定

1. 最後に、W&B設定ページに移動してください。`http(s)://YOUR-W&B-SERVER-HOST/system-admin`。
2. "外部ファイルストレージバックエンドを使用する"オプションを有効にします。
3. AWS S3バケットの名前、バケットが保存されているリージョン、およびSQSキューを以下の形式で指定します。
* **File Storage Bucket**：`gs://<bucket-name>`
* **File Storage Region**：空白
* **Notification Subscription**：`pubsub:/<project-name>/<topic-name>/<subscription-name>`

![](/images/hosting/configure_file_store.png)

4. **設定を更新**を押して新しい設定を適用します。