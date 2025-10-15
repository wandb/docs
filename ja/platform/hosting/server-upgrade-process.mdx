---
title: W&B ライセンスと バージョン を更新する
description: W&B (Weights & Biases) のバージョンとライセンスを異なるインストールメソッドで更新するためのガイド。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-server-upgrade-process
    parent: self-managed
url: /ja/guides/hosting/server-upgrade-process
weight: 6
---

W&B Server の バージョン と ライセンスのアップデートは、W&B Server のインストール方法と同じ方法で行います。次の表に、さまざまな デプロイメント メソッド に基づいたライセンスとバージョンのアップデート方法を示します。

| Release Type    | Description         |
| ---------------- | ------------------ |
| [Terraform]({{< relref path="#update-with-terraform" lang="ja" >}}) | W&B は、クラウド デプロイメント用に 3 つのパブリック Terraform モジュールをサポートしています: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), および [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)。 |
| [Helm]({{< relref path="#update-with-helm" lang="ja" >}})              | 既存の Kubernetes クラスターに W&B をインストールするために [Helm Chart](https://github.com/wandb/helm-charts) を使用できます。  |

## Terraform を使用してアップデート

Terraform を使ってライセンスと バージョン を更新します。以下の表に、クラウド プラットフォーム に基づく W&B 管理Terraform モジュールを示します。

|Cloud provider| Terraform module|
|-----|-----|
|AWS|[AWS Terraform module](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform module](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. まず、お使いの クラウド プロバイダー 用の W&B 管理の Terraform モジュールに移動します。前の表を参照して、クラウド プロバイダー に基づいた適切な Terraform モジュールを見つけてください。
2. Terraform 設定内で、Terraform `wandb_app` モジュールの設定で `wandb_version` と `license` を更新します:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # 新しいライセンス キー
       wandb_version = "new_wandb_version" # 希望する W&B バージョン
       ...
   }
   ```
3. `terraform plan` および `terraform apply` コマンドで Terraform 設定を適用します。
   ```bash
   terraform init
   terraform apply
   ```

4. (オプション) `terraform.tfvars` またはその他の `.tfvars` ファイルを使用する場合。

   新しい W&B バージョン と ライセンス キー を指定して `terraform.tfvars` ファイルを更新または作成します。
   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```
   設定を適用します。Terraform ワークスペース ディレクトリー で以下を実行します:  
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```
## Helm を使用してアップデート

### Spec を使って W&B をアップデート

1. Helm チャート `*.yaml` 設定 ファイルで `image.tag` および/または `license` の 値 を変更して新しい バージョン を指定します:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 以下の コマンド で Helm アップグレード を実行します:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### ライセンスと バージョン を直接アップデート

1. 新しい ライセンス キー と イメージ タグ を 環境 変数として設定します:

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. 以下の コマンド で Helm リリース をアップグレードし、新しい 値 を既存の設定とマージします:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

詳細については、パブリック リポジトリの[アップグレード ガイド](https://github.com/wandb/helm-charts/blob/main/upgrade.md)を参照してください。

## 管理者 UI を使用してアップデート

この方法は、通常、自己ホスト型 Docker インストール で、環境 変数 を使用して W&B サーバー コンテナ内に設定されていないライセンスを更新する場合にのみ使用されます。

1. [W&B デプロイメント ページ](https://deploy.wandb.ai/) から新しいライセンスを取得し、アップグレードしようとしている デプロイメント に対して正しい組織およびデプロイメント ID と一致することを確認します。
2. W&B 管理者 UI に `<host-url>/system-settings` でアクセスします。
3. ライセンス管理セクションに移動します。
4. 新しいライセンスキーを入力し、変更を保存します。