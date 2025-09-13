---
title: W&B のライセンスとバージョンの更新
description: さまざまなインストール方法で W&B のバージョンやライセンスを更新するためのガイド。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-server-upgrade-process
    parent: self-managed
url: guides/hosting/server-upgrade-process
weight: 6
---

W&B サーバーの バージョン と ライセンス は、W&B サーバーをインストールしたのと同じ方法でアップデートします。以下の表は、異なる デプロイメント 方法に基づく ライセンス と バージョン のアップデート方法を示します。
| リリースタイプ | 説明 |
| ---------------- | ------------------ |
| [Terraform]({{< relref path="#update-with-terraform" lang="ja" >}}) | W&B は、クラウド デプロイメント向けに 3 つのパブリック Terraform モジュールをサポートしています。[AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest)、[GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest)、[Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)。 |
| [Helm]({{< relref path="#update-with-helm" lang="ja" >}}) | 既存の Kubernetes クラスターに W&B をインストールするために [Helm Chart](https://github.com/wandb/helm-charts) を使用できます。 |
## Terraform を使用したアップデート
Terraform を使用して、ライセンス と バージョン をアップデートします。以下の表に、クラウド プラットフォーム別の W&B 管理 Terraform モジュールを示します。
|クラウドプロバイダー| Terraform モジュール|
|-----|-----|
|AWS|[AWS Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|
1. まず、利用している クラウド プロバイダー 向けの W&B 管理 Terraform モジュールに移動します。該当する Terraform モジュールは上記の表を参照してください。
2. Terraform の 設定 内で、Terraform の `wandb_app` モジュール 設定の `wandb_version` と `license` をアップデートします。
   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # 新しいライセンスキー
       wandb_version = "new_wandb_version" # 希望する W&B バージョン
       ...
   }
   ```
3. `terraform plan` と `terraform apply` を使って Terraform の 設定 を適用します。
   ```bash
   terraform init
   terraform apply
   ```
4. (オプション) `terraform.tfvars` などの `.tfvars` ファイルを使用する場合:
   新しい W&B バージョン と ライセンスキー を含む `terraform.tfvars` ファイルを作成または更新します。
   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```
   設定 を適用します。Terraform ワークスペース ディレクトリで以下を実行します。
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```
## Helm を使用したアップデート
### spec を使用した W&B のアップデート
1. Helm チャートの `*.yaml` 設定ファイルで、`image.tag` および/または `license` の 値 を変更して、新しい バージョン を指定します。
   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```
2. 次の コマンド を使用して Helm のアップグレードを実行します。
   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```
### ライセンス と バージョン を直接アップデート
1. 新しい ライセンスキー と イメージタグ を 環境変数 として設定します。
   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```
2. 以下の コマンド を使用して Helm リリース をアップグレードし、新しい 値 を既存の 設定 とマージします。
   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```
詳細は、パブリック リポジトリの [アップグレード ガイド](https://github.com/wandb/helm-charts/blob/main/upgrade.md) を参照してください。
## 管理 UI を使用したアップデート
この方法は、W&B サーバー コンテナー内で 環境変数 として設定されていない ライセンス（通常は自己管理型 Docker インストール）のアップデートにのみ有効です。
1. [W&B Deployment Page](https://deploy.wandb.ai/) から新しい ライセンス を取得し、アップグレード対象の デプロイメント の適切な組織および デプロイメント ID と一致していることを確認します。
2. `<host-url>/system-settings` で W&B 管理 UI にアクセスします。
3. ライセンス 管理 セクション に移動します。
4. 新しい ライセンスキー を入力して、変更を保存します。