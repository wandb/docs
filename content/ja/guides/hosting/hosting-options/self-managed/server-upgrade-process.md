---
title: Update W&B license and version
description: さまざまなインストール メソッド にわたって、W&B (Weights & Biases) の バージョン とライセンスを更新するための
  ガイド 。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-server-upgrade-process
    parent: self-managed
url: guides/hosting/server-upgrade-process
weight: 6
---

W&B サーバー の バージョンとライセンスは、 W&B サーバー のインストール時と同じ 方法で更新します。次の表は、異なる デプロイメント 方法に基づいてライセンスと バージョン を更新する方法をまとめたものです。

| リリースタイプ | 説明 |
| --- | --- |
| [Terraform]({{< relref path="#update-with-terraform" lang="ja" >}}) | W&B は、 クラウド デプロイメント 用に3つのパブリック Terraform モジュールをサポートしています: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), and [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest). |
| [Helm]({{< relref path="#update-with-helm" lang="ja" >}}) | [Helm Chart](https://github.com/wandb/helm-charts) を使用して、既存の Kubernetes クラスター に W&B をインストールできます。 |

## Terraform で更新する

Terraform でライセンスと バージョン を更新します。以下の表は、 W&B が管理する クラウド プラットフォーム に基づいた Terraform モジュールの一覧です。

|クラウド プロバイダー| Terraform モジュール |
|-----|-----|
|AWS|[AWS Terraform module](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform module](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. まず、適切な クラウド プロバイダー 用に W&B がメンテナンスしている Terraform モジュールに移動します。クラウド プロバイダー に基づいて適切な Terraform モジュールを見つけるには、前の表を参照してください。
2. Terraform の 設定 内で、Terraform `wandb_app` モジュール 設定 の `wandb_version` と `license` を更新します:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # Your new license key
       wandb_version = "new_wandb_version" # Desired W&B version
       ...
   }
   ```
3. `terraform plan` と `terraform apply` で Terraform の 設定 を適用します。
   ```bash
   terraform init
   terraform apply
   ```

4. (オプション) `terraform.tfvars` または他の `.tfvars` ファイルを使用する場合。

   新しい W&B の バージョン とライセンス キー で `terraform.tfvars` ファイルを更新または作成します。
   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```
   設定 を適用します。Terraform ワークスペース ディレクトリー で以下を実行します:
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```
## Helm で更新する

### spec で W&B を更新する

1. Helm chart の `*.yaml` 設定 ファイルで `image.tag` または `license` の 値 を変更して、新しい バージョン を指定します:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 次の コマンド で Helm の アップグレード を実行します:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### ライセンスと バージョン を直接更新する

1. 新しいライセンス キー とイメージ タグ を 環境 変数として設定します:

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. 以下の コマンド で Helm リリース をアップグレードし、新しい 値 を既存の 設定 とマージします:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

詳細については、パブリック リポジトリー の [upgrade guide](https://github.com/wandb/helm-charts/blob/main/upgrade.md) を参照してください。

## 管理 UI で更新する

この 方法 は、通常セルフホストされた Docker インストールで、 W&B サーバー コンテナー の 環境 変数で設定されていないライセンスを更新する場合にのみ機能します。

1. [W&B Deployment Page](https://deploy.wandb.ai/) から新しいライセンスを取得し、アップグレードする デプロイメント の正しい Organization および デプロイメント ID と一致していることを確認します。
2. `<host-url>/system-settings` で W&B Admin UI にアクセスします。
3. ライセンス管理セクションに移動します。
4. 新しいライセンス キー を入力し、変更を保存します。
