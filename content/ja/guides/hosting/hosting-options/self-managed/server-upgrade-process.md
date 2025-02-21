---
title: Update W&B license and version
description: W&B (Weights & Biases) のバージョンとライセンスを異なるインストールメソッドで更新するためのガイド。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-server-upgrade-process
    parent: self-managed
url: guides/hosting/server-upgrade-process
weight: 6
---

W&B Server のバージョンとライセンスを、W&B サーバーをインストールしたのと同じ方法で更新することができます。以下の表は、さまざまなデプロイメント メソッドに基づくライセンスとバージョンの更新方法を示しています。

| リリースタイプ | 説明 |
| ----------- | ---- |
| [Terraform]({{< relref path="#update-with-terraform" lang="ja" >}}) | W&B はクラウドデプロイメントのために[AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest)、[GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest)、および[Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)の3つのパブリックTerraformモジュールをサポートしています。 |
| [Helm]({{< relref path="#update-with-helm" lang="ja" >}}) | [Helm Chart](https://github.com/wandb/helm-charts)を使用して、既存のKubernetesクラスターにW&Bをインストールすることができます。 |

## Terraformで更新する

Terraformを使用してライセンスとバージョンを更新します。以下の表は、クラウドプラットフォームに基づくW&B 管理のTerraformモジュールを示しています。

| クラウド プロバイダー | Terraform モジュール |
| -------------- | ------------------- |
| AWS      | [AWS Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/aws/latest) |
| GCP      | [GCP Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/google/latest) |
| Azure    | [Azure Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) |

1. 最初に、適切なクラウドプロバイダーに対応するW&Bが管理しているTerraformモジュールに移動します。対応するTerraformモジュールを見つけるには、前の表を参照してください。
2. Terraform 設定内で、`wandb_version` および `license` を Terraform `wandb_app` モジュールの設定で更新します。

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # 新しいライセンスキー
       wandb_version = "new_wandb_version" # 希望する W&B バージョン
       ...
   }
   ```

3. `terraform plan` と `terraform apply` を使用して、Terraform 設定を適用します。

   ```bash
   terraform init
   terraform apply
   ```

4. (オプション) `terraform.tfvars` またはその他の `.tfvars` ファイルを使用している場合。

   新しいW&Bバージョンとライセンスキーで `terraform.tfvars` ファイルを更新または作成します。

   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```

   設定を適用します。Terraformワークスペースディレクトリーで以下を実行してください:  
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```

## Helm で更新する

### specでW&Bを更新

1. Helmチャート `*.yaml` の設定ファイル内で `image.tag` および/または `license` 値を変更して新しいバージョンを指定します。

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 以下のコマンドで、Helm アップグレードを実行します。

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### ライセンスとバージョンを直接更新

1. 環境変数として新しいライセンスキーとイメージタグを設定します。

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. 以下のコマンドで、既存の設定に新しい値をマージしてHelmリリースをアップグレードします。

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

詳細については、パブリックリポジトリの[アップグレードガイド](https://github.com/wandb/helm-charts/blob/main/upgrade.md)を参照してください。

## 管理者UIで更新する

このメソッドは、自己ホスト型 Docker インストールで通常、W&B サーバー コンテナ内の環境変数で設定されていないライセンスを更新する場合にのみ機能します。

1. [W&Bデプロイメントページ](https://deploy.wandb.ai/)から新しいライセンスを取得し、アップグレードしようとしているデプロイメントの正しい組織とデプロイメントIDと一致することを確認します。
2. `<host-url>/system-settings` でW&B管理者UIにアクセスします。
3. ライセンス管理セクションに移動します。
4. 新しいライセンスキーを入力し、変更を保存します。