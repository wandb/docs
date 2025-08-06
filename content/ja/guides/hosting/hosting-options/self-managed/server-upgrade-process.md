---
title: W&B ライセンスとバージョンの更新
description: さまざまなインストール方法で W&B のバージョンやライセンスを更新するためのガイド。
menu:
  default:
    identifier: server-upgrade-process
    parent: self-managed
url: guides/hosting/server-upgrade-process
weight: 6
---

W&B Server のバージョンおよびライセンスの更新は、インストール時と同じ方法で行います。以下の表は、各デプロイメント方法ごとのライセンスとバージョンの更新方法をまとめたものです。

| リリースタイプ    | 説明         |
| ---------------- | ------------------ |
| [Terraform]({{< relref "#update-with-terraform" >}}) | W&B ではクラウド デプロイメント用に 3 つの公開 Terraform モジュール（[AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)）をサポートしています。 |
| [Helm]({{< relref "#update-with-helm" >}})              | 既存の Kubernetes クラスターに W&B をインストールする際、[Helm Chart](https://github.com/wandb/helm-charts) を利用できます。  |

## Terraform を使ったアップデート

Terraform を使ってライセンスおよびバージョンを更新できます。以下の表は、クラウド プラットフォームごとの W&B 管理 Terraform モジュール一覧です。

|クラウドプロバイダー| Terraform モジュール|
|-----|-----|
|AWS|[AWS Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform モジュール](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. まず、ご利用のクラウドプロバイダーに対応した W&B 管理 Terraform モジュールのページに移動します。最適な Terraform モジュールは上記の表をご参照ください。
2. ご自身の Terraform 設定内で、`wandb_app` モジュールの `wandb_version` および `license` を更新します:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # 新しいライセンスキー
       wandb_version = "new_wandb_version" # 希望する W&B バージョン
       ...
   }
   ```
3. `terraform plan` および `terraform apply` で Terraform 設定を適用します。
   ```bash
   terraform init
   terraform apply
   ```

4. （任意）もし `terraform.tfvars` などの `.tfvars` ファイルを使用している場合:

   新しい W&B バージョンとライセンスキーを反映させて `terraform.tfvars` ファイルを更新または作成してください。
   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```
   適用します。Terraform のワークスペース ディレクトリーで以下を実行します:  
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```

## Helm を使ったアップデート

### spec で W&B を更新する

1. Helm chart の `*.yaml` 設定ファイル内で、`image.tag` および／または `license` の値を修正し新しいバージョンを指定します:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 下記コマンドで Helm アップグレードを実行します:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### ライセンスとバージョンを直接更新する

1. 新しいライセンスキーとイメージタグを環境変数に設定します:

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. 既存の設定に新しい値をマージして、以下のコマンドで Helm リリースをアップグレードします:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

詳細については、パブリックリポジトリ内の [アップグレードガイド](https://github.com/wandb/helm-charts/blob/main/upgrade.md) をご確認ください。

## 管理 UI を使ったアップデート

この方法は、W&B server コンテナで環境変数としてライセンスが設定されていない場合、主に自己管理型 Docker インストールでのみ利用可能です。

1. [W&B デプロイメントページ](https://deploy.wandb.ai/) から新しいライセンスを取得します。更新したいデプロイメント用の組織・デプロイメント ID に対応するライセンスを取得してください。
2. `<host-url>/system-settings` で W&B Admin UI に アクセスします。
3. ライセンス管理セクションへ移動します。
4. 新しいライセンスキーを入力し、変更を保存します。