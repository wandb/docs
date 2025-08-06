---
title: W&B ライセンスとバージョンの更新
description: さまざまなインストール メソッドで W&B のバージョンとライセンスを更新するためのガイド。
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-server-upgrade-process
    parent: self-managed
url: guides/hosting/server-upgrade-process
weight: 6
---

W&B Server のバージョンやライセンスの更新は、インストール時と同じ手法で行ってください。以下の表は、異なるデプロイメント方法ごとにライセンスとバージョンを更新する方法をまとめたものです。

| リリースタイプ    | 説明         |
| ---------------- | ------------------ |
| [Terraform]({{< relref path="#update-with-terraform" lang="ja" >}}) | W&B はクラウドデプロイメント向けに [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest)、[GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest)、[Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) の3つの公開 Terraform モジュールをサポートしています。 |
| [Helm]({{< relref path="#update-with-helm" lang="ja" >}})              | Kubernetes クラスター上に W&B をインストールするための [Helm Chart](https://github.com/wandb/helm-charts) を利用できます。  |

## Terraform での更新

Terraform を利用してライセンスやバージョンを更新します。下記の表は各クラウドプラットフォーム向けの W&B 管理 Terraform モジュール一覧です。

|クラウドプロバイダー| Terraform モジュール|
|-----|-----|
|AWS|[AWS Terraform module](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform module](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. まず、あなたのクラウドプロバイダーに合った W&B 管理 Terraform モジュールのページにアクセスします。該当モジュールは上記の表でご確認ください。
2. Terraform の設定内で、`wandb_app` モジュールの設定にある `wandb_version` と `license` を新しい値に更新します。

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # 新しいライセンスキー
       wandb_version = "new_wandb_version" # 利用したい W&B バージョン
       ...
   }
   ```

3. `terraform plan` および `terraform apply` を実行して Terraform の設定を反映します。
   ```bash
   terraform init
   terraform apply
   ```

4. （オプション）`terraform.tfvars` や他の `.tfvars` ファイルを使う場合

   新しい W&B バージョンとライセンスキーで `terraform.tfvars` ファイルを更新または作成します。
   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```
   設定を適用するには、Terraform の作業ディレクトリーで次のコマンドを実行します。  
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```

## Helm での更新

### spec を使った W&B の更新

1. Helm チャートの `*.yaml` 設定ファイルで `image.tag` および/または `license` の値を編集して新しいバージョンやライセンスを指定します。

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 次のコマンドで Helm アップグレードを実行します。

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### ライセンスとバージョンを直接指定して更新

1. 新しいライセンスキーとイメージタグを環境変数として設定します。

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. 既存の設定と新しい値をマージして Helm リリースをアップグレードします。

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

詳細は、公開リポジトリ内の [アップグレードガイド](https://github.com/wandb/helm-charts/blob/main/upgrade.md) をご覧ください。

## admin UI での更新

この方法は、W&B サーバーのコンテナで環境変数としてライセンスが設定されていない場合（通常は自己管理型 Docker インストール）にのみ有効です。

1. [W&B Deployment Page](https://deploy.wandb.ai/) から新しいライセンスを取得し、アップグレードしたいデプロイメントの組織およびデプロイメント ID と一致していることを確認します。
2. `<host-url>/system-settings` で W&B の管理 UI にアクセスします。
3. ライセンス管理セクションへ移動します。
4. 新しいライセンスキーを入力し、変更を保存します。