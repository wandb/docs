---
description: W&B（Weights & Biases）のバージョンとライセンスを異なるインストールメソッドで更新するためのガイド。
displayed_sidebar: default
---


# W&B のライセンスとバージョンの更新

インストール方法に従って W&B サーバーのバージョンとライセンスを更新します。以下の表は、さまざまなデプロイメントメソッドに基づくライセンスとバージョンの更新方法を示しています。

| リリースタイプ      | 説明         |
| ---------------- | ------------------ |
| [Terraform](#update-with-terraform) | W&B はクラウドデプロイメントのための3つの公開Terraformモジュールをサポートしています: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), and [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest). |
| [Helm](#update-with-helm)              | 既存のKubernetesクラスターにW&Bをインストールするために、[Helm Chart](https://github.com/wandb/helm-charts) を使用できます。  |
| [Docker](#update-with-docker-container)     | 最新のDockerイメージは [W&B Docker Registry](https://hub.docker.com/r/wandb/local/tags) にあります。 |

## Terraform での更新

Terraform を使ってライセンスとバージョンを更新します。以下の表には、クラウドプラットフォームに基づく W&B 管理の Terraform モジュールが示されています。

|クラウドプロバイダー| Terraform モジュール|
|-----|-----|
|AWS|[AWS Terraform module](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform module](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. まず、適切なクラウドプロバイダーに対応する W&B 管理の Terraform モジュールに移動します。適切な Terraform モジュールを見つけるために、前述の表を参照してください。
2. Terraform 設定内で、 `wandb_version` と `license` を Terraform `wandb_app` モジュール設定で更新します:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # 新しいライセンスキー
       wandb_version = "new_wandb_version" # 望む W&B のバージョン
       ...
   }
   ```
3. `terraform plan` と `terraform apply` を使って Terraform 設定を適用します。
   ```bash
   terraform init
   terraform apply
   ```

4. (オプション) `terraform.tfvars` またはその他の `.tfvars` ファイルを使用する場合:
   1. 新しい W&B バージョンとライセンスキーで `terraform.tfvars` ファイルを更新または作成します。
   2. 設定を適用します。Terraform ワークスペースディレクトリーで以下を実行します:  
   ```bash
   terraform plan -var-file="terraform.tfvars"
   terraform apply -var-file="terraform.tfvars"
   ```

## Helm での更新

### スペックを使ったW&Bの更新

1. Helm チャート `*.yaml` 設定ファイルで `image.tag` と/または `license` 値を変更して新しいバージョンを指定します:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 以下のコマンドを使って Helm アップグレードを実行します:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### ライセンスとバージョンの直接更新

1. 新しいライセンスキーとイメージタグを環境変数として設定します:

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. 以下のコマンドを使って Helm リリースをアップグレードし、新しい値を既存の設定とマージします:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

詳細については、公開リポジトリにある [upgrade guide](https://github.com/wandb/helm-charts/blob/main/UPGRADE.md) を参照してください。

## Docker コンテナでの更新

1. [W&B Docker Registry](https://hub.docker.com/r/wandb/local/tags) から新しいバージョンを選択します。
2. 以下のコマンドで新しい Docker イメージバージョンをプルします:

   ```bash
   docker pull wandb/local:<new_version>
   ```

3. コンテナのデプロイメントと管理のベストプラクティスに従って、新しいイメージバージョンを実行するように Docker コンテナを更新します。

## 管理者 UI を使った更新

この方法は、通常自己ホスト型の Docker インストールで、 環境変数で設定されていないライセンスの更新にのみ使用されます。

1. [W&B Deployment Page](https://deploy.wandb.ai/)から新しいライセンスを取得し、アップグレードするデプロイメントに対応する正しい組織とデプロイメントIDが一致していることを確認します。
2. `<host-url>/system-settings` にアクセスして W&B 管理者 UI にアクセスします。
3. ライセンス管理セクションに移動します。
4. 新しいライセンスキーを入力し、変更を保存します。