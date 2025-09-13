---
title: 自己管理型
description: W&B を本番環境にデプロイする
cascade:
- url: guides/hosting/self-managed/:filename
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
url: guides/hosting/hosting-options/self-managed
---

## W&B Self-Managed をクラウドまたはオンプレミスインフラストラクチャーで使用する

{{% alert %}}
W&B は、[W&B マルチテナント クラウド]({{< relref path="../saas_cloud.md" lang="ja" >}}) または [W&B 専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) といったフルマネージドのデプロイメントオプションを推奨しています。W&B のフルマネージドサービスは、最小限の設定で簡単かつ安全に使用できます。
{{% /alert %}}

W&B Server を [AWS、GCP、または Azure のクラウドアカウント]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) または [オンプレミスインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) にデプロイします。

IT/DevOps/MLOps チームは、以下の責任を負います。
- デプロイメントのプロビジョニング。
- 組織のポリシーおよび該当する場合は [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) に従ってインフラストラクチャーを保護すること。
- アップグレードの管理とパッチの適用。
- セルフマネージドの W&B Server インスタンスを継続的に維持すること。

## セルフマネージドクラウドアカウント内に W&B Server をデプロイする

W&B は、公式の W&B Terraform スクリプトを使用して、W&B Server を AWS、GCP、または Azure のクラウドアカウントにデプロイすることを推奨しています。

W&B Server を [AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}})、[GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ja" >}})、または [Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ja" >}}) にセットアップする方法の詳細については、各クラウドプロバイダーのドキュメントを参照してください。

## オンプレミスインフラストラクチャーに W&B Server をデプロイする

オンプレミスインフラストラクチャーに W&B Server をセットアップするには、いくつかのインフラストラクチャーコンポーネントを設定する必要があります。これらのコンポーネントには、以下が含まれますが、これに限りません。

- (強く推奨) Kubernetes クラスター
- MySQL 8 データベースクラスター
- Amazon S3 互換のオブジェクトストレージ
- Redis キャッシュクラスター

オンプレミスインフラストラクチャーに W&B Server をインストールする詳細な手順については、[オンプレミスインフラストラクチャーへのインストール]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ja" >}}) を参照してください。W&B は、各コンポーネントに関する推奨事項を提供し、インストールプロセス全体を通じてガイダンスを提供できます。

## カスタムクラウドプラットフォームに W&B Server をデプロイする

AWS、GCP、または Azure 以外のクラウドプラットフォームに W&B Server をデプロイすることができます。そのための要件は、[オンプレミスインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) にデプロイする場合と同様です。

## W&B Server ライセンスの取得

W&B Server の設定を完了するには、W&B のトライアルライセンスが必要です。[Deploy Manager](https://deploy.wandb.ai/deploy) を開いて、無料のトライアルライセンスを生成してください。

{{% alert %}}
まだ W&B アカウントをお持ちでない場合は、無料ライセンスを生成するためにアカウントを作成してください。

重要なセキュリティおよびその他の企業向け機能のサポートを含む W&B Server のエンタープライズライセンスが必要な場合は、[このフォームを送信](https://wandb.ai/site/for-enterprise/self-hosted-trial) するか、W&B チームにお問い合わせください。
{{% /alert %}}

URL は、**W&B Local のライセンスを取得** フォームにリダイレクトされます。以下の情報を提供してください。

1. **プラットフォームを選択** の手順でデプロイメントタイプを選択します。
2. **基本情報** の手順でライセンスの所有者を選択するか、新しい組織を追加します。
3. **ライセンスの取得** の手順で、**インスタンス名** フィールドにインスタンスの名前を入力し、必要に応じて **説明** フィールドに説明を入力します。
4. **ライセンスキーの生成** ボタンを選択します。

デプロイメントの概要と、インスタンスに関連付けられたライセンスが表示されます。