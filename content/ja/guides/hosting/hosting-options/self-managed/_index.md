---
title: Self-managed
description: W&B をプロダクション環境にデプロイする
cascade:
- url: guides/hosting/self-managed/:filename
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
url: guides/hosting/hosting-options/self-managed
---

## セルフマネージドクラウドまたはオンプレミスインフラストラクチャーの使用

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドなデプロイメントオプションを推奨します。W&B のフルマネージドサービスは、シンプルで安全に使用でき、設定は最小限で済みます。
{{% /alert %}}

[AWS、GCP、または Azure クラウドアカウント]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) または [オンプレミスインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) に W&B Server をデプロイします。

お客様の IT/DevOps/MLOps チームは、お客様のデプロイメントのプロビジョニング、アップグレードの管理、およびセルフマネージドな W&B Server インスタンスの継続的なメンテナンスを担当します。

## セルフマネージドクラウドアカウント内への W&B Server のデプロイ

W&B は、W&B Server を AWS、GCP、または Azure クラウドアカウントにデプロイするために、公式の W&B Terraform スクリプトを使用することを推奨します。

[AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}}), [GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ja" >}}) または [Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ja" >}}) での W&B Server のセットアップ方法の詳細については、特定のクラウドプロバイダーのドキュメントを参照してください。

## オンプレミスインフラストラクチャーへの W&B Server のデプロイ

オンプレミスインフラストラクチャーに W&B Server をセットアップするには、いくつかのインフラストラクチャーコンポーネントを設定する必要があります。これらのコンポーネントには、以下が含まれますが、これらに限定されません。

- (強く推奨) Kubernetes cluster
- MySQL 8 database cluster
- Amazon S3 互換 object storage
- Redis cache cluster

オンプレミスインフラストラクチャーへの W&B Server のインストール方法の詳細については、[オンプレミスインフラストラクチャーへのインストール]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ja" >}}) を参照してください。W&B は、さまざまなコンポーネントに関する推奨事項を提供し、インストールプロセスを通じてガイダンスを提供できます。

## カスタムクラウドプラットフォームへの W&B Server のデプロイ

AWS、GCP、または Azure ではないクラウドプラットフォームに W&B Server をデプロイできます。そのための要件は、[オンプレミスインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) にデプロイする場合と同様です。

## W&B Server のライセンスの取得

W&B サーバーの設定を完了するには、W&B trial ライセンスが必要です。[Deploy Manager](https://deploy.wandb.ai/deploy) を開いて、無料の trial ライセンスを生成してください。

{{% alert %}}
まだ W&B アカウントをお持ちでない場合は、アカウントを作成して無料ライセンスを生成してください。

重要なセキュリティやその他のエンタープライズフレンドリーな機能のサポートを含む W&B Server のエンタープライズライセンスが必要な場合は、[このフォームを送信](https://wandb.ai/site/for-enterprise/self-hosted-trial) するか、W&B チームにお問い合わせください。
{{% /alert %}}

URL をクリックすると、**Get a License for W&B Local** フォームにリダイレクトされます。次の情報を提供してください。

1. **Choose Platform** ステップで、デプロイメントタイプを選択します。
2. **Basic Information** ステップで、ライセンスの所有者を選択するか、新しい組織を追加します。
3. **Get a License** ステップの **Name of Instance** フィールドにインスタンスの名前を入力し、必要に応じて **Description** フィールドに説明を入力します。
4. **Generate License Key** ボタンを選択します。

ページに、デプロイメントの概要と、インスタンスに関連付けられたライセンスが表示されます。
