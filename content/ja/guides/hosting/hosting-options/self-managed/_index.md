---
title: Self-managed
description: プロダクションでの W&B のデプロイメント
cascade:
- url: guides/hosting/self-managed/:filename
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
url: guides/hosting/hosting-options/self-managed
---

## セルフマネージドのクラウドまたはオンプレミスインフラストラクチャーを使用する

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) デプロイメントタイプのような完全マネージドのデプロイメントオプションを推奨しています。W&Bの完全マネージドサービスは、簡単かつ安全に使用でき、設定がほとんど必要ありません。
{{% /alert %}}

W&B Server を [AWS、GCP、または Azure クラウドアカウント]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) または [オンプレミスのインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) にデプロイします。

IT/DevOps/MLOps チームは、デプロイメントの用意、アップグレードの管理、およびセルフマネージド W&B Server インスタンスの継続的な保守を担当します。

## セルフマネージドクラウドアカウント内で W&B Server をデプロイする

W&B は official W&B Terraform scripts を使用して、AWS、GCP、または Azure クラウドアカウントに W&B Server をデプロイすることをお勧めします。

W&B Server を [AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}})、 [GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ja" >}})、または [Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ja" >}}) に設定する方法の詳細については、特定のクラウドプロバイダーのドキュメントを参照してください。

## オンプレミスインフラストラクチャーで W&B Server をデプロイする

オンプレミスインフラストラクチャーに W&B Server を設定するには、いくつかのインフラストラクチャーコンポーネントを設定する必要があります。これらのコンポーネントには、以下のものが含まれますが、これらに限定されません：

- （強く推奨）Kubernetes クラスター
- MySQL 8 データベースクラスター
- Amazon S3-互換のオブジェクトストレージ
- Redis キャッシュクラスター

オンプレミスインフラストラクチャーへのインストール方法については、[Install on on-prem infrastructure]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ja" >}}) を参照してください。W&Bは、さまざまなコンポーネントに対する推奨事項を提供し、インストールプロセスを通じてガイダンスを提供できます。

## カスタムクラウドプラットフォームで W&B Server をデプロイする

AWS、GCP、または Azure 以外のクラウドプラットフォームに W&B Server をデプロイすることができます。そのための要件は、[オンプレミスインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}})にデプロイする場合と似ています。

## W&B Server ライセンスを取得する

W&B server の設定を完了するには W&B トライアルライセンスが必要です。[Deploy Manager](https://deploy.wandb.ai/deploy)を開いて、無料のトライアルライセンスを生成してください。

{{% alert %}}
まだ W&B アカウントをお持ちでない場合は、無料のライセンスを生成するためにアカウントを作成してください。

重要なセキュリティや他のエンタープライズ向け機能を含む W&B Server のエンタープライズライセンスが必要な場合は、[submit this form](https://wandb.ai/site/for-enterprise/self-hosted-trial) または W&B チームにお問い合わせください。
{{% /alert %}}

URLはあなたを **Get a License for W&B Local** フォームにリダイレクトします。以下の情報を提供してください：

1. **Choose Platform** ステップからデプロイメントタイプを選択します。
2. **Basic Information** ステップでライセンスの所有者を選択するか、新しい組織を追加します。
3. **Get a License** ステップの **Name of Instance** フィールドでインスタンスの名前を提供し、オプションで **Description** フィールドに説明を追加します。
4. **Generate License Key** ボタンを選択します。

インスタンスに関連付けられたライセンスと共に、デプロイメントの概要を示すページが表示されます。