---
title: 自己管理
description: W&B をプロダクション環境に展開する
cascade:
- url: /ja/guides/hosting/self-managed/:filename
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
url: /ja/guides/hosting/hosting-options/self-managed
---

## セルフ管理クラウドまたはオンプレインフラストラクチャーを使用

{{% alert %}}
W&B は、[W&B Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) のようなフルマネージドデプロイメントオプションを推奨します。W&B のフルマネージドサービスは、簡単で安全に使用でき、ほとんど設定が不要です。
{{% /alert %}}

[あなたの AWS, GCP, または Azure クラウドアカウント]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) または [オンプレミスのインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) に W&B Server をデプロイしてください。

あなたの IT/DevOps/MLOps チームが、デプロイメントのプロビジョニング、アップグレードの管理、セルフマネージド W&B Server インスタンスの継続的な保守を担当します。

## セルフ管理クラウドアカウントに W&B Server をデプロイする

W&B は、公式の W&B Terraform スクリプトを使用して、AWS、GCP、または Azure のクラウドアカウントに W&B Server をデプロイすることを推奨します。

[AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}})、[GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ja" >}}) または [Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ja" >}}) に W&B Server を設定する方法については、各クラウドプロバイダーのドキュメントを参照してください。

## オンプレインフラストラクチャーに W&B Server をデプロイする

W&B Server をオンプレインフラストラクチャーに設定するには、いくつかのインフラストラクチャーコンポーネントを設定する必要があります。これらのコンポーネントには、以下が含まれますが、それに限定されません：

- (強く推奨) Kubernetes クラスター
- MySQL 8 データベースクラスター
- Amazon S3 互換オブジェクトストレージ
- Redis キャッシュクラスター

オンプレインフラストラクチャーに W&B Server をインストールする方法については、[Install on on-prem infrastructure]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ja" >}}) を参照してください。W&B はさまざまなコンポーネントに関する推奨事項を提供し、インストールプロセスをガイドします。

## カスタムクラウドプラットフォームに W&B Server をデプロイする

AWS、GCP、または Azure ではないクラウドプラットフォームに W&B Server をデプロイすることができます。これには、[オンプレインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) にデプロイする場合と同様の要件があります。

## W&B Server ライセンスの取得

W&B server の設定を完了するには、W&B のトライアルライセンスが必要です。 [Deploy Manager](https://deploy.wandb.ai/deploy) を開いて、無料のトライアルライセンスを生成してください。

{{% alert %}}
まだ W&B アカウントをお持ちでない場合は、無料ライセンスを生成するためにアカウントを作成してください。

重要なセキュリティ & 企業向け機能のサポートを含む、W&B Server のエンタープライズライセンスが必要な場合は、[このフォームを送信](https://wandb.ai/site/for-enterprise/self-hosted-trial)するか、W&B チームに連絡してください。
{{% /alert %}}

URL は **Get a License for W&B Local** フォームにリダイレクトします。次の情報を提供してください:

1. **Choose Platform** ステップでデプロイタイプを選択します。
2. **Basic Information** ステップでライセンスの所有者を選択するか、新しい組織を追加します。
3. **Get a License** ステップでインスタンスの名前を入力し、任意で **Description** フィールドに説明を提供します。
4. **Generate License Key** ボタンを選択します。

インスタンスに関連付けられたライセンスとともにデプロイメントの概要を示すページが表示されます。