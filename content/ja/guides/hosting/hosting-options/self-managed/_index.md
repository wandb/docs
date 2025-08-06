---
title: セルフマネージド
description: プロダクション環境での W&B のデプロイ
menu:
  default:
    identifier: self-managed
    parent: deployment-options
url: guides/hosting/hosting-options/self-managed
cascade:
- url: guides/hosting/self-managed/:filename
---

## クラウドやオンプレミスインフラストラクチャーでの W&B Self-Managed 利用

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref "../saas_cloud.md" >}}) や [W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) など、フルマネージドのデプロイメントオプションを推奨しています。W&B のフルマネージドサービスはシンプルかつ安全に利用でき、設定も最小限もしくは不要です。
{{% /alert %}}

[自分の AWS, GCP, Azure クラウドアカウント]({{< relref "#deploy-wb-server-within-self-managed-cloud-accounts" >}}) や、[オンプレミスインフラストラクチャー]({{< relref "#deploy-wb-server-in-on-prem-infrastructure" >}}) 内に W&B Server をデプロイできます。

IT／DevOps／MLOps チームが担当する内容:
- デプロイメントのプロビジョニング
- 組織のポリシーや[Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide)（該当する場合）に沿ったインフラストラクチャーのセキュリティ確保
- アップグレードの管理やパッチの適用
- Self-Managed W&B Server インスタンスの継続的な保守





## 自己管理クラウドアカウント内での W&B Server デプロイ

AWS, GCP, Azure のクラウドアカウントに W&B Server をデプロイする際は、公式の W&B Terraform スクリプトの利用をおすすめします。

W&B Server のセットアップについて、各クラウドプロバイダーごとの詳細は次のドキュメントをご覧ください。[AWS]({{< relref "/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" >}})、[GCP]({{< relref "/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" >}})、[Azure]({{< relref "/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" >}})。

## オンプレミスインフラストラクチャーでの W&B Server デプロイ

オンプレミスインフラストラクチャーに W&B Server をセットアップするには、いくつかのインフラストラクチャーコンポーネントの設定が必要です。主なコンポーネント例は以下の通りですが、これらに限定されません。

- （推奨）Kubernetes クラスター
- MySQL 8 データベースクラスター
- Amazon S3 互換オブジェクトストレージ
- Redis キャッシュクラスター

W&B Server をオンプレミスインフラストラクチャーにインストールする詳細な手順は、[Install on on-prem infrastructure]({{< relref "/guides/hosting/hosting-options/self-managed/bare-metal.md" >}}) をご覧ください。W&B では各コンポーネントの推奨構成やインストールプロセスに関するガイダンスも提供しています。

## カスタムクラウドプラットフォームへの W&B Server デプロイ

W&B Server は、AWS、GCP、Azure 以外のクラウドプラットフォームにもデプロイ可能です。その場合の要件は、[オンプレミスインフラストラクチャー]({{< relref "#deploy-wb-server-in-on-prem-infrastructure" >}}) でのデプロイと同様です。

## W&B Server ライセンスの取得

W&B Server の設定を完了するには、W&B のトライアルライセンスが必要です。無料トライアルライセンスを発行するには、[Deploy Manager](https://deploy.wandb.ai/deploy) を開いてください。

{{% alert %}}
まだ W&B アカウントをお持ちでない場合は、ライセンス発行のためにアカウントを作成してください。

エンタープライズ向けのセキュリティ対応や各種高機能を含む W&B Server のライセンスが必要な場合は、[こちらのフォーム](https://wandb.ai/site/for-enterprise/self-hosted-trial)よりお申し込みいただくか、W&B チームまでご連絡ください。
{{% /alert %}}

URL から **Get a License for W&B Local** のフォームにリダイレクトされます。以下の情報を記入してください。

1. **Choose Platform** ステップでデプロイメントタイプを選択
2. **Basic Information** ステップでライセンスの所有者を選ぶか新たに組織を追加
3. **Get a License** ステップの **Name of Instance** 項目にインスタンス名を入力し、必要に応じて **Description** 項目に説明を記載
4. **Generate License Key** ボタンを選択

インスタンスに紐付いたライセンスやデプロイメントの概要が表示されるページが開きます。