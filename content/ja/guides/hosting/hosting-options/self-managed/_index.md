---
title: セルフマネージド
description: プロダクション環境で W&B をデプロイする
cascade:
- url: guides/hosting/self-managed/:filename
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
url: guides/hosting/hosting-options/self-managed
---

## W&B セルフマネージドをクラウドまたはオンプレミスインフラストラクチャーで利用する

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) など、完全管理型のデプロイメントオプションを推奨しています。W&B のフルマネージドサービスは設定がほとんど、または全く必要なく、シンプルかつセキュアにご利用いただけます。
{{% /alert %}}

[お使いの AWS、GCP、または Azure クラウドアカウント]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) や [オンプレミスインフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) に W&B Server をデプロイできます。

お客様の IT/DevOps/MLOps チームに求められる役割:
- デプロイメントのプロビジョニング
- 組織のポリシーや [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) に従ったインフラストラクチャーのセキュリティ確保（該当する場合）
- アップグレードやパッチ適用の管理
- セルフマネージド W&B Server インスタンスの継続的なメンテナンス


## セルフマネージドクラウドアカウント内で W&B Server をデプロイする

W&B では、公式の W&B Terraform スクリプトを利用して、AWS、GCP、または Azure のクラウドアカウントに W&B Server をデプロイすることを推奨しています。

W&B Server のセットアップについては、各クラウドプロバイダーのドキュメントをご参照ください：[AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}})、[GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ja" >}})、[Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ja" >}})。

## オンプレミスインフラストラクチャーで W&B Server をデプロイする

オンプレミスインフラストラクチャーに W&B Server をセットアップするには、いくつかのインフラストラクチャーコンポーネントの設定が必要です。主な構成要素の例としては、以下が挙げられます（これらに限りません）:

- （強く推奨）Kubernetes クラスター
- MySQL 8 データベースクラスター
- Amazon S3 互換のオブジェクトストレージ
- Redis キャッシュクラスター

オンプレミスインフラストラクチャーでの W&B Server インストールの詳細手順については、[Install on on-prem infrastructure]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ja" >}}) をご参照ください。W&B では各コンポーネントに関する推奨事項やインストールプロセス全体のガイダンスもご提供可能です。

## カスタムクラウドプラットフォーム上で W&B Server をデプロイする

AWS、GCP、Azure 以外のクラウドプラットフォームにも W&B Server をデプロイできます。その場合の要件は、[オンプレミスインフラストラクチャーでのデプロイ]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) とほぼ同様です。

## W&B Server ライセンスの取得

W&B server の設定を完了するには、W&B トライアルライセンスが必要です。無料トライアルライセンスの発行には [Deploy Manager](https://deploy.wandb.ai/deploy) をご利用ください。

{{% alert %}}
まだ W&B アカウントをお持ちでない場合は、アカウントを作成して無料ライセンスを取得してください。

W&B Server で高度なセキュリティ機能などが必要なエンタープライズライセンスをご希望の場合は、[こちらのフォーム](https://wandb.ai/site/for-enterprise/self-hosted-trial) の送信、もしくはお使いの W&B チームまでお問い合わせください。
{{% /alert %}}

指定の URL から **Get a License for W&B Local** フォームへリダイレクトされます。以下の情報をご入力ください：

1. **Choose Platform** ステップでデプロイメントタイプを選択
2. **Basic Information** ステップでライセンスのオーナーを選択、または新しい組織を追加
3. **Get a License** ステップの **Name of Instance** 欄にインスタンス名を入力し、必要に応じて **Description** 欄に説明を追加
4. **Generate License Key** ボタンを選択

デプロイメント概要ページにインスタンスに紐づくライセンスが表示されます。