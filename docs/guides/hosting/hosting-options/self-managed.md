---
title: Self managed
description: プロダクションに W&B をデプロイする
displayed_sidebar: default
---

# Self managed

:::info
W&B は [W&B Multi-tenant Cloud](./saas_cloud.md) または [W&B Dedicated Cloud](./dedicated_cloud.md) デプロイメントタイプなど、完全に管理されたデプロイメントオプションを推奨しています。W&B の完全管理サービスは、設定がほとんど不要で、シンプルかつ安全に利用できます。
:::

[自分の AWS、GCP、または Azure クラウドアカウント](#deploy-wb-server-within-self-managed-cloud-accounts) または [オンプレミスのインフラストラクチャー](#deploy-wb-server-in-on-premises-infrastructure) に W&B Server をデプロイします。

IT/DevOps/MLOps チームは、デプロイメントのプロビジョニング、アップグレードの管理、および自己管理型の W&B Server インスタンスの継続的なメンテナンスを担当します。

## 自己管理型クラウドアカウントに W&B Server をデプロイ

W&B では、公式の W&B Terraform スクリプトを使用して、AWS、GCP、または Azure クラウドアカウントに W&B Server をデプロイすることを推奨しています。

[AWS](../self-managed/aws-tf.md)、[GCP](../self-managed/gcp-tf.md)、または [Azure](../self-managed/azure-tf.md) に W&B Server を設定する方法の詳細については、各クラウドプロバイダーのドキュメントを参照してください。

## オンプレミスのインフラストラクチャーに W&B Server をデプロイ

オンプレミスのインフラストラクチャーに W&B Server を設定するには、いくつかのインフラストラクチャーコンポーネントを構成する必要があります。そのコンポーネントには以下が含まれますが、これに限定されません：

- （強く推奨）Kubernetes クラスター
- MySQL 8 データベースクラスター
- Amazon S3 互換オブジェクトストレージ
- Redis キャッシュクラスター

オンプレミスのインフラストラクチャーに W&B Server をインストールする方法の詳細については、[Install on on-prem infrastructure](../self-managed/bare-metal.md) を参照してください。W&B は、さまざまなコンポーネントに関する推奨事項を提供し、インストールプロセス全体をサポートします。

## カスタムクラウドプラットフォームに W&B Server をデプロイ

W&B Server を AWS、GCP、または Azure 以外のクラウドプラットフォームにデプロイすることができます。その要件は [オンプレミスのインフラストラクチャー](#deploy-wb-server-in-on-prem-infrastructure) にデプロイする場合と同様です。

## W&B Server ライセンスの取得

W&B サーバーの設定を完了するには W&B トライアルライセンスが必要です。無料トライアルライセンスを生成するには、[Deploy Manager](https://deploy.wandb.ai/deploy) を開いてください。

:::note
W&B アカウントをお持ちでない場合は、無料ライセンスを生成するためにアカウントを作成する必要があります。
:::

URL は **Get a License for W&B Local** フォームにリダイレクトされます。以下の情報を提供してください：

1. **Choose Platform** ステップでデプロイメントタイプを選択します。
2. **Basic Information** ステップでライセンスの所有者を選択するか、新しい組織を追加します。
3. **Get a License** ステップでインスタンスの名前を **Name of Instance** フィールドに入力し、オプションで **Description** フィールドに説明を入力します。
4. **Generate License Key** ボタンを選択します。

デプロイメントの概要とインスタンスに関連付けられたライセンスを含むページが表示されます。

:::info
重要なセキュリティおよびその他のエンタープライズ向け機能のサポートを含む W&B Server のエンタープライズライセンスが必要な場合は、[このフォームを送信](https://wandb.ai/site/for-enterprise/self-hosted-trial) するか、W&B チームにご連絡ください。
:::