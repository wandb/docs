---
title: Self-managed
description: W&B を プロダクション 環境にデプロイする
cascade:
- url: guides/hosting/self-managed/:filename
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
url: guides/hosting/hosting-options/self-managed
---

## セルフマネージドクラウドまたはオンプレミス の インフラストラクチャー を使用する

{{% alert %}}
W&B では、[W&B Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ja" >}}) や [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) の デプロイメント タイプなど、フルマネージド の デプロイメント オプションを推奨しています。W&B の フルマネージド サービスは、シンプルで安全に使用でき、設定は最小限で済みます。
{{% /alert %}}

[AWS 、GCP 、または Azure クラウド アカウント]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ja" >}}) 、または [オンプレミス の インフラストラクチャー]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) 内に W&B Server を デプロイ します。

IT / DevOps / MLOps チームは、 デプロイメント のプロビジョニング、アップグレードの管理、およびセルフマネージド の W&B Server インスタンス の継続的なメンテナンスを担当します。

## セルフマネージドクラウド アカウント 内に W&B Server を デプロイ する

W&B では、公式の W&B Terraform スクリプトを使用して、W&B Server を AWS 、GCP 、または Azure クラウド アカウント に デプロイ することを推奨しています。

W&B Server を [AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ja" >}}), [GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ja" >}}) or [Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ja" >}}) に設定する方法については、特定のクラウド プロバイダー の ドキュメント を参照してください。

## オンプレミス の インフラストラクチャー に W&B Server を デプロイ する

オンプレミス の インフラストラクチャー に W&B Server を設定するには、いくつかの インフラストラクチャー コンポーネント を構成する必要があります。これらのコンポーネント には、以下が含まれますが、これらに限定されません。

- (強く推奨) Kubernetes cluster
- MySQL 8 database cluster
- Amazon S3 互換 object storage
- Redis cache cluster

オンプレミス の インフラストラクチャー に W&B Server を インストール する方法の詳細については、[オンプレミス の インフラストラクチャー への インストール]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ja" >}}) を参照してください。W&B は、さまざまなコンポーネント に関する推奨事項を提供し、 インストール プロセス を通じてガイダンスを提供できます。

## カスタム クラウド プラットフォーム 上に W&B Server を デプロイ する

AWS 、GCP 、または Azure ではない クラウド プラットフォーム に W&B Server を デプロイ できます。そのための要件は、[オンプレミス の インフラストラクチャー での デプロイ]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ja" >}}) の場合と同様です。

## W&B Server ライセンス を取得する

W&B サーバー の 構成を完了するには、W&B trial ライセンス が必要です。[Deploy Manager](https://deploy.wandb.ai/deploy) を開いて、無料の trial ライセンス を生成します。

{{% alert %}}
まだ W&B アカウント をお持ちでない場合は、アカウント を作成して無料の ライセンス を生成してください。

重要なセキュリティ やその他の エンタープライズ 向けの機能のサポートを含む W&B Server の エンタープライズ ライセンス が必要な場合は、[このフォーム](https://wandb.ai/site/for-enterprise/self-hosted-trial) を送信するか、W&B チーム にお問い合わせください。
{{% /alert %}}

URL は、**W&B Local の ライセンス を取得** フォーム に リダイレクト されます。次の情報を提供してください。

1. **プラットフォーム を選択** ステップ で デプロイメント タイプ を選択します。
2. **基本情報** ステップ で ライセンス の所有者を選択するか、新しい 組織 を追加します。
3. **ライセンス を取得** ステップ の **インスタンス の名前** フィールド に インスタンス の名前を入力し、必要に応じて **説明** フィールド に説明を入力します。
4. **ライセンス キー を生成** ボタン を選択します。

デプロイメント の概要と、 インスタンス に関連付けられた ライセンス が表示された ページ が表示されます。
