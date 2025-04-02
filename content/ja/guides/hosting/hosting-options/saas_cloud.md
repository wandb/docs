---
title: Use W&B Multi-tenant SaaS
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-saas_cloud
    parent: deployment-options
weight: 1
---

W&B Multi-tenant Cloud は、W&B の Google Cloud Platform (GCP) アカウント内の [GPC の北米リージョン](https://cloud.google.com/compute/docs/regions-zones) にデプロイされた、フルマネージドのプラットフォームです。 W&B Multi-tenant Cloud は、GCP の自動スケーリングを利用して、トラフィックの増減に基づいてプラットフォームが適切にスケーリングされるようにします。

{{< img src="/images/hosting/saas_cloud_arch.png" alt="" >}}

## データセキュリティ

エンタープライズプラン以外のユーザーの場合、すべての data は共有クラウドストレージにのみ保存され、共有クラウドコンピューティングサービスで処理されます。料金プランによっては、ストレージ制限が適用される場合があります。

エンタープライズプランのユーザーは、[セキュアストレージコネクタを使用して、独自の bucket (BYOB) を持ち込む]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) ことができます。[team level]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}}) で、model、datasets などのファイルを保存できます。複数の Teams に対して 1 つの bucket を設定することも、異なる W&B Teams に対して個別の buckets を使用することもできます。team に対してセキュアストレージコネクタを設定しない場合、その data は共有クラウドストレージに保存されます。

## Identity and access management (IAM)
エンタープライズプランをご利用の場合、W&B Organization でセキュアな認証と効果的な承認のために、identity and access managements 機能を使用できます。 Multi-tenant Cloud の IAM では、次の機能が利用可能です。

* OIDC または SAML による SSO 認証。 Organization の SSO を設定する場合は、W&B team またはサポートにお問い合わせください。
* Organization のスコープ内および team 内で、[適切な user ロールを設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}}) します。
* W&B project のスコープを定義して、[制限付き projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) で、誰が W&B runs を表示、編集、送信できるかを制限します。

## モニター
Organization 管理者は、アカウントビューの [Billing] タブから、アカウントの使用状況と請求を管理できます。 Multi-tenant Cloud 上の共有クラウドストレージを使用している場合、管理者は organization 内の異なる Teams 間でストレージ使用量を最適化できます。

## メンテナンス
W&B Multi-tenant Cloud は、マルチテナントのフルマネージドプラットフォームです。 W&B Multi-tenant Cloud は W&B によって管理されているため、W&B プラットフォームのプロビジョニングとメンテナンスのオーバーヘッドとコストは発生しません。

## コンプライアンス
Multi-tenant Cloud のセキュリティコントロールは、定期的 内部および外部で監査されます。 SOC2 レポートおよびその他のセキュリティとコンプライアンスに関するドキュメントをリクエストするには、[W&B Security Portal](https://security.wandb.ai/) を参照してください。

## 次のステップ
エンタープライズ機能以外をお探しの場合は、[Multi-tenant Cloud に直接アクセス](https://wandb.ai) してください。エンタープライズプランを開始するには、[このフォーム](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial) を送信してください。
