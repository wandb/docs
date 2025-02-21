---
title: Use W&B Multi-tenant SaaS
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-saas_cloud
    parent: deployment-options
weight: 1
---

W&B Multi-tenant Cloud は、W&B の Google Cloud Platform (GCP) アカウントの [GPC の北米リージョン](https://cloud.google.com/compute/docs/regions-zones) にデプロイされた、フルマネージドのプラットフォームです。W&B Multi-tenant Cloud は GCP の自動スケーリングを利用して、トラフィックの増減に基づいてプラットフォームが適切にスケールするようにします。

{{< img src="/images/hosting/saas_cloud_arch.png" alt="" >}}

## データセキュリティ

エンタープライズプラン以外の ユーザー の場合、すべての データ は共有 クラウド ストレージにのみ保存され、共有 クラウド コンピューティングサービスで処理されます。料金プランによっては、ストレージ制限が適用される場合があります。

エンタープライズプランの ユーザー は、[セキュアストレージコネクタを使用して、独自の バケット (BYOB) を持ち込む]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) ことができます。これは、[Team レベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}}) で、 model 、 dataset などのファイルを保存するために行います。複数の Team で 1 つの バケット を設定することも、異なる W&B Teams で個別の バケット を使用することもできます。Team に対してセキュアストレージコネクタを設定しない場合、その データ は共有 クラウド ストレージに保存されます。

## ID と アクセス 管理 (IAM)
エンタープライズプランをご利用の場合、W&B Organization でセキュアな認証と効果的な認可のために、ID と アクセス 管理機能を使用できます。Multi-tenant Cloud の IAM では、次の機能が利用可能です。

*   OIDC または SAML による SSO 認証。Organization の SSO を設定する場合は、W&B Team またはサポートにお問い合わせください。
*   Organization のスコープ内および Team 内で、[適切な ユーザー ロールを設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}}) します。
*   W&B project のスコープを定義して、W&B の run を表示、編集、送信できる ユーザー を [制限付き project]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) で制限します。

## モニター
Organization 管理者は、アカウントビューの「Billing」タブからアカウントの使用状況と請求を管理できます。Multi-tenant Cloud で共有 クラウド ストレージを使用している場合、管理者は Organization 内の異なる Team 間でストレージ使用量を最適化できます。

## メンテナンス
W&B Multi-tenant Cloud は、マルチテナントのフルマネージド プラットフォーム です。W&B Multi-tenant Cloud は W&B によって管理されているため、W&B プラットフォーム のプロビジョニングとメンテナンスのオーバーヘッドとコストは発生しません。

## コンプライアンス
Multi-tenant Cloud のセキュリティ制御は、社内外で定期的に監査されます。SOC2 レポートおよびその他のセキュリティとコンプライアンスに関するドキュメントをリクエストするには、[W&B Security Portal](https://security.wandb.ai/) を参照してください。

## 次のステップ
エンタープライズ機能をお探しの場合は、[Multi-tenant Cloud に直接アクセス](https://wandb.ai) してください。エンタープライズプランを開始するには、[このフォーム](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial) を送信してください。
