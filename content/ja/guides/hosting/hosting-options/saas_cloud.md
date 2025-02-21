---
title: Use W&B Multi-tenant SaaS
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-saas_cloud
    parent: deployment-options
weight: 1
---

W&B Multi-tenant Cloud は、W&B の Google Cloud Platform (GCP) アカウント内にデプロイされた完全に管理されたプラットフォームで、[GPC の北アメリカ地域](https://cloud.google.com/compute/docs/regions-zones)で運用されています。W&B Multi-tenant Cloud は GCP の自動スケーリングを利用し、トラフィックの増減に基づいてプラットフォームが適切にスケールすることを保証します。

{{< img src="/images/hosting/saas_cloud_arch.png" alt="" >}}

## データ セキュリティ

エンタープライズプラン以外のユーザーの場合、すべてのデータは共有クラウドストレージにのみ保存され、共有クラウドコンピュートサービスで処理されます。ご利用の料金プランによっては、ストレージの制限が適用される場合があります。

エンタープライズプランのユーザーは、[安全なストレージコネクタを使用して独自のバケット (BYOB) を持ち込む]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})ことができ、[チームレベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}})でモデルやデータセットなどのファイルを保存します。複数のチームで単一のバケットを設定することもできますし、異なる W&B Teams 用に個別のバケットを使用することもできます。チーム用の安全なストレージコネクタを設定しない場合、そのデータは共有クラウドストレージに保存されます。

## アイデンティティおよびアクセス管理 (IAM)
エンタープライズプランの場合、W&B Organizationでの安全な認証と効果的な認可のために、アイデンティティおよびアクセス管理機能を利用できます。Multi-tenant Cloud における IAM の次の機能が利用可能です：

* OIDC または SAML を使用した SSO 認証。組織向けに SSO を設定する場合は、W&B チームまたはサポートに連絡してください。
* [適切なユーザー ロールを設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}}) し、組織およびチーム内での役割の範囲を決定。
* [制限されたプロジェクト]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を使用して、W&B プロジェクトの範囲を定義し、誰がそれに W&B runs を表示、編集、提出できるかを制限。

## モニター
組織管理者は、アカウントビューの `Billing` タブからアカウントの使用状況と請求を管理できます。Multi-tenant Cloud の共有クラウドストレージを使用している場合、管理者は組織内の異なるチーム間でのストレージ使用量を最適化できます。

## メンテナンス
W&B Multi-tenant Cloud はマルチテナントの完全に管理されたプラットフォームです。W&B Multi-tenant Cloud は W&B によって管理されているため、W&B プラットフォームのプロビジョニングおよびメンテナンスにかかるオーバーヘッドとコストは発生しません。

## コンプライアンス
Multi-tenant Cloud のセキュリティコントロールは定期的に内部および外部で監査されています。SOC2レポートやその他のセキュリティおよびコンプライアンス文書をリクエストするには、[W&B Security Portal](https://security.wandb.ai/) を参照してください。

## 次のステップ
エンタープライズ以外の機能をお探しの場合は、[Multi-tenant Cloud に直接アクセス](https://wandb.ai) してください。エンタープライズプランを開始するには、[このフォーム](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial)にご記入ください。