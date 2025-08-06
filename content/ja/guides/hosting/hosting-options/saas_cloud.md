---
title: W&B マルチテナントクラウドを利用する
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-saas_cloud
    parent: deployment-options
weight: 1
---

W&B Multi-tenant Cloud は、W&B の Google Cloud Platform (GCP) アカウント上にデプロイされたフルマネージド型のプラットフォームです。[GCP の北米リージョン](https://cloud.google.com/compute/docs/regions-zones)で稼働しています。W&B Multi-tenant Cloud は、GCP のオートスケーリング機能を活用し、トラフィックの増減に応じてプラットフォームのスケールを自動で調整します。

{{< img src="/images/hosting/saas_cloud_arch.png" alt="Multi-tenant Cloud のアーキテクチャ図" >}}

W&B Multi-tenant Cloud は組織のニーズに合わせてスケールし、1 プロジェクトあたり 250,000 件のメトリクス、各メトリクスで最大 100 万件のデータ ポイントを記録できます。より大規模なデプロイメントを希望される場合は、[サポート](mailto:support@wandb.com)までお問い合わせください。

## データセキュリティ

Free または Pro プランのユーザーの場合、すべてのデータは共有クラウドストレージにのみ保存され、共有クラウド計算サービスによって処理されます。ご利用のプランによっては、ストレージの上限が適用されることがあります。

Enterprise プランのユーザーは、[セキュアストレージコネクターを利用して独自のバケット（BYOB）をチーム単位で設定]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) し、[設定オプション]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}})に従って、モデルやデータセットなどのファイルを保存可能です。1 つのバケットを複数の Teams で共有することも、Teams ごとにバケットを分けることも可能です。チームに対して BYOB を設定しない場合、そのチームのデータは共有クラウドストレージに保存されます。

ご自身のデプロイメントが、組織のポリシーや[セキュリティ技術実装ガイドライン（STIG）](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide)に準拠していることを確認してください（該当する場合）。

## アイデンティティとアクセス管理（IAM）

Enterprise プランをご利用の場合、W&B デプロイメントのセキュアな認証と効率的な認可のために、高度なアイデンティティおよびアクセス管理機能が利用できます。

* OIDC または SAML による SSO 認証。組織で SSO 構成をご希望の場合は、W&B チームまたはサポートまでご連絡ください。
* [適切なユーザーロールの設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}})：組織全体やチームごとにユーザーのロールを設定できます。
* [制限付きプロジェクト]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})で W&B プロジェクトのスコープを定義し、誰が参照・編集・run の実行が可能かを制限できます。

## モニタリング

組織管理者は、アカウント画面の `Billing` タブからアカウントの使用状況や請求を管理できます。Multi-tenant Cloud 上の共有クラウドストレージを利用している場合は、管理者が組織内の各 Teams 間でストレージ使用量を最適化することが可能です。

## メンテナンス

W&B Multi-tenant Cloud は、マルチテナント型のフルマネージドプラットフォームです。W&B によって運用・管理されるため、ユーザーは W&B プラットフォームのプロビジョニングや保守にかかる負担やコストを負う必要はありません。

## コンプライアンス

Multi-tenant Cloud のセキュリティ管理は、定期的に社内外で監査が行われています。[W&B Security Portal](https://security.wandb.ai/) を参照し、SOC2 レポートやその他のセキュリティ／コンプライアンス文書をご請求ください。

## 次のステップ

[Multi-tenant Cloud へ直接アクセス](https://wandb.ai)し、ほとんどの機能を無料で利用開始できます。データセキュリティや IAM 機能を強化した Enterprise 向けも体験したい場合は、[Enterprise トライアルをリクエスト](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial)してください。