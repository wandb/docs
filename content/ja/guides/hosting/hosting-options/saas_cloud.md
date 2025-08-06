---
title: W&B マルチテナントクラウドを利用する
menu:
  default:
    identifier: saas_cloud
    parent: deployment-options
weight: 1
---

W&B Multi-tenant Cloud は、W&B の Google Cloud Platform (GCP) アカウント上にデプロイされた完全マネージド型のプラットフォームです。[GCP の北米リージョン](https://cloud.google.com/compute/docs/regions-zones)で稼働しています。W&B Multi-tenant Cloud は、GCP のオートスケーリング機能を活用して、トラフィックの増減に合わせてプラットフォームが自動的にスケールする設計となっています。

{{< img src="/images/hosting/saas_cloud_arch.png" alt="Multi-tenant Cloud アーキテクチャーダイアグラム" >}}

W&B Multi-tenant Cloud は、ご利用組織のニーズに合わせてスケールし、1 プロジェクトあたり最大 250,000 のメトリクスと、各メトリクスあたり最大 100 万件のデータポイントのログ記録に対応しています。さらに大規模なデプロイメントが必要な場合は、[サポート](mailto:support@wandb.com)までご連絡ください。

## データセキュリティ

Free または Pro プランのユーザーの場合、すべてのデータは共有クラウドストレージのみに保存され、共有クラウドコンピュートサービスで処理されます。ご利用の料金プランによっては、ストレージ容量に制限がある場合があります。

Enterprise プランのユーザーは、[セキュアストレージコネクタを利用した BYOB (Bring Your Own Bucket)]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) を [チーム単位]({{< relref "/guides/hosting/data-security/secure-storage-connector.md#configuration-options" >}}) で設定し、モデル・Datasets などのファイルを保存できます。1つのバケットを複数のチームで共有したり、各 W&B Teams ごとにバケットを分けて利用することも可能です。BYOB をチームに設定しない場合、そのチームのデータは共有クラウドストレージに保存されます。

デプロイメントがご自身の組織ポリシーや [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) に準拠しているかどうかは、お客様側でご確認ください。

## アイデンティティおよびアクセス管理 (IAM)
Enterprise プランをご利用の場合、拡張されたアイデンティティおよびアクセス管理によって、W&B デプロイメントを安全かつ効率的に認証・認可できます。

* OIDC または SAML を利用した SSO 認証。組織向けに SSO を設定したい場合は、W&B チームまたはサポートにご連絡ください。
* 組織やチーム単位で[適切なユーザーロールの設定]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" >}})が可能です。
* [制限付き Projects]({{< relref "/guides/hosting/iam/access-management/restricted-projects.md" >}})を活用することで、どのユーザーが W&B Project を閲覧・編集し、W&B Runs を送信できるかスコープを制限できます。

## モニタリング
組織の管理者は、アカウント画面の `Billing` タブから利用状況や請求を管理できます。Multi-tenant Cloud で共有クラウドストレージを使用している場合、管理者は組織内の異なるチーム間でストレージ利用を最適化できます。

## メンテナンス
W&B Multi-tenant Cloud は、マルチテナント対応のフルマネージドプラットフォームです。W&B が運用管理しているため、ユーザー自身が W&B プラットフォームのプロビジョニングや保守作業、追加コストなどの負担を負う必要はありません。

## コンプライアンス
Multi-tenant Cloud に対するセキュリティ管理は、社内および外部によって定期的に監査されています。SOC2 レポートやその他のセキュリティ・コンプライアンス関連資料は、[W&B Security Portal](https://security.wandb.ai/) からご請求いただけます。

## 次のステップ
[Multi-tenant Cloud へ直接アクセス](https://wandb.ai)して、ほとんどの機能を無料で始められます。より高度なデータセキュリティや IAM 機能をお試しになりたい場合は、[Enterprise トライアルをリクエスト](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial)してください。