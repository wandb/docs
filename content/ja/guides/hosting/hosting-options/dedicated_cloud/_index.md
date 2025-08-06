---
title: 専用クラウド
menu:
  default:
    identifier: dedicated-cloud
    parent: deployment-options
url: guides/hosting/hosting-options/dedicated_cloud
---

## W&B Dedicated Cloud でシングルテナントSaaSを利用する

W&B Dedicated Cloud は、W&Bの AWS、GCP、または Azure クラウドアカウントにデプロイされるシングルテナントのフルマネージドプラットフォームです。各 Dedicated Cloud インスタンスは、他の Dedicated Cloud インスタンスとネットワーク、コンピューティング、ストレージが完全に分離されています。W&B固有のメタデータやデータは、分離されたクラウドストレージに保存され、分離されたクラウドコンピューティングサービスによって処理されます。

W&B Dedicated Cloud は [各クラウドプロバイダーごとの複数のグローバルリージョン]({{< relref "./dedicated_regions.md" >}}) で利用可能です。

## データセキュリティ

[セキュアストレージコネクター]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) を利用し、[インスタンスレベルとチームレベル]({{< relref "/guides/hosting/data-security/secure-storage-connector.md#configuration-options" >}}) で自身のバケット（BYOB）を持ち込むことで、モデルやデータセットなどのファイルを保存できます。

W&B Multi-tenant Cloud と同様に、1つのバケットを複数のチームで共有したり、チームごとにバケットを分離して利用したりできます。特定チームでセキュアストレージコネクターを設定しない場合、そのデータはインスタンスレベルのバケットに保存されます。

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="Dedicated Cloud architecture diagram" >}}

セキュアストレージコネクターによる BYOB に加えて、[IP アロウリスティング]({{< relref "/guides/hosting/data-security/ip-allowlisting.md" >}}) を使うことで、Dedicated Cloud インスタンスへの アクセス を信頼できるネットワークからのみに制限できます。

[クラウドプロバイダーのセキュア接続ソリューション]({{< relref "/guides/hosting/data-security/private-connectivity.md" >}}) を利用して、Dedicated Cloudインスタンスへプライベート接続することも可能です。

ご利用にあたっては、お使いの組織ポリシーおよび [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) など関連するガイドラインへの準拠が必要となります。

## アイデンティティおよびアクセス管理 (IAM)

W&B Organization 内で安全な認証と効果的な認可のために IAM 機能をご利用いただけます。 Dedicated Cloud インスタンスの IAM で利用できる主な機能は以下の通りです。

* [OpenID Connect (OIDC) を利用した SSO]({{< relref "/guides/hosting/iam/authentication/sso.md" >}}) または [LDAP]({{< relref "/guides/hosting/iam/authentication/ldap.md" >}}) で認証する
* [ユーザー役割の適切な設定]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" >}}) を組織やチームの範囲で行う
* [制限付きプロジェクト]({{< relref "/guides/hosting/iam/access-management/restricted-projects.md" >}}) によって、W&Bプロジェクトの閲覧・編集・Run投稿の権限範囲を制御する
* [アイデンティティフェデレーション]({{< relref "/guides/hosting/iam/authentication/identity_federation.md" >}}) で JSON Web Token を活用し、W&B API へアクセスする

## モニタリング

[Audit logs]({{< relref "/guides/hosting/monitoring-usage/audit-logging.md" >}}) を使ってチーム内のユーザーアクティビティを記録・追跡し、エンタープライズのガバナンス要件に対応できます。また、Dedicated Cloud インスタンスでは [W&B Organization Dashboard]({{< relref "/guides/hosting/monitoring-usage/org_dashboard.md" >}}) で組織の利用状況を確認できます。

## メンテナンス

W&B Multi-tenant Cloud と同様、Dedicated Cloud でも W&B プラットフォームのプロビジョニングや保守にともなう手間やコストは発生しません。

Dedicated Cloud のアップデート管理方法は [サーバーリリースプロセス]({{< relref "/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" >}}) をご参照ください。

## コンプライアンス

W&B Dedicated Cloud のセキュリティコントロールは、内部および外部で定期的に監査されています。製品評価のために必要なセキュリティ・コンプライアンス文書は [W&B Security Portal](https://security.wandb.ai/) からリクエスト可能です。

## マイグレーションオプション

[Self-Managed インスタンス]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) または [Multi-tenant Cloud]({{< relref "../saas_cloud.md" >}}) から Dedicated Cloud への移行も特定の制限やマイグレーション関連条件のもとでサポートしています。

## 次のステップ

Dedicated Cloud の利用をご希望の場合は、[こちらのフォーム](https://wandb.ai/site/for-enterprise/dedicated-saas-trial)よりご連絡ください。