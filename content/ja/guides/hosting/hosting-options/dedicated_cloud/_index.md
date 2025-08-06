---
title: 専用クラウド
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-_index
    parent: deployment-options
url: guides/hosting/hosting-options/dedicated_cloud
---

## シングルテナント SaaS 向け W&B 専用クラウドの利用

W&B Dedicated Cloud は、シングルテナントで完全に管理されたプラットフォームであり、W&B の AWS、GCP、または Azure クラウドアカウントにデプロイされます。各 Dedicated Cloud インスタンスは、他の W&B Dedicated Cloud インスタンスとは独立したネットワーク、コンピュート、ストレージを持ちます。W&B 特有のメタデータやデータは、隔離されたクラウドストレージに保存され、個別のクラウドコンピュートサービスで処理されます。

W&B Dedicated Cloud は [各クラウドプロバイダーごとに複数のグローバルリージョンで利用可能です]({{< relref path="./dedicated_regions.md" lang="ja" >}})

## データセキュリティ

[セキュアストレージコネクター]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を利用して [インスタンスおよびチームレベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}}) で BYOB（ご自身のバケット持込）を行い、モデルや Datasets などのファイルを保存できます。

W&B Multi-tenant Cloud と同様に、1 つのバケットを複数チームで共有することも、チームごとに個別のバケットを利用することも可能です。チームにセキュアストレージコネクターの設定がされていない場合、そのデータはインスタンスレベルのバケットに保存されます。

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="Dedicated Cloud architecture diagram" >}}

セキュアストレージコネクターによる BYOB に加え、[IP allowlisting]({{< relref path="/guides/hosting/data-security/ip-allowlisting.md" lang="ja" >}}) を利用することで信頼できるネットワークのみから Dedicated Cloud インスタンスへのアクセスを制限できます。

[クラウドプロバイダーのセキュア接続サービス]({{< relref path="/guides/hosting/data-security/private-connectivity.md" lang="ja" >}}) を使って、Dedicated Cloud インスタンスにプライベート接続することも可能です。

ご利用中のデプロイメントが自社のポリシーや [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) に準拠していることの確認は、お客様の責任となります。

## アイデンティティとアクセス管理（IAM）

W&B Organization 内で安全な認証と効果的な認可を行うため、アイデンティティとアクセス管理機能を利用できます。Dedicated Cloud インスタンスで利用可能な IAM 機能は以下の通りです：

* [OpenID Connect (OIDC) を使った SSO]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ja" >}}) や [LDAP]({{< relref path="/guides/hosting/iam/authentication/ldap.md" lang="ja" >}}) による認証
* 組織単位およびチーム単位での [適切なユーザーロールの設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}})
* W&B Project のスコープを定義することで、閲覧・編集・ run の実行権限を制限できる [restricted projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})
* [アイデンティティフェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}}) と JSON Web Token を活用し、W&B API へのアクセスを実現

## モニタリング

[Audit logs]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md" lang="ja" >}}) を用いて、チーム内のユーザー活動を追跡し、エンタープライズのガバナンス要件に対応できます。また、Dedicated Cloud インスタンスの [W&B Organization Dashboard]({{< relref path="/guides/hosting/monitoring-usage/org_dashboard.md" lang="ja" >}}) で組織の利用状況も可視化できます。

## メンテナンス

W&B Multi-tenant Cloud と同様に、Dedicated Cloud を利用することで W&B プラットフォームのプロビジョニングや保守にかかる手間やコストが不要となります。

W&B が Dedicated Cloud でどのようにアップデート管理を行うかについては、[サーバーリリースプロセス]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ja" >}}) をご参照ください。

## コンプライアンス

W&B Dedicated Cloud のセキュリティ管理は定期的に社内外で監査されています。セキュリティおよびコンプライアンス関連の資料が必要な場合は、[W&B セキュリティポータル](https://security.wandb.ai/) からご依頼ください。

## マイグレーションオプション

[Self-Managed instance]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) や [Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ja" >}}) から Dedicated Cloud へのマイグレーションも、特定の制限やマイグレーションに関する条件のもとでサポートされています。

## 次のステップ

Dedicated Cloud のご利用をご希望の場合は、[こちらのフォーム](https://wandb.ai/site/for-enterprise/dedicated-saas-trial) よりご連絡ください。