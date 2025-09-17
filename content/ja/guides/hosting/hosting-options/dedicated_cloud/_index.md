---
title: 専用クラウド
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-_index
    parent: deployment-options
url: guides/hosting/hosting-options/dedicated_cloud
---

## W&B Dedicated Cloud をシングルテナント SaaS として使用する
W&B Dedicated Cloud は、W&B の AWS、GCP、Azure クラウド アカウントにデプロイされるシングルテナントのフルマネージドプラットフォームです。各 Dedicated Cloud インスタンスは、他の W&B Dedicated Cloud インスタンスから分離された独自のネットワーク、コンピューティング、ストレージを備えています。お客様の W&B 固有のメタデータとデータは、分離されたクラウド ストレージに保存され、分離されたクラウド コンピューティング サービスを使用して処理されます。
W&B Dedicated Cloud は、[各クラウド プロバイダーの複数のグローバル リージョン]({{< relref path="./dedicated_regions.md" lang="ja" >}}) で利用できます。

## データ セキュリティ
[セキュア ストレージ コネクター]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を [インスタンスおよびチーム レベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}}) で使用して、独自のバケット (BYOB) を持ち込み、Models、Datasets などのファイルを保存できます。
W&B マルチテナント クラウドと同様に、複数のチームに対して単一のバケットを設定することも、異なるチームに個別のバケットを使用することもできます。チームにセキュア ストレージ コネクターを設定しない場合、そのデータはインスタンス レベルのバケットに保存されます.
{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="Dedicated Cloud アーキテクチャ図" >}}
セキュア ストレージ コネクターを使用した BYOB に加えて、[IP 許可リスト]({{< relref path="/guides/hosting/data-security/ip-allowlisting.md" lang="ja" >}}) を使用して、信頼できるネットワークの場所からのみ Dedicated Cloud インスタンスへのアクセスを制限できます。
[クラウド プロバイダーのセキュア接続ソリューション]({{< relref path="/guides/hosting/data-security/private-connectivity.md" lang="ja" >}}) を使用して、Dedicated Cloud インスタンスにプライベートに接続できます。
該当する場合は、お客様のデプロイメントが組織のポリシーおよび [セキュリティ技術実装ガイドライン (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide) に準拠していることを確認する責任はお客様にあります。

## ID およびアクセス管理 (IAM)
W&B Organization でのセキュアな認証と効果的な認可のために、ID およびアクセス管理機能を使用します。Dedicated Cloud インスタンスの IAM で利用できる機能は次のとおりです。
*   [OpenID Connect (OIDC) を使用した SSO]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ja" >}}) または [LDAP]({{< relref path="/guides/hosting/iam/authentication/ldap.md" lang="ja" >}}) で認証します。
*   組織の範囲内およびチーム内で、[適切なユーザー ロールを設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}}) します。
*   [制限付き Projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を使用して、W&B Project の範囲を定義し、誰がそれを表示、編集、および W&B Runs を送信できるかを制限します。
*   [ID フェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}}) と JSON Web トークンを活用して W&B API にアクセスします。

## 監視
[監査ログ]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md" lang="ja" >}}) を使用して、チーム内のユーザー アクティビティを追跡し、企業のガバナンス要件に準拠します。また、[W&B Organization ダッシュボード]({{< relref path="/guides/hosting/monitoring-usage/org_dashboard.md" lang="ja" >}}) を使用して、Dedicated Cloud インスタンスでの組織の使用状況を表示できます。

## メンテナンス
W&B マルチテナント クラウドと同様に、Dedicated Cloud を使用すると、W&B プラットフォームのプロビジョニングとメンテナンスのオーバーヘッドとコストが発生しません。
W&B が Dedicated Cloud でアップデートを管理する方法を理解するには、[サーバー リリース プロセス]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ja" >}}) を参照してください。

## コンプライアンス
W&B Dedicated Cloud のセキュリティ コントロールは、定期的に内部および外部で監査されます。製品評価用のセキュリティおよびコンプライアンス ドキュメントをリクエストするには、[W&B Security Portal](https://security.wandb.ai/) を参照してください。

## 移行オプション
[Self-Managed インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) または [マルチテナント クラウド]({{< relref path="../saas_cloud.md" lang="ja" >}}) から Dedicated Cloud への移行は、特定の制限および移行関連の制約の対象となりますが、サポートされています。

## 次のステップ
Dedicated Cloud の使用にご興味がある場合は、[こちらのフォーム](https://wandb.ai/site/for-enterprise/dedicated-saas-trial) をご提出ください。