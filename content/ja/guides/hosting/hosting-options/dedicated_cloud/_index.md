---
title: Dedicated Cloud
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-_index
    parent: deployment-options
url: guides/hosting/hosting-options/dedicated_cloud
---

## 専用クラウド (シングルテナントSaaS)

W&B 専用クラウド は、W&B の AWS、GCP、または Azure クラウドアカウントにデプロイされた、シングルテナントで完全に管理されたプラットフォームです。各 専用クラウド インスタンスは、他の W&B 専用クラウド インスタンスから独立したネットワーク、コンピューティング、ストレージを持っています。お客様の W&B 固有の メタデータ と データ は、独立したクラウドストレージに保存され、独立したクラウドコンピューティングサービスを使用して処理されます。

W&B 専用クラウド は、[各クラウドプロバイダーの複数のグローバルリージョン]({{< relref path="./dedicated_regions.md" lang="ja" >}})で利用可能です。

## データセキュリティ

[セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用して、[インスタンスおよび Team レベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}})で、お客様自身の バケット (BYOB) を持ち込み、モデル、データセット などのファイル を保存できます。

W&B マルチテナント Cloud と同様に、複数の Team に対して単一の バケット を構成するか、異なる Team に対して別々の バケット を使用できます。Team に対してセキュアストレージコネクタを構成しない場合、その データ はインスタンスレベルの バケット に保存されます。

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="" >}}

セキュアストレージコネクタによる BYOB に加えて、[IP 許可リスト]({{< relref path="/guides/hosting/data-security/ip-allowlisting.md" lang="ja" >}})を利用して、信頼できるネットワークロケーションからのみ 専用クラウド インスタンスへの アクセス を制限できます。

また、[クラウドプロバイダーのセキュアな接続ソリューション]({{< relref path="/guides/hosting/data-security/private-connectivity.md" lang="ja" >}})を使用して、専用クラウド インスタンスにプライベートに接続することもできます。

## ID と アクセス 管理 (IAM)

W&B Organization でセキュアな認証と効果的な認可のために、ID と アクセス 管理機能を使用します。専用クラウド インスタンスの IAM では、次の機能が利用可能です。

* [OpenID Connect (OIDC) を使用した SSO]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ja" >}})または[LDAP]({{< relref path="/guides/hosting/iam/authentication/ldap.md" lang="ja" >}})で認証します。
* Organization のスコープ内および Team 内で[適切な ユーザー ロールを構成]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}})します。
* W&B プロジェクト のスコープを定義して、[制限付き プロジェクト]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})で、誰が W&B の run を表示、編集、および送信できるかを制限します。
* [ID フェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}})で JSON Web Tokens を利用して、W&B API に アクセス します。

## モニター

[監査 ログ]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md" lang="ja" >}})を使用して、Team 内の ユーザー アクティビティを追跡し、エンタープライズガバナンス要件に準拠します。また、[W&B Organization Dashboard]({{< relref path="/guides/hosting/monitoring-usage/org_dashboard.md" lang="ja" >}})で 専用クラウド インスタンスの Organization の使用状況を表示できます。

## メンテナンス

W&B マルチテナント Cloud と同様に、専用クラウド では W&B プラットフォーム のプロビジョニングとメンテナンスのオーバーヘッドとコストは発生しません。

W&B が 専用クラウド での更新をどのように管理するかを理解するには、[サーバー リリース プロセス]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ja" >}})を参照してください。

## コンプライアンス

W&B 専用クラウド のセキュリティコントロールは、定期的 に内部および外部で監査されます。製品評価の演習のためにセキュリティおよびコンプライアンスドキュメントをリクエストするには、[W&B Security Portal](https://security.wandb.ai/)を参照してください。

## 移行オプション

[自己管理インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}})または[マルチテナント Cloud]({{< relref path="../saas_cloud.md" lang="ja" >}})からの 専用クラウド への移行がサポートされています。

## 次のステップ

専用クラウド の使用にご興味がある場合は、[このフォーム](https://wandb.ai/site/for-enterprise/dedicated-saas-trial)を送信してください。
