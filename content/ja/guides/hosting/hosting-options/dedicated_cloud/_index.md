---
title: Dedicated Cloud
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-_index
    parent: deployment-options
url: guides/hosting/hosting-options/dedicated_cloud
---

## 専用クラウド (シングルテナント SaaS) の利用

W&B 専用クラウドは、W&B の AWS、GCP、または Azure クラウドアカウントにデプロイされる、シングルテナントでフルマネージドな platform です。各 Dedicated Cloud インスタンスは、他の W&B Dedicated Cloud インスタンスから独立したネットワーク、コンピューティング、ストレージを持っています。お客様の W&B 固有の metadata と data は、独立したクラウドストレージに保存され、独立したクラウドコンピューティングサービスを使用して処理されます。

W&B 専用クラウドは、[各クラウドプロバイダーの複数のグローバルリージョン]({{< relref path="./dedicated_regions.md" lang="ja" >}})で利用可能です。

## data セキュリティ

[セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用して、[インスタンスおよび Teams レベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}})で独自のバケット (BYOB) を持ち込み、 models、datasets などのファイル を保存できます。

W&B マルチテナント Cloud と同様に、複数の Teams に対して単一のバケットを構成することも、異なる Teams に対して個別のバケットを使用することもできます。Team に対してセキュアストレージコネクタを構成しない場合、その data はインスタンスレベルのバケットに保存されます。

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="" >}}

セキュアストレージコネクタを使用した BYOB に加えて、[IP アドレス許可リスト]({{< relref path="/guides/hosting/data-security/ip-allowlisting.md" lang="ja" >}})を利用して、信頼できるネットワークロケーションからのみ Dedicated Cloud インスタンスへの access を制限できます。

[クラウドプロバイダーのセキュアな接続ソリューション]({{< relref path="/guides/hosting/data-security/private-connectivity.md" lang="ja" >}})を使用して、Dedicated Cloud インスタンスにプライベートに接続することもできます。

## ID と access 管理 (IAM)

W&B Organization でセキュアな認証と効果的な承認を行うために、ID と access 管理機能を使用します。Dedicated Cloud インスタンスの IAM では、次の機能を利用できます。

* [OpenID Connect (OIDC) を使用した SSO]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ja" >}})または [LDAP]({{< relref path="/guides/hosting/iam/authentication/ldap.md" lang="ja" >}})で認証します。
* organization のスコープ内および Team 内で[適切な user ロールを構成]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}})します。
* W&B project のスコープを定義して、[制限付き Projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})を使用して、誰が W&B runs を表示、編集、送信できるかを制限します。
* [ID フェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}})で JSON Web Tokens を利用して、W&B APIs に access します。

## モニター

[監査 ログ]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md" lang="ja" >}})を使用して、Teams 内の user アクティビティを追跡し、エンタープライズガバナンスの要件に準拠します。また、[W&B Organization Dashboard]({{< relref path="/guides/hosting/monitoring-usage/org_dashboard.md" lang="ja" >}})で Dedicated Cloud インスタンスの organization の使用状況を表示できます。

## メンテナンス

W&B マルチテナント Cloud と同様に、Dedicated Cloud では W&B platform のプロビジョニングとメンテナンスのオーバーヘッドとコストが発生しません。

W&B が Dedicated Cloud でのアップデートをどのように管理するかを理解するには、[server リリース process]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ja" >}})を参照してください。

## コンプライアンス

W&B Dedicated Cloud のセキュリティ制御は、定期的 に内部および外部で監査されます。製品評価の演習に関するセキュリティおよびコンプライアンスドキュメントをリクエストするには、[W&B Security Portal](https://security.wandb.ai/)を参照してください。

## 移行オプション

[自己管理インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}})または [マルチテナント Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}})から Dedicated Cloud への移行がサポートされています。

## 次のステップ

Dedicated Cloud の使用にご興味がある場合は、[このフォーム](https://wandb.ai/site/for-enterprise/dedicated-saas-trial)を送信してください。
