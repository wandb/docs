---
title: Dedicated Cloud
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-_index
    parent: deployment-options
url: guides/hosting/hosting-options/dedicated_cloud
---

## 専用クラウドの使用（シングルテナント SaaS）

W&B Dedicated Cloud は、W&B の AWS、GCP、または Azure クラウド アカウントに展開されるシングルテナントの完全管理プラットフォームです。各 Dedicated Cloud インスタンスは他の W&B Dedicated Cloud インスタンスから隔離されたネットワーク、コンピュート、ストレージを持っています。W&B 固有のメタデータとデータは、隔離されたクラウド ストレージに保存され、隔離されたクラウド コンピュート サービスを使用して処理されます。

W&B Dedicated Cloud は、[複数のクラウド プロバイダーのグローバル リージョン](./dedicated_regions.md) で利用可能です。

## データセキュリティ

[セキュア ストレージ コネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}}) を使用して、[インスタンスとチームレベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}}) でファイル（Models、Datasets など）を保存するための独自のバケット (BYOB) を持ち込むことができます。

W&B マルチテナント クラウドと同様に、1 つのバケットを複数のチームで共有したり、異なるチーム用に個別のバケットを使用したりすることができます。 セキュア ストレージ コネクタをチーム用に設定しない場合、そのデータはインスタンスレベルのバケットに保存されます。

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="" >}}

セキュア ストレージ コネクタによる BYOB に加えて、[IP 許可リスト]({{< relref path="/guides/hosting/data-security/ip-allowlisting.md" lang="ja" >}}) を利用して、信頼できるネットワーク ロケーションからのみ専用クラウド インスタンスへのアクセスを制限できます。

また、[クラウド プロバイダーのセキュアな接続ソリューション]({{< relref path="/guides/hosting/data-security/private-connectivity.md" lang="ja" >}}) を使用して、専用クラウドインスタンスにプライベートに接続することもできます。

## アイデンティティとアクセス管理 (IAM)

W&B 組織内で安全な認証と効果的な承認を行うために、アイデンティティとアクセス管理機能を使用します。 専用クラウドインスタンスで利用可能な IAM 機能は次のとおりです。

* [OpenID Connect (OIDC) を使用した SSO]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ja" >}}) または [LDAP]({{< relref path="/guides/hosting/iam/authentication/ldap.md" lang="ja" >}}) で認証します。
* 組織の範囲とチーム内での[適切なユーザー ロールを設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}})します。
* [制限付き Projects]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}}) を使用して、W&B Projects の範囲を定義し、誰が W&B Runs に表示、編集、提出できるかを制限します。
* W&B API にアクセスするために、[アイデンティティ フェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}}) と JSON Web トークンを活用する。

## モニター

[Audit logs]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md" lang="ja" >}}) を使用してチーム内のユーザー アクティビティを追跡し、エンタープライズ ガバナンスの要件に準拠します。また、専用クラウド インスタンスでの W&B 組織の使用状況を[W&B Organization Dashboard]({{< relref path="/guides/hosting/monitoring-usage/org_dashboard.md" lang="ja" >}}) で確認できます。

## メンテナンス

W&B マルチテナント クラウドと同様に、Dedicated Cloud を使用することで W&B プラットフォームのプロビジョニングと保守のオーバーヘッドやコストを負担することはありません。

Dedicated Cloud の更新が W&B によってどのように管理されているかを理解するには、[サーバー リリース プロセス]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ja" >}}) を参照してください。

## コンプライアンス

W&B Dedicated Cloud のセキュリティ コントロールは、定期的に内部および外部で監査されます。製品評価演習のためのセキュリティおよびコンプライアンス文書をリクエストするには、[W&B セキュリティ ポータル](https://security.wandb.ai/) を参照してください。

## 移行オプション

専用クラウドへの移行は、[自己管理インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) または [マルチテナントクラウド]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ja" >}}) からサポートされています。

## 次のステップ

専用クラウドの使用に興味がある場合は、[このフォーム](https://wandb.ai/site/for-enterprise/dedicated-saas-trial) を提出してください。