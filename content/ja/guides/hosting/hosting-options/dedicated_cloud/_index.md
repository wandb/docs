---
title: 専用クラウド
menu:
  default:
    identifier: ja-guides-hosting-hosting-options-dedicated_cloud-_index
    parent: deployment-options
url: /ja/guides/hosting/hosting-options/dedicated_cloud
---

## 専用クラウドの利用 (シングルテナント SaaS)

W&B 専用クラウドは、シングルテナントで完全に管理されたプラットフォームであり、W&B の AWS、GCP または Azure クラウドアカウントにデプロイされます。各専用クラウドインスタンスは、他の W&B 専用クラウドインスタンスと隔離されたネットワーク、コンピュート、ストレージを持っています。W&B 固有のメタデータとデータは、隔離されたクラウドストレージに保存され、隔離されたクラウドコンピュートサービスを使って処理されます。

W&B 専用クラウドは[各クラウドプロバイダーにとって複数の世界各地域で利用可能です]({{< relref path="./dedicated_regions.md" lang="ja" >}})

## データセキュリティ

[セキュアストレージコネクタ]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ja" >}})を使用して、[インスタンスおよびチームレベル]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md#configuration-options" lang="ja" >}})で自分のバケット (BYOB) を持ち込むことで、モデル、データセットなどのファイルを保存できます。

W&B マルチテナントクラウドと同様に、複数のチーム用に 1 つのバケットを設定するか、異なるチームに対して別々のバケットを使用することができます。チーム用にセキュアストレージコネクタを設定しない場合、そのデータはインスタンスレベルのバケットに保存されます。

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="" >}}

セキュアストレージコネクタを使用した BYOB に加えて、信頼できるネットワーク場所からのみ専用クラウドインスタンスにアクセスを制限するために [IP 許可リスト]({{< relref path="/guides/hosting/data-security/ip-allowlisting.md" lang="ja" >}})を利用できます。

また、[クラウドプロバイダーのセキュア接続ソリューション]({{< relref path="/guides/hosting/data-security/private-connectivity.md" lang="ja" >}})を使用して、専用クラウドインスタンスにプライベートに接続することもできます。

## アイデンティティとアクセス管理 (IAM)

W&B 組織での安全な認証と効果的な認可のためのアイデンティティおよびアクセス管理機能を利用してください。専用クラウドインスタンスの IAM には、次の機能があります:

* [OpenID Connect (OIDC) を使用した SSO]({{< relref path="/guides/hosting/iam/authentication/sso.md" lang="ja" >}}) または [LDAP]({{< relref path="/guides/hosting/iam/authentication/ldap.md" lang="ja" >}}) で認証します。
* 組織の範囲およびチーム内で[適切なユーザー役割を設定]({{< relref path="/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" lang="ja" >}})します。
* [制限付きプロジェクト]({{< relref path="/guides/hosting/iam/access-management/restricted-projects.md" lang="ja" >}})で、誰が W&B プロジェクトを閲覧、編集、または W&B runs に送信できるかの範囲を定義します。
* JSON Web トークンを[アイデンティティフェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ja" >}})と組み合わせて利用し、W&B API にアクセスします。

## 監視

[監査ログ]({{< relref path="/guides/hosting/monitoring-usage/audit-logging.md" lang="ja" >}})を使用して、チーム内でのユーザー活動を追跡し、企業ガバナンス要件に準拠します。また、専用クラウドインスタンス内の W&B 組織ダッシュボードを使用して、組織の利用状況を確認できます。

## メンテナンス

W&B マルチテナントクラウドと同様に、専用クラウドを使用することで、W&B プラットフォームのプロビジョニングやメンテナンスに関するオーバーヘッドやコストを負担することはありません。

専用クラウドで W&B がどのようにアップデートを管理するかを理解するためには、[サーバーリリースプロセス]({{< relref path="/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" lang="ja" >}})を参照してください。

## コンプライアンス

W&B 専用クラウドのセキュリティ管理は、定期的に内部および外部で監査されています。セキュリティ評価演習のためのセキュリティとコンプライアンス文書をリクエストするには、[W&B セキュリティポータル](https://security.wandb.ai/)を参照してください。

## 移行オプション

[セルフマネージドインスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}})または[マルチテナントクラウド]({{< relref path="../saas_cloud.md" lang="ja" >}})から専用クラウドへの移行がサポートされています。

## 次のステップ

専用クラウドの利用に興味がある場合は、[このフォーム](https://wandb.ai/site/for-enterprise/dedicated-saas-trial)を提出してください。