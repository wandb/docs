---
displayed_sidebar: default
---


# 専用クラウド (シングルテナントSaaS)

W&B Dedicated Cloudは、W&BのAWS、GCP、またはAzureクラウドアカウントにデプロイされたシングルテナントのフルマネージドプラットフォームです。各Dedicated Cloudインスタンスは、他のW&B Dedicated Cloudインスタンスから隔離されたネットワーク、コンピュート、ストレージを持ちます。あなたのW&Bの固有のメタデータとデータは、隔離されたクラウドストレージに保存され、隔離されたクラウドコンピュートサービスを使用して処理されます。

W&B Dedicated Cloudは、[各クラウドプロバイダーの複数のグローバル地域で利用可能](./dedicated_regions.md)です。

## データセキュリティ
[セキュアストレージコネクタ](../data-security/secure-storage-connector.md)を使用して、[インスタンスおよびチームレベル](../data-security/secure-storage-connector.md#configuration-options)で独自のバケットを持ち込む (BYOB) ことができ、モデル、データセットなどのファイルを保存できます。

W&Bマルチテナントクラウドと同様に、単一のバケットを複数のチームで共有することも、異なるチームごとに別々のバケットを使用することもできます。チーム用にセキュアストレージコネクタを設定しない場合、そのデータはインスタンスレベルのバケットに保存されます。

![](/images/hosting/dedicated_cloud_arch.png)

セキュアストレージコネクタを使用したBYOBに加えて、特定のネットワーク場所からのみDedicated Cloudインスタンスへのアクセスを制限するために、[IP許可リスト](../data-security/ip-allowlisting.md)を利用できます。

また、[クラウドプロバイダーのセキュア接続ソリューション](../data-security/private-connectivity.md)を使用して、Dedicated Cloudインスタンスに接続することもできます。この機能は現在、[AWS PrivateLink](https://aws.amazon.com/privatelink/)を使用したAWSインスタンスのDedicated Cloudで利用可能です。

## アイデンティティとアクセス管理 (IAM)
W&B Organizationにおけるセキュアな認証と効果的な認可のために、アイデンティティとアクセス管理機能を使用します。Dedicated Cloudインスタンスで利用できるIAMの機能は次のとおりです:

* [OpenID Connect (OIDC)](../iam/sso.md) または [LDAP](../iam/ldap.md) を使用してSSOで認証します。
* 組織およびチーム内で適切な[ユーザーロールを設定](../iam/manage-users.md)します。
* [制限付きProjects](../iam/restricted-projects.md)を使用して、W&BのProjectsの可視者、編集者、およびRunsの提出者を限定する範囲を定義します。

## モニタリング
[監査ログ](../monitoring-usage/audit-logging.md)を使用してチーム内のユーザー活動を追跡し、企業ガバナンス要件に準拠します。また、Dedicated Cloudインスタンス内でW&B Organization Dashboardを使用して組織の使用状況を確認できます。

## メンテナンス
W&Bマルチテナントクラウドと同様に、Dedicated CloudではW&Bプラットフォームのプロビジョニングと保守のオーバーヘッドやコストを負担する必要がありません。

Dedicated Cloudでの更新管理については、[サーバーリリースプロセス](../server-release-process.md)を参照してください。

## コンプライアンス
W&B Dedicated Cloudのセキュリティ管理は内部および外部で定期的に監査されます。 SOC2レポートやその他のセキュリティおよびコンプライアンス文書を要求するには、[W&Bセキュリティポータル](https://security.wandb.ai/)を参照してください。

## 移行オプション
[セルフマネージドインスタンス](./self-managed.md) または[マルチテナントクラウド](./saas_cloud.md)からDedicated Cloudへの移行がサポートされています。

## 次のステップ
Dedicated Cloudの使用に関心がある場合は、[このフォーム](https://wandb.ai/site/for-enterprise/dedicated-saas-trial)にご記入ください。